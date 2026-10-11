#!/usr/bin/env python3
"""
Voxtral-Mini-3B-2507 LoRA fine-tuning on IndicVoices (as / bn / hi / or)
v3 -- targets the Assamese / Odia collapse seen in earlier runs.

What's different vs the previous version
----------------------------------------
1. PER-SAMPLE LANGUAGE CONDITIONING via apply_chat_template
   Each sample gets a per-language English instruction
       "Transcribe the following <Language> audio word-for-word."
   instead of the broken `apply_transcription_request(language="hi", ...)`
   that forced the same system prompt for every sample. This is the
   primary cause of the random script-switching observed in the prior run.

2. PER-LANGUAGE LOSS REWEIGHTING
   Tekken byte-fallback inflates token counts for Odia (~3x) and Assamese
   (~2x) relative to Hindi, which dilutes their gradient. We compute a
   per-sample weight inversely proportional to the language's empirical
   bytes-per-character so under-served languages catch up.

3. HIGHER-CAPACITY LoRA + MLP TARGETS
   r=64, alpha=64, targets attention (q/k/v/o) AND MLP (gate/up/down).
   New script generation needs MLP capacity, not just attention.

4. MULTI-MODAL PROJECTOR FULLY UNFROZEN (no LoRA)
   The two `linear_1` / `linear_2` projector layers are tiny (~16 MB) and
   are the audio<->text bridge. Training them in full bf16 (not via LoRA)
   gives the model real capacity to learn new audio->script mappings.

5. PER-LANGUAGE VALIDATION LOSS
   Logged each epoch so you can see when Odia / Assamese actually converge.

6. NUM_EPOCHS = 15, cosine LR, gradient checkpointing.

Logging style still matches qwen2_audio_lora_indic_train.py:
    training_log.csv, validation_log.csv, per_language_loss.csv,
    per_language_val_loss.csv, training_curves.png, final_model/
"""

import os
import csv
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import (
    VoxtralForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT   = r"/path/to/data/IndicVoices_subset_4lang_combined"
OUTPUT_DIR  = r"/path/to/data/IndicVoices_subset_4lang_combined/voxtral_lora_indic_v3"
MODEL_ID    = "mistralai/Voxtral-Mini-3B-2507"

NUM_EPOCHS         = 5
BATCH_SIZE         = 2
GRAD_ACCUM_STEPS   = 4        # effective batch = 16
LEARNING_RATE      = 1e-4     # bumped from 5e-5: more capacity needs more LR
WARMUP_RATIO       = 0.05
MAX_TEXT_TOKENS    = 768      # Odia byte-fallback is verbose, give headroom
SEED               = 42

LORA_R       = 64             # was 32
LORA_ALPHA   = 64             # 1:1 with r
LORA_DROPOUT = 0.05

LANG_NAMES = {"as": "Assamese", "bn": "Bengali",
              "hi": "Hindi",    "or": "Odia"}
ALL_LANGS = ["as", "bn", "hi", "or"]
BATCH_SIZE         = 2

# Per-language gradient reweighting. Hindi (anchor) = 1.0. Bengali / Assamese
# share Eastern Nagari and tokenize ~1.5x worse than Devanagari in Tekken.
# Odia is byte-fallback ~3x worse. Boost the under-served languages.
LANG_LOSS_WEIGHT = {"hi": 1.0, "bn": 1.5, "as": 2.0, "or": 3.0}


def instruction_for(lang_code: str) -> str:
    name = LANG_NAMES.get(lang_code, lang_code.capitalize())
    return f"Transcribe the following {name} audio word-for-word."


# =============================================================================
# UTILS
# =============================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def resolve_audio_path(csv_path_value, split_audio_dir):
    fname = os.path.basename(str(csv_path_value).replace("\\", "/"))
    return os.path.join(split_audio_dir, fname)


def load_split(split_name):
    split_dir  = os.path.join(DATA_ROOT, split_name)
    audio_dir  = os.path.join(split_dir, "audio")
    csv_path   = os.path.join(split_dir, f"{split_name}.csv")
    df = pd.read_csv(csv_path)

    samples = []
    for _, row in df.iterrows():
        ap = resolve_audio_path(row["path"], audio_dir)
        if not os.path.isfile(ap):
            continue
        text = row.get("normalized") or row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        lang = str(row["lang"]).strip().lower()
        if lang not in LANG_NAMES:
            continue
        samples.append({"audio_path": ap,
                        "text": text.strip(),
                        "lang": lang})
    print(f"[{split_name}] loaded {len(samples)} samples from {csv_path}")
    return samples


# =============================================================================
# COLLATOR -- per-sample chat template, manual batch assembly
# =============================================================================
class VoxtralChatASRCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.eos_id = self.tokenizer.eos_token_id

    def _prompt_for(self, audio_path, lang):
            conv = [{
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text",  "text": instruction_for(lang)},
                ],
            }]
            return self.processor.apply_chat_template(
                conv,
                tokenize=True,
                return_tensors="pt",
            )

    def __call__(self, batch):
        prompts = [self._prompt_for(s["audio_path"], s["lang"]) for s in batch]
        feat_keys = [k for k in prompts[0].keys()
                     if k not in ("input_ids", "attention_mask")]

        ids_list, attn_list, lab_list = [], [], []
        for s, p in zip(batch, prompts):
            p_ids  = p["input_ids"][0].tolist()
            p_attn = p["attention_mask"][0].tolist()
            real = int(sum(p_attn))
            p_ids = p_ids[:real]; p_attn = p_attn[:real]

            t_ids = self.tokenizer(
                s["text"], add_special_tokens=False, padding=False,
                truncation=True, max_length=MAX_TEXT_TOKENS,
            )["input_ids"]

            ids  = p_ids  + t_ids + [self.eos_id]
            attn = p_attn + [1] * (len(t_ids) + 1)
            lab  = [-100] * len(p_ids) + t_ids + [self.eos_id]
            ids_list.append(ids); attn_list.append(attn); lab_list.append(lab)

        max_len = max(len(x) for x in ids_list)
        def pad(seq, fill): return seq + [fill] * (max_len - len(seq))
        ids_list  = [pad(x, self.pad_id) for x in ids_list]
        attn_list = [pad(x, 0)           for x in attn_list]
        lab_list  = [pad(x, -100)        for x in lab_list]

        out = {
            "input_ids":      torch.tensor(ids_list,  dtype=torch.long),
            "attention_mask": torch.tensor(attn_list, dtype=torch.long),
            "labels":         torch.tensor(lab_list,  dtype=torch.long),
            # extra: per-sample language tag for loss reweighting
            "_langs": [s["lang"] for s in batch],
        }
        for k in feat_keys:
            out[k] = torch.cat([p[k] for p in prompts], dim=0)
        return out


# =============================================================================
# WEIGHTED LOSS
# =============================================================================
def weighted_loss(logits, labels, langs):
    """Per-sample loss with per-language reweighting."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100, reduction="none",
    ).view(shift_labels.shape)

    sample_losses = []
    for j, lang in enumerate(langs):
        mask = shift_labels[j] != -100
        if not mask.any():
            continue
        sl = per_tok[j][mask].mean()
        w  = LANG_LOSS_WEIGHT.get(lang, 1.0)
        sample_losses.append(w * sl)
    return torch.stack(sample_losses).mean(), per_tok, shift_labels


# =============================================================================
# LOGGER
# =============================================================================
class TrainLogger:
    def __init__(self, output_dir, all_langs):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.all_langs  = all_langs
        self.train_csv     = os.path.join(output_dir, "training_log.csv")
        self.val_csv       = os.path.join(output_dir, "validation_log.csv")
        self.lang_csv      = os.path.join(output_dir, "per_language_loss.csv")
        self.lang_val_csv  = os.path.join(output_dir, "per_language_val_loss.csv")

        self.train_steps_log = []
        self.val_steps_log   = []
        self.lang_loss_accum  = defaultdict(float)
        self.lang_token_count = defaultdict(int)

        with open(self.train_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["global_step", "epoch", "batch_idx", "batch_loss",
                 "running_avg_loss", "lr", "gpu_mem_gb"])
        with open(self.val_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["global_step", "epoch", "val_loss"])
        with open(self.lang_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["global_step", "epoch"] +
                                   [f"loss_{l}" for l in all_langs])
        with open(self.lang_val_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["global_step", "epoch"] +
                                   [f"val_loss_{l}" for l in all_langs])

        self._running_sum = 0.0; self._running_n = 0

    def log_train_step(self, step, epoch, batch_idx, loss, lr):
        self._running_sum += loss; self._running_n += 1
        running_avg = self._running_sum / self._running_n
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        with open(self.train_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([step, epoch, batch_idx,
                                    f"{loss:.6f}", f"{running_avg:.6f}",
                                    f"{lr:.2e}", f"{gpu_mem:.2f}"])
        self.train_steps_log.append((step, loss))

    def accum_lang(self, lang, sample_loss, n_tokens):
        self.lang_loss_accum[lang]  += sample_loss * n_tokens
        self.lang_token_count[lang] += n_tokens

    def flush_lang(self, step, epoch):
        row = [step, epoch]
        print("\n  Per-language TRAIN loss:")
        for lang in self.all_langs:
            if self.lang_token_count[lang] > 0:
                avg = self.lang_loss_accum[lang] / self.lang_token_count[lang]
                row.append(f"{avg:.6f}")
                print(f"    {lang}: {avg:.4f}  ({self.lang_token_count[lang]} tok)")
            else:
                row.append("")
        with open(self.lang_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        self.lang_loss_accum.clear()
        self.lang_token_count.clear()

    def log_val(self, step, epoch, val_loss, per_lang_val):
        with open(self.val_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([step, epoch, f"{val_loss:.6f}"])
        self.val_steps_log.append((step, val_loss))
        row = [step, epoch]
        print("  Per-language VAL loss:")
        for lang in self.all_langs:
            v = per_lang_val.get(lang)
            row.append(f"{v:.6f}" if v is not None else "")
            if v is not None:
                print(f"    {lang}: {v:.4f}")
        with open(self.lang_val_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def plot(self):
        if not self.train_steps_log: return
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ts, tl = zip(*self.train_steps_log)
        axes[0].plot(ts, tl, alpha=0.3, label="batch loss")
        if len(tl) >= 20:
            window = max(10, len(tl) // 50)
            smooth = np.convolve(tl, np.ones(window)/window, mode="valid")
            axes[0].plot(ts[window-1:], smooth, label=f"smoothed (w={window})")
        axes[0].set_xlabel("global step"); axes[0].set_ylabel("loss")
        axes[0].set_title("Training loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        if self.val_steps_log:
            vs, vl = zip(*self.val_steps_log)
            axes[1].plot(vs, vl, "o-", color="orange")
            axes[1].set_xlabel("global step"); axes[1].set_ylabel("val loss")
            axes[1].set_title("Validation loss"); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved {out}")


# =============================================================================
# VALIDATION (per-language loss)
# =============================================================================
@torch.no_grad()
def run_validation(model, processor, collator, val_samples, device):
    model.eval()
    lang_loss_sum  = defaultdict(float)
    lang_tok_count = defaultdict(int)
    losses = []
    for i in tqdm(range(0, len(val_samples), BATCH_SIZE),
                  desc="Validation", leave=False):
        chunk = val_samples[i:i + BATCH_SIZE]
        try:
            batch = collator(chunk)
            langs = batch.pop("_langs")
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            out = model(**batch)
            losses.append(out.loss.item())

            # per-language val loss (token-weighted)
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            per_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100, reduction="none",
            ).view(shift_labels.shape)
            for j, lang in enumerate(langs):
                m = shift_labels[j] != -100
                n = int(m.sum().item())
                if n > 0:
                    lang_loss_sum[lang]  += per_tok[j][m].mean().item() * n
                    lang_tok_count[lang] += n
        except Exception as e:
            print(f"  [val skip] {e}")
            continue

    per_lang = {l: (lang_loss_sum[l] / lang_tok_count[l]) if lang_tok_count[l] > 0 else None
                for l in ALL_LANGS}
    model.train()
    return (float(np.mean(losses)) if losses else float("nan")), per_lang


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # ---------------- data
    train_samples = load_split("train")
    val_samples   = load_split("validation")
    if not train_samples:
        raise RuntimeError("No training samples found - check DATA_ROOT.")

    # report training mix
    counts = defaultdict(int)
    for s in train_samples: counts[s["lang"]] += 1
    print("Train mix:", dict(counts))
    print("Loss weights:", LANG_LOSS_WEIGHT)

    # ---------------- processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ---------------- 4-bit model
    print("Loading Voxtral in 4-bit (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["multi_modal_projector"],
    )
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    # Freeze audio tower entirely (its weights are 4-bit; we don't touch it).
    for p in model.audio_tower.parameters():
        p.requires_grad = False

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    # ---------------- LoRA on language model only (attention + MLP)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Freeze any LoRA adapters that landed inside the audio tower (suffix
    # match leaks q/k/v/o into Whisper).
    n_audio_lora_frozen = 0
    for name, p in model.named_parameters():
        if "audio_tower" in name and "lora_" in name:
            p.requires_grad = False
            n_audio_lora_frozen += 1
    print(f"Froze {n_audio_lora_frozen} LoRA params inside audio_tower")

    # FULLY UNFREEZE the multi-modal projector (no LoRA, full bf16 training).
    # These two linears are the audio<->text bridge and are tiny (~16 MB).
    # Giving them full capacity is the single biggest win for new scripts.
# FULLY UNFREEZE the multi-modal projector (kept in bf16 via skip_modules above).
    n_proj = 0
    for name, p in model.named_parameters():
        if "multi_modal_projector" in name and "lora_" not in name:
            p.requires_grad = True
            n_proj += p.numel()
    print(f"Multi-modal projector fully trainable: {n_proj:,} params")

    model.print_trainable_parameters()

    # ---------------- optimizer / scheduler
    collator = VoxtralChatASRCollator(processor)
    steps_per_epoch = math.ceil(len(train_samples) / BATCH_SIZE)
    total_optim_steps = math.ceil(steps_per_epoch * NUM_EPOCHS / GRAD_ACCUM_STEPS)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_optim_steps),
        num_training_steps=total_optim_steps,
    )

    logger = TrainLogger(OUTPUT_DIR, ALL_LANGS)
    print(f"\nTrain {len(train_samples)} | Val {len(val_samples)} | "
          f"batch {BATCH_SIZE} x accum {GRAD_ACCUM_STEPS} | "
          f"steps/epoch {steps_per_epoch} | total optim steps {total_optim_steps}")

    # ---------------- training loop
    global_step = 0
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*70}\nEpoch {epoch}/{NUM_EPOCHS}\n{'='*70}")
        order = list(range(len(train_samples)))
        random.shuffle(order)

        epoch_loss_sum = 0.0; epoch_loss_n = 0
        pbar = tqdm(total=len(train_samples), desc=f"Epoch {epoch}", unit="audio")

        for batch_idx in range(steps_per_epoch):
            idxs  = order[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
            chunk = [train_samples[i] for i in idxs]
            fnames = [(s["lang"], os.path.basename(s["audio_path"])) for s in chunk]
            print(f"\n  Batch {batch_idx + 1}/{steps_per_epoch} {fnames}")

            try:
                batch = collator(chunk)
                langs = batch.pop("_langs")
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

                out = model(**batch)
                loss, per_tok, shift_labels = weighted_loss(
                    out.logits, batch["labels"], langs)
                (loss / GRAD_ACCUM_STEPS).backward()

                # accumulate UNWEIGHTED per-language loss for logging
                with torch.no_grad():
                    for j, lang in enumerate(langs):
                        m = shift_labels[j] != -100
                        if m.any():
                            sl = per_tok[j][m].mean().item()
                            logger.accum_lang(lang, sl, int(m.sum().item()))

                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or \
                   (batch_idx + 1) == steps_per_epoch:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step(); scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                bl = loss.item()
                epoch_loss_sum += bl; epoch_loss_n += 1
                lr_now = scheduler.get_last_lr()[0]
                logger.log_train_step(global_step, epoch, batch_idx + 1, bl, lr_now)
                print(f"    weighted loss={bl:.4f}  lr={lr_now:.2e}  "
                      f"gpu={torch.cuda.memory_allocated()/1e9:.1f}GB")

            except torch.cuda.OutOfMemoryError:
                print("    [OOM] skip"); optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    [skip] {e}")
                optimizer.zero_grad(set_to_none=True)

            pbar.update(len(chunk))
        pbar.close()

        avg_epoch = epoch_loss_sum / max(epoch_loss_n, 1)
        print(f"\n>> Epoch {epoch} avg train loss: {avg_epoch:.4f}")
        logger.flush_lang(global_step, epoch)

        if val_samples:
            vloss, per_lang_val = run_validation(
                model, processor, collator, val_samples, device)
            print(f">> Epoch {epoch} validation loss: {vloss:.4f}")
            logger.log_val(global_step, epoch, vloss, per_lang_val)

        logger.plot()

        # save checkpoint each epoch so you can early-stop
        ckpt = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch}")
        model.save_pretrained(ckpt)
        print(f"  saved {ckpt}")

    # ---------------- final save
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    print(f"\nSaving LoRA adapter + processor to {final_dir}")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.plot()
    print("\nDone.")


if __name__ == "__main__":
    main()