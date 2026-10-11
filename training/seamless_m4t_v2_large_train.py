import os
import csv
import time
import random
from math import gcd
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    SeamlessM4Tv2ForSpeechToText,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =========================================================
# CONFIG - FIXED HYPERPARAMETERS FOR LARGE DATASETS
# =========================================================
MODEL_NAME = "facebook/seamless-m4t-v2-large"

DATASET_ROOT = r"/path/to/data/IndicVoices_subset_4lang_combined"  # <-- CHANGE THIS  # <-- CHANGE THIS
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "validation")

TRAIN_CSV = os.path.join(TRAIN_DIR, "train.csv")
VAL_CSV = os.path.join(VAL_DIR, "validation.csv")

OUTPUT_DIR = os.path.join(DATASET_ROOT, "seamless_m4t_v2_large_lora_4lang")

# =========================================================
# 🔧 FIXED HYPERPARAMETERS
# =========================================================
BATCH_SIZE = 4
EPOCHS = 5  # ✅ INCREASED: Multiple epochs help reinforce learning
LR = 1e-5  # ✅ LOWERED: 5x lower than before for stability
WARMUP_RATIO = 0.10  # ✅ INCREASED: 10% warmup for better stability
MAX_GRAD_NORM = 0.5  # ✅ LOWERED: Tighter gradient clipping
WEIGHT_DECAY = 0.01  # ✅ ADDED: Regularization to prevent overfitting
RANDOM_SEED = 42

GRAD_ACCUM_STEPS = 8  # effective batch = 4 * 8 = 32
VAL_EVERY_N_OPT_STEPS = 100  # Less frequent validation for speed
LOG_PER_LANG_EVERY = 100
PLOT_EVERY_N_STEPS = 500

# ✅ NEW: Early stopping configuration
EARLY_STOPPING_PATIENCE = 10  # Stop if val loss doesn't improve for N validations
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement to count

# ✅ NEW: Checkpointing - save best model
SAVE_BEST_ONLY = True
BEST_VAL_LOSS = float("inf")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# LANGUAGE MAPPING
# =========================================================
CSV_LANG_TO_M4T = {
    "as": "asm",  # Assamese
    "bn": "ben",  # Bengali
    "hi": "hin",  # Hindi
    "or": "ory",  # Odia
}

M4T_TO_PRETTY = {
    "asm": "Assamese",
    "ben": "Bengali",
    "hin": "Hindi",
    "ory": "Odia",
}


# =========================================================
# IO HELPERS
# =========================================================
def extract_filename_from_mixed_path(path_str: str) -> str:
    if path_str is None:
        return ""
    normalized = str(path_str).strip().replace("\\", "/")
    if not normalized:
        return ""
    return os.path.basename(normalized)


def resolve_path(p: str, split_audio_dir: str) -> str:
    p = str(p).strip()
    if os.path.exists(p):
        return p

    base = extract_filename_from_mixed_path(p)
    if not base:
        return ""

    cand = os.path.join(split_audio_dir, base)
    if os.path.exists(cand):
        return cand

    # Fallback: keep a predictable local path even if file is missing.
    return cand


def load_audio_full(path: str, target_sr: int):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)

    return wav, target_sr


def get_gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9


# =========================================================
# LOAD CSVs
# =========================================================
def load_split(csv_path: str, audio_dir: str):
    df = pd.read_csv(csv_path)
    required_cols = {"path", "text", "lang"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    samples = []
    skipped_empty_filename = 0
    for _, row in df.iterrows():
        lang2 = str(row["lang"]).strip().lower()
        if lang2 not in CSV_LANG_TO_M4T:
            continue

        raw_path = row["path"]
        file_name = extract_filename_from_mixed_path(raw_path)
        if not file_name:
            skipped_empty_filename += 1
            continue

        p = resolve_path(raw_path, audio_dir)
        samples.append(
            {
                "audio": p,
                "text": str(row["text"]),
                "lang2": lang2,
                "lang3": CSV_LANG_TO_M4T[lang2],
            }
        )
    if skipped_empty_filename > 0:
        print(
            f"  ⚠️ Skipped {skipped_empty_filename} rows with empty/invalid audio filename in CSV path."
        )
    return samples


print(f"Loading train CSV: {TRAIN_CSV}")
train_samples = load_split(TRAIN_CSV, os.path.join(TRAIN_DIR, "audio"))
print(f"Train samples: {len(train_samples)}")

print(f"Loading val CSV: {VAL_CSV}")
val_samples = load_split(VAL_CSV, os.path.join(VAL_DIR, "audio"))
print(f"Val samples: {len(val_samples)}")

print("Train language distribution:")
dist = defaultdict(int)
for s in train_samples:
    dist[s["lang3"]] += 1
for k in sorted(dist):
    print(f"  {M4T_TO_PRETTY.get(k, k)} ({k}): {dist[k]}")


# =========================================================
# LOAD PROCESSOR + MODEL (4-bit) + LoRA
# =========================================================
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

TARGET_SR = processor.feature_extractor.sampling_rate
print(f"Target sampling rate: {TARGET_SR}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading SeamlessM4Tv2ForSpeechToText in 4-bit...")
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# ✅ FIXED: Expand target modules for better adaptation
CANDIDATE_TARGETS = [
    "q_proj",
    "v_proj",
    "k_proj",
    "out_proj",
]  # Added more attention modules
present = set()
for name, _ in model.named_modules():
    for t in CANDIDATE_TARGETS:
        if name.endswith(t):
            present.add(t)
TARGET_MODULES = sorted(list(present)) if present else ["q_proj", "v_proj"]

print(f"LoRA target_modules = {TARGET_MODULES}")

# ✅ FIXED: More conservative LoRA config
lora_config = LoraConfig(
    r=8,  # ✅ LOWERED: Smaller rank to prevent overfitting
    lora_alpha=16,  # ✅ ADJUSTED: 2x rank (standard practice)
    target_modules=TARGET_MODULES,
    lora_dropout=0.1,  # ✅ INCREASED: More regularization
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# ✅ FIXED: Add weight decay for regularization
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8
)
scaler = torch.amp.GradScaler("cuda")

# --- LR scheduler: linear warmup + cosine decay ---
total_batches = len(train_samples) // BATCH_SIZE
total_opt_steps = (total_batches // GRAD_ACCUM_STEPS) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_opt_steps)

from torch.optim.lr_scheduler import LambdaLR
import math


def cosine_with_warmup(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(
        max(1, total_opt_steps - warmup_steps)
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_warmup)
print(
    f"LR scheduler: cosine with {warmup_steps} warmup steps out of {total_opt_steps} total opt steps"
)


# =========================================================
# LOGGING
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, "training_log.csv")
VAL_LOG_CSV = os.path.join(OUTPUT_DIR, "validation_log.csv")
PER_LANG_CSV = os.path.join(OUTPUT_DIR, "per_language_loss.csv")

with open(TRAIN_LOG_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "global_step",
            "epoch",
            "batch_idx",
            "batch_loss",
            "running_avg_loss",
            "learning_rate",
            "gpu_mem_allocated_gb",
            "gpu_mem_reserved_gb",
            "samples_seen",
            "elapsed_sec",
        ]
    )

with open(VAL_LOG_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "global_step",
            "epoch",
            "val_loss",
            "val_loss_by_lang",
            "elapsed_sec",
            "is_best",
        ]
    )

all_langs = sorted(set(s["lang3"] for s in train_samples))
with open(PER_LANG_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        ["global_step", "epoch"]
        + [f"loss_{k}" for k in all_langs]
        + [f"count_{k}" for k in all_langs]
    )

train_steps_log = []
train_avg_log = []
val_log = []
per_lang_log = []

lang_loss_accum = defaultdict(float)
lang_loss_count = defaultdict(int)

training_start = time.time()

RUNNING_AVG_WINDOW = 50
running_loss_window = []

# Early stopping tracking
validation_losses = []
no_improvement_count = 0


def log_train_step(
    global_step, epoch, batch_idx, batch_loss, running_avg, lr, samples_seen
):
    alloc, reserved = get_gpu_mem_gb()
    elapsed = time.time() - training_start
    with open(TRAIN_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                global_step,
                epoch,
                batch_idx,
                f"{batch_loss:.6f}",
                f"{running_avg:.6f}",
                f"{lr:.2e}",
                f"{alloc:.2f}",
                f"{reserved:.2f}",
                samples_seen,
                f"{elapsed:.1f}",
            ]
        )
    train_steps_log.append((global_step, batch_loss))
    train_avg_log.append((global_step, running_avg))


def flush_per_lang(global_step, epoch):
    if not lang_loss_count:
        return

    row = [global_step, epoch]
    lang_avgs = {}
    for k in all_langs:
        if lang_loss_count[k] > 0:
            avg = lang_loss_accum[k] / lang_loss_count[k]
            row.append(f"{avg:.6f}")
            lang_avgs[k] = avg
        else:
            row.append("")
    for k in all_langs:
        row.append(lang_loss_count.get(k, 0))

    with open(PER_LANG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)

    per_lang_log.append((global_step, dict(lang_avgs)))

    print("\n  📊 Per-language loss breakdown:")
    for k in all_langs:
        if lang_loss_count.get(k, 0) > 0:
            print(
                f"    {M4T_TO_PRETTY.get(k, k)} ({k}): {lang_avgs[k]:.4f} (n={lang_loss_count[k]})"
            )

    lang_loss_accum.clear()
    lang_loss_count.clear()


def save_plots(suffix=""):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        if train_steps_log:
            steps = [x[0] for x in train_steps_log]
            losses = [x[1] for x in train_steps_log]
            axes[0].plot(steps, losses, alpha=0.3, linewidth=0.7, label="Batch Loss")

        if train_avg_log:
            steps = [x[0] for x in train_avg_log]
            avgs = [x[1] for x in train_avg_log]
            axes[0].plot(steps, avgs, linewidth=1.5, label="Running Avg")

        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Global Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        if val_log:
            vsteps = [x[0] for x in val_log]
            vloss = [x[1] for x in val_log]
            axes[1].plot(vsteps, vloss, marker="o", linewidth=1.7, label="Val Loss")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        else:
            axes[1].text(
                0.5,
                0.5,
                "No validation logged yet",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )

        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Global Step")
        axes[1].set_ylabel("Loss")

        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"training_curves{suffix}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📈 Saved: {out}")

        if per_lang_log:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            series = defaultdict(lambda: ([], []))
            for step, d in per_lang_log:
                for k, v in d.items():
                    series[k][0].append(step)
                    series[k][1].append(v)

            for k in sorted(series.keys()):
                ax.plot(
                    series[k][0],
                    series[k][1],
                    marker="o",
                    markersize=3,
                    linewidth=1.4,
                    label=f"{k}",
                )

            ax.set_title("Per-Language Training Loss")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Avg Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

            plt.tight_layout()
            out2 = os.path.join(OUTPUT_DIR, f"per_language_loss{suffix}.png")
            plt.savefig(out2, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  📈 Saved: {out2}")

    except Exception as e:
        print(f"  ⚠️ Plotting skipped/failed: {e}")


# =========================================================
# CORE: build labels with correct tgt language token
# =========================================================
def get_lang_token_id(lang3: str) -> int:
    tok = processor.tokenizer
    if hasattr(tok, "lang_code_to_id") and lang3 in tok.lang_code_to_id:
        return tok.lang_code_to_id[lang3]

    cand = f"__{lang3}__"
    tid = tok.convert_tokens_to_ids(cand)
    if tid is not None and tid != tok.unk_token_id:
        return tid

    raise ValueError(f"Could not find language token id for {lang3}")


def collate_batch(batch):
    audios = []
    texts = []
    file_names = []
    batch_langs = []

    for s in batch:
        wav, _ = load_audio_full(s["audio"], TARGET_SR)
        audios.append(wav)
        texts.append(s["text"])
        file_names.append(os.path.basename(s["audio"]))
        batch_langs.append(s["lang3"])

    inputs = processor(
        audio=audios,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )

    tok = processor.tokenizer
    label_ids = []
    max_len = 0
    for t, lang3 in zip(texts, batch_langs):
        lang_id = get_lang_token_id(lang3)
        ids = tok.encode(t, add_special_tokens=True)
        if len(ids) == 0 or ids[0] != lang_id:
            ids = [lang_id] + ids
        label_ids.append(ids)
        max_len = max(max_len, len(ids))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    labels = torch.full((len(label_ids), max_len), fill_value=-100, dtype=torch.long)
    for i, ids in enumerate(label_ids):
        ids_t = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(ids)] = ids_t

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    labels = labels.to(model.device)

    return inputs, labels, file_names, batch_langs


# =========================================================
# VALIDATION
# =========================================================
@torch.no_grad()
def run_validation(global_step, epoch):
    global BEST_VAL_LOSS, no_improvement_count

    model.eval()
    total = 0.0
    steps = 0
    by_lang_sum = defaultdict(float)
    by_lang_cnt = defaultdict(int)

    for i in range(0, len(val_samples), BATCH_SIZE):
        batch = val_samples[i : i + BATCH_SIZE]
        try:
            inputs, labels, _, langs = collate_batch(batch)
        except Exception as e:
            print(f"  [VAL] skipping batch due to error: {e}")
            continue

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(**inputs, labels=labels)

        loss = float(out.loss.item())
        total += loss
        steps += 1
        for lg in langs:
            by_lang_sum[lg] += loss
            by_lang_cnt[lg] += 1

    model.train()

    if steps == 0:
        avg = float("nan")
    else:
        avg = total / steps

    breakdown = {}
    for lg in sorted(by_lang_sum.keys()):
        if by_lang_cnt[lg] > 0:
            breakdown[lg] = by_lang_sum[lg] / by_lang_cnt[lg]

    elapsed = time.time() - training_start

    # ✅ NEW: Check if this is the best model
    is_best = False
    if not np.isnan(avg):
        validation_losses.append(avg)
        if avg < BEST_VAL_LOSS - EARLY_STOPPING_MIN_DELTA:
            BEST_VAL_LOSS = avg
            is_best = True
            no_improvement_count = 0

            # Save best checkpoint
            best_dir = os.path.join(CHECKPOINTS_DIR, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  🏆 New best model! Val loss: {avg:.4f}")
        else:
            no_improvement_count += 1

    with open(VAL_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                global_step,
                epoch,
                f"{avg:.6f}",
                str(breakdown),
                f"{elapsed:.1f}",
                is_best,
            ]
        )

    val_log.append((global_step, avg))

    print(f"\n  🔍 VALIDATION @ step {global_step}: val_loss={avg:.4f}")
    for lg, v in breakdown.items():
        print(f"     {M4T_TO_PRETTY.get(lg, lg)} ({lg}): {v:.4f}")

    if is_best:
        print(f"  ⭐ This is the BEST model so far!")

    return avg


# =========================================================
# TRAIN LOOP
# =========================================================
print(f"\nSaving outputs to: {OUTPUT_DIR}")
model.train()

global_step = 0
opt_step = 0
should_stop = False

for epoch in range(1, EPOCHS + 1):
    if should_stop:
        break

    print("\n====================")
    print(f"Epoch {epoch}/{EPOCHS}")
    print("====================")

    random.shuffle(train_samples)

    epoch_loss_sum = 0.0
    epoch_batches = 0

    progress = tqdm(total=len(train_samples), desc=f"Epoch {epoch}", ncols=100)

    for i in range(0, len(train_samples), BATCH_SIZE):
        batch = train_samples[i : i + BATCH_SIZE]

        try:
            inputs, labels, file_names, batch_langs = collate_batch(batch)
        except RuntimeError as e:
            print(f"\n[SKIP] batch due to runtime error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            progress.update(len(batch))
            continue
        except Exception as e:
            print(f"\n[SKIP] batch due to error: {e}")
            progress.update(len(batch))
            continue

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        # ✅ NEW: Check for NaN loss (sign of training instability)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ NaN/Inf loss detected! Skipping batch.")
            print(f"   Raw loss: {outputs.loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            progress.update(len(batch))
            continue

        scaler.scale(loss).backward()

        # Optim step w/ accumulation
        do_step = ((epoch_batches + 1) % GRAD_ACCUM_STEPS == 0) or (
            i + BATCH_SIZE >= len(train_samples)
        )
        if do_step:
            # ✅ FIXED: Check for gradient issues before stepping
            scaler.unscale_(optimizer)

            # Check for gradient explosion
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm**0.5

            if total_grad_norm > 100.0:
                print(f"\n⚠️ Gradient explosion detected! Norm: {total_grad_norm:.2f}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                progress.update(len(batch))
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            opt_step += 1

        batch_loss = float(loss.item() * GRAD_ACCUM_STEPS)
        epoch_loss_sum += batch_loss
        epoch_batches += 1
        global_step += 1

        # Running avg
        running_loss_window.append(batch_loss)
        if len(running_loss_window) > RUNNING_AVG_WINDOW:
            running_loss_window.pop(0)
        running_avg = sum(running_loss_window) / len(running_loss_window)

        # Per-language accum
        for lg in batch_langs:
            lang_loss_accum[lg] += batch_loss
            lang_loss_count[lg] += 1

        # Console output
        print("\n------------------------------------")
        print(
            f"Batch {epoch_batches} | Global Step {global_step} | Optimizer Step {opt_step}"
        )
        print("Files:")
        for fn in file_names:
            print("  ", fn)
        print(f"Batch Loss: {batch_loss:.4f} | Running Avg: {running_avg:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        alloc, reserved = get_gpu_mem_gb()
        print(f"GPU Memory: {alloc:.1f}GB alloc / {reserved:.1f}GB reserved")

        # CSV logging
        lr = optimizer.param_groups[0]["lr"]
        samples_seen = i + len(batch)
        log_train_step(
            global_step, epoch, epoch_batches, batch_loss, running_avg, lr, samples_seen
        )

        # Periodic per-language flush
        if global_step % LOG_PER_LANG_EVERY == 0:
            flush_per_lang(global_step, epoch)

        # Validation periodically
        if opt_step > 0 and (opt_step % VAL_EVERY_N_OPT_STEPS == 0) and do_step:
            run_validation(global_step, epoch)

            # ✅ NEW: Early stopping check
            if no_improvement_count >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\n🛑 Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} validations."
                )
                should_stop = True
                break

        # Plot periodically
        if global_step % PLOT_EVERY_N_STEPS == 0:
            save_plots(suffix=f"_step{global_step}")

        progress.update(len(batch))

    progress.close()

    if should_stop:
        break

    # End epoch flush per-language
    if lang_loss_count:
        flush_per_lang(global_step, epoch)

    avg_epoch_loss = epoch_loss_sum / max(1, epoch_batches)
    print(f"\n✅ Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}")

    print("\n--- End-of-Epoch Validation ---")
    run_validation(global_step, epoch)
    save_plots(suffix=f"_epoch{epoch}")

    # Save LoRA adapter + processor per epoch
    epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    print(f"\nSaving LoRA adapter to: {epoch_dir}")
    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)

# Save final adapter
print("\n==============================")
print("Training complete. Saving FINAL adapter.")
print("==============================")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
save_plots(suffix="_final")

print("\nOutput files:")
print("  📄", TRAIN_LOG_CSV)
print("  📄", VAL_LOG_CSV)
print("  📄", PER_LANG_CSV)
print("  📈", os.path.join(OUTPUT_DIR, "training_curves_final.png"))
print("  📈", os.path.join(OUTPUT_DIR, "per_language_loss_final.png"))
print("  🏆 Best checkpoint:", os.path.join(CHECKPOINTS_DIR, "best"))
print("\n✅ Final LoRA adapter saved to:", OUTPUT_DIR)
