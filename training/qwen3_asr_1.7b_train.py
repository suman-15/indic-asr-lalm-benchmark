import os
import csv
import math
import json
import time
import random
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from tqdm import tqdm

from transformers import BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"

TRAIN_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/train"
VAL_DIR   = r"/path/to/data/IndicVoices_subset_4lang_combined/validation"

OUTPUT_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/qwen3_asr_lora_indic"

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
RANDOM_SEED = 42

VAL_EVERY_N_OPT_STEPS = 100
PLOT_EVERY_N_STEPS = 100
LOG_PER_LANG_EVERY = 200
CHECKPOINT_EVERY_N_OPT_STEPS = 200

USE_FLASH_ATTENTION_2 = True
SORT_BUCKET_SIZE = 64

CPU_NUM_THREADS = 1

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Optional
COMPUTE_WER_CER = False


# ============================================================
# GLOBAL ENV
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

torch.set_num_threads(CPU_NUM_THREADS)
torch.set_num_interop_threads(CPU_NUM_THREADS)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not found. This script requires a GPU.")

DEVICE = torch.device("cuda:0")


# ============================================================
# OPTIONAL METRICS
# ============================================================

try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except Exception:
    JIWER_AVAILABLE = False
    if COMPUTE_WER_CER:
        warnings.warn("jiwer is not installed. WER/CER will be skipped.")


# ============================================================
# LANGUAGE HELPERS
# ============================================================

LANG_CODE_TO_NAME = {
    "as": "Assamese",
    "bn": "Bengali",
    "hi": "Hindi",
    "or": "Odia",
    "odia": "Odia",
    "assamese": "Assamese",
    "bengali": "Bengali",
    "hindi": "Hindi",
}

def normalize_language(lang_value: str) -> str:
    value = str(lang_value).strip().lower()
    return LANG_CODE_TO_NAME.get(value, str(lang_value).strip())

def build_target_text(lang_name: str, transcript: str) -> str:
    transcript = str(transcript).strip()
    return f"language {lang_name}<asr_text>{transcript}"


# ============================================================
# MODEL / PROCESSOR LOADING
# ============================================================

def patch_outer_forward(model):
    """
    Qwen3-ASR training/inference wrapper exposes .thinker.forward().
    Patch outer forward so normal training calls work.
    """
    cls = model.__class__
    if getattr(cls, "_forward_patched_for_training", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Could not patch forward: model.thinker.forward not found. "
            "Please ensure qwen-asr is installed correctly."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched_for_training = True


def gpu_supports_bf16() -> bool:
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def load_model_and_processor():
    from qwen_asr import Qwen3ASRModel

    compute_dtype = torch.bfloat16 if gpu_supports_bf16() else torch.float16
    preferred_attn = "flash_attention_2" if USE_FLASH_ATTENTION_2 else "sdpa"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    try:
        wrapper = Qwen3ASRModel.from_pretrained(
            MODEL_NAME,
            dtype=compute_dtype,
            device_map="cuda:0",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            attn_implementation=preferred_attn,
        )
        model = wrapper.model
        processor = wrapper.processor
        patch_outer_forward(model)
        return model, processor, compute_dtype, preferred_attn
    except Exception as e1:
        if preferred_attn == "flash_attention_2":
            print(f"[Info] flash_attention_2 load failed, retrying with sdpa.\n{e1}")
            wrapper = Qwen3ASRModel.from_pretrained(
                MODEL_NAME,
                dtype=compute_dtype,
                device_map="cuda:0",
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            model = wrapper.model
            processor = wrapper.processor
            patch_outer_forward(model)
            return model, processor, compute_dtype, "sdpa"
        raise


# ============================================================
# DATA HELPERS
# ============================================================

def validate_csv(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    required = ["path", "text", "lang"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=required).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[Info] Dropped {dropped} rows with missing path/text/lang from {csv_path}")

    return df


def resolve_audio_path(path_value: str, split_dir: str) -> str:
    raw = str(path_value).strip()
    norm = raw.replace("\\", os.sep).replace("/", os.sep)

    candidates = []

    if os.path.isabs(norm):
        candidates.append(norm)

    candidates.append(os.path.join(split_dir, norm))
    candidates.append(os.path.join(split_dir, "audio", os.path.basename(norm)))
    candidates.append(os.path.join(split_dir, os.path.basename(norm)))

    for cand in candidates:
        if os.path.exists(cand):
            return os.path.normpath(cand)

    return os.path.normpath(candidates[0])


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if wav.size == 0:
        raise RuntimeError(f"Audio file is empty: {path}")

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

    return np.asarray(wav, dtype=np.float32)


def load_split(split_dir: str, split_csv_name: str):
    csv_path = os.path.join(split_dir, split_csv_name)
    df = pd.read_csv(csv_path)
    df = validate_csv(df, csv_path)

    samples = []
    missing_count = 0

    for _, row in df.iterrows():
        audio_path = resolve_audio_path(row["path"], split_dir)
        if not os.path.exists(audio_path):
            missing_count += 1
            continue

        lang_name = normalize_language(row["lang"])
        transcript = str(row["text"]).strip()
        duration = float(row["duration"]) if "duration" in row and pd.notna(row["duration"]) else -1.0

        samples.append({
            "audio_path": audio_path,
            "file_name": os.path.basename(audio_path),
            "lang_name": lang_name,
            "transcript": transcript,
            "target_text": build_target_text(lang_name, transcript),
            "duration": duration,
        })

    if missing_count > 0:
        print(f"[Info] {missing_count} rows skipped because audio file was not found in {csv_path}")

    if not samples:
        raise RuntimeError(f"No valid samples found in {csv_path}")

    return samples, csv_path


def bucketed_random_order(samples, bucket_size=64):
    shuffled = samples[:]
    random.shuffle(shuffled)

    ordered = []
    for i in range(0, len(shuffled), bucket_size):
        bucket = shuffled[i:i + bucket_size]
        bucket.sort(key=lambda x: x["duration"] if x["duration"] >= 0 else 1e9)
        ordered.extend(bucket)
    return ordered


# ============================================================
# PROMPT / LABEL BUILDING
# ============================================================

def render_prefix_text(processor) -> str:
    """
    Build the prompt prefix using the released Qwen3-ASR chat template.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a speech recognition model. Transcribe the input audio accurately."
        },
        {
            "role": "user",
            "content": [{"type": "audio", "audio": "placeholder.wav"}]
        },
    ]

    prefix_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    if not isinstance(prefix_text, str):
        raise RuntimeError("processor.apply_chat_template(...) did not return a string.")

    return prefix_text


def move_inputs_to_device(inputs: dict, device: torch.device, model_dtype: torch.dtype):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
            out[k] = v
        else:
            out[k] = v
    return out


def build_batch_tensors(processor, batch_samples, prefix_text: str, eos_token: str):
    audio_arrays = [load_audio(s["audio_path"], target_sr=16000) for s in batch_samples]
    prefix_texts = [prefix_text] * len(batch_samples)
    targets = [s["target_text"] for s in batch_samples]
    full_texts = [p + t + eos_token for p, t in zip(prefix_texts, targets)]

    full_inputs = processor(
        text=full_texts,
        audio=audio_arrays,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    prefix_inputs = processor(
        text=prefix_texts,
        audio=audio_arrays,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()

    labels = full_inputs["input_ids"].clone()
    for i, prefix_len in enumerate(prefix_lens):
        labels[i, :prefix_len] = -100

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    full_inputs["labels"] = labels
    return full_inputs


# ============================================================
# LOGGING
# ============================================================

class TrainingLogger:
    def __init__(self, output_dir: str, languages):
        self.output_dir = output_dir
        self.languages = sorted(set(languages))

        self.train_csv = os.path.join(output_dir, "training_log.csv")
        self.val_csv = os.path.join(output_dir, "validation_log.csv")
        self.lang_csv = os.path.join(output_dir, "per_language_loss.csv")

        self.train_steps = []
        self.train_running = []
        self.val_steps = []
        self.per_lang_points = []

        self.global_loss_times_tokens = 0.0
        self.global_tokens = 0

        self.lang_loss_accum = defaultdict(float)
        self.lang_token_accum = defaultdict(int)

        self.start_time = time.time()
        self._init_csvs()

    def _init_csvs(self):
        os.makedirs(self.output_dir, exist_ok=True)

        with open(self.train_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "global_step", "epoch", "batch_idx", "batch_loss",
                "running_avg_loss", "running_std",
                "learning_rate", "gpu_mem_allocated_gb", "gpu_mem_reserved_gb",
                "samples_seen", "valid_tokens", "elapsed_sec"
            ])

        with open(self.val_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "global_step", "epoch",
                "val_loss", "val_loss_std",
                "token_weighted_val_loss",
                "val_loss_by_lang", "elapsed_sec"
            ])

        with open(self.lang_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["global_step", "epoch"]
            header += [f"loss_{lang}" for lang in self.languages]
            header += [f"tokens_{lang}" for lang in self.languages]
            writer.writerow(header)

    @staticmethod
    def gpu_mem():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return alloc, reserved

    def log_train_step(self, global_step, epoch, batch_idx, batch_loss, running_avg, running_std, lr, samples_seen, valid_tokens):
        elapsed = time.time() - self.start_time
        alloc, reserved = self.gpu_mem()

        with open(self.train_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                global_step, epoch, batch_idx,
                f"{batch_loss:.6f}", f"{running_avg:.6f}", f"{running_std:.6f}",
                f"{lr:.8e}", f"{alloc:.2f}", f"{reserved:.2f}",
                samples_seen, valid_tokens, f"{elapsed:.1f}",
            ])

        self.train_steps.append((global_step, batch_loss))
        self.train_running.append((global_step, running_avg))
        self.global_loss_times_tokens += batch_loss * valid_tokens
        self.global_tokens += valid_tokens

    def log_validation(self, global_step, epoch, avg_loss, std_loss, token_weighted_loss, by_lang):
        elapsed = time.time() - self.start_time

        with open(self.val_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                global_step, epoch,
                f"{avg_loss:.6f}", f"{std_loss:.6f}",
                f"{token_weighted_loss:.6f}",
                json.dumps(by_lang, ensure_ascii=False),
                f"{elapsed:.1f}",
            ])

        self.val_steps.append((global_step, avg_loss))

    def flush_per_language(self, global_step, epoch):
        if not self.lang_token_accum:
            return

        row = [global_step, epoch]
        current_point = {}

        for lang in self.languages:
            tokens = self.lang_token_accum.get(lang, 0)
            if tokens > 0:
                avg_loss = self.lang_loss_accum[lang] / tokens
                row.append(f"{avg_loss:.6f}")
                current_point[lang] = avg_loss
            else:
                row.append("")

        for lang in self.languages:
            row.append(self.lang_token_accum.get(lang, 0))

        with open(self.lang_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.per_lang_points.append((global_step, current_point))
        self.lang_loss_accum.clear()
        self.lang_token_accum.clear()

    def token_weighted_train_avg(self):
        if self.global_tokens == 0:
            return 0.0
        return self.global_loss_times_tokens / self.global_tokens


def save_training_plots(output_dir, logger: TrainingLogger, suffix=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Warn] Plot generation skipped: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if logger.train_steps:
        x = [p[0] for p in logger.train_steps]
        y = [p[1] for p in logger.train_steps]
        axes[0].plot(x, y, alpha=0.35, linewidth=0.8, label="Batch loss")

    if logger.train_running:
        x = [p[0] for p in logger.train_running]
        y = [p[1] for p in logger.train_running]
        axes[0].plot(x, y, linewidth=1.5, label="Running avg")

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Global step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if logger.val_steps:
        x = [p[0] for p in logger.val_steps]
        y = [p[1] for p in logger.val_steps]
        axes[1].plot(x, y, marker="o", linewidth=1.5, label="Validation loss")

    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("Global step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_curves{suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    if logger.per_lang_points:
        plt.figure(figsize=(12, 5))
        lang_series = defaultdict(lambda: ([], []))
        for step, d in logger.per_lang_points:
            for lang, loss_val in d.items():
                lang_series[lang][0].append(step)
                lang_series[lang][1].append(loss_val)

        for lang in sorted(lang_series.keys()):
            xs, ys = lang_series[lang]
            plt.plot(xs, ys, marker="o", linewidth=1.3, label=lang)

        plt.title("Per-language training loss")
        plt.xlabel("Global step")
        plt.ylabel("Token-weighted loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"per_language_loss{suffix}.png"), dpi=150, bbox_inches="tight")
        plt.close()


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(output_dir, tag, model, processor, optimizer, scheduler, scaler, global_step, epoch):
    ckpt_dir = os.path.join(output_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)

    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
            "cuda_state": torch.cuda.get_rng_state_all(),
        },
        os.path.join(ckpt_dir, "training_state.pt"),
    )

    with open(os.path.join(ckpt_dir, "base_model_name.txt"), "w", encoding="utf-8") as f:
        f.write(MODEL_NAME + "\n")

    print(f"[Checkpoint] Saved: {ckpt_dir}")


# ============================================================
# LOSS BREAKDOWN
# ============================================================

def compute_per_sample_losses(logits: torch.Tensor, labels: torch.Tensor):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    flat_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.shape)

    sample_losses = []
    sample_tokens = []

    for i in range(shift_labels.size(0)):
        valid_mask = shift_labels[i] != -100
        token_count = int(valid_mask.sum().item())
        if token_count == 0:
            sample_losses.append(float("nan"))
            sample_tokens.append(0)
            continue
        sample_loss = float(flat_loss[i][valid_mask].mean().item())
        sample_losses.append(sample_loss)
        sample_tokens.append(token_count)

    return sample_losses, sample_tokens


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def run_validation(model, processor, val_samples, batch_size, prefix_text, eos_token, model_dtype, logger, global_step, epoch):
    model.eval()

    all_sample_losses = []
    all_token_counts = []
    lang_loss_sum = defaultdict(float)
    lang_token_sum = defaultdict(int)

    for start in tqdm(range(0, len(val_samples), batch_size), desc="Validation", leave=False):
        batch_samples = val_samples[start:start + batch_size]

        try:
            batch_inputs = build_batch_tensors(processor, batch_samples, prefix_text, eos_token)
            batch_inputs = move_inputs_to_device(batch_inputs, DEVICE, model_dtype)

            with torch.autocast("cuda", dtype=model_dtype, enabled=True):
                outputs = model(**batch_inputs)

            sample_losses, sample_tokens = compute_per_sample_losses(
                outputs.logits.detach(),
                batch_inputs["labels"].detach()
            )

            for s, loss_val, token_count in zip(batch_samples, sample_losses, sample_tokens):
                if token_count == 0 or not math.isfinite(loss_val):
                    continue
                all_sample_losses.append(loss_val)
                all_token_counts.append(token_count)
                lang_loss_sum[s["lang_name"]] += loss_val * token_count
                lang_token_sum[s["lang_name"]] += token_count

        except Exception as e:
            print(f"[Val skip] {e}")

    model.train()

    if not all_sample_losses:
        avg_loss = float("nan")
        std_loss = float("nan")
        token_weighted = float("nan")
        by_lang = {}
    else:
        avg_loss = float(np.mean(all_sample_losses))
        std_loss = float(np.std(all_sample_losses))
        token_weighted = float(sum(l * t for l, t in zip(all_sample_losses, all_token_counts)) / sum(all_token_counts))
        by_lang = {
            lang: float(lang_loss_sum[lang] / lang_token_sum[lang])
            for lang in sorted(lang_loss_sum.keys())
            if lang_token_sum[lang] > 0
        }

    logger.log_validation(global_step, epoch, avg_loss, std_loss, token_weighted, by_lang)

    print("\n[Validation]")
    print(f"  Global step: {global_step}")
    print(f"  Epoch: {epoch}")
    print(f"  Val loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"  Token-weighted val loss: {token_weighted:.4f}")
    for lang, loss_val in by_lang.items():
        print(f"  {lang}: {loss_val:.4f}")

    return avg_loss


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_samples, train_csv_path = load_split(TRAIN_DIR, "train.csv")
    val_samples, val_csv_path = load_split(VAL_DIR, "validation.csv")

    print("=" * 80)
    print("Qwen3-ASR-1.7B LoRA Fine-tuning for Assamese/Bengali/Hindi/Odia")
    print("=" * 80)
    print(f"Train CSV: {train_csv_path}")
    print(f"Val CSV:   {val_csv_path}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRAD_ACCUM_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 80)

    lang_counts = defaultdict(int)
    for s in train_samples:
        lang_counts[s["lang_name"]] += 1

    print("\n[Training language distribution]")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")

    model, processor, model_dtype, attn_impl = load_model_and_processor()

    print("\n[Model loaded]")
    print(f"  Compute dtype: {model_dtype}")
    print(f"  Attention implementation: {attn_impl}")

    # IMPORTANT:
    # We DO NOT call prepare_model_for_kbit_training(model) here,
    # because it triggers the exact get_input_embeddings error you hit.

    try:
        model.config.use_cache = False
    except Exception:
        pass

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=LR)

    steps_per_epoch = math.ceil(len(train_samples) / BATCH_SIZE)
    total_optimizer_steps = math.ceil(steps_per_epoch / GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = max(1, int(total_optimizer_steps * WARMUP_RATIO))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(model_dtype == torch.float16))

    prefix_text = render_prefix_text(processor)
    eos_token = processor.tokenizer.eos_token or ""

    all_langs = [s["lang_name"] for s in train_samples]
    logger = TrainingLogger(OUTPUT_DIR, all_langs)

    print("\n[Baseline validation before training]")
    run_validation(
        model=model,
        processor=processor,
        val_samples=val_samples,
        batch_size=BATCH_SIZE,
        prefix_text=prefix_text,
        eos_token=eos_token,
        model_dtype=model_dtype,
        logger=logger,
        global_step=0,
        epoch=0,
    )

    global_step = 0
    optimizer_step = 0
    running_loss_window = deque(maxlen=50)

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 80)

        model.train()
        epoch_samples = bucketed_random_order(train_samples, bucket_size=SORT_BUCKET_SIZE)

        epoch_loss_times_tokens = 0.0
        epoch_tokens = 0
        batch_idx = 0

        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(total=len(epoch_samples), desc=f"Epoch {epoch}", dynamic_ncols=True)

        for start in range(0, len(epoch_samples), BATCH_SIZE):
            batch_idx += 1
            batch_samples = epoch_samples[start:start + BATCH_SIZE]
            file_names = [s["file_name"] for s in batch_samples]

            try:
                batch_inputs = build_batch_tensors(processor, batch_samples, prefix_text, eos_token)
                batch_inputs = move_inputs_to_device(batch_inputs, DEVICE, model_dtype)

                with torch.autocast("cuda", dtype=model_dtype, enabled=True):
                    outputs = model(**batch_inputs)
                    raw_loss = outputs.loss
                    loss = raw_loss / GRAD_ACCUM_STEPS

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                batch_loss = float(raw_loss.detach().item())
                sample_losses, sample_tokens = compute_per_sample_losses(
                    outputs.logits.detach(),
                    batch_inputs["labels"].detach()
                )

                valid_tokens_in_batch = int(sum(sample_tokens))
                epoch_loss_times_tokens += batch_loss * valid_tokens_in_batch
                epoch_tokens += valid_tokens_in_batch

                for s, loss_val, token_count in zip(batch_samples, sample_losses, sample_tokens):
                    if token_count == 0 or not math.isfinite(loss_val):
                        continue
                    logger.lang_loss_accum[s["lang_name"]] += loss_val * token_count
                    logger.lang_token_accum[s["lang_name"]] += token_count

                should_step = (
                    ((batch_idx % GRAD_ACCUM_STEPS) == 0)
                    or ((start + BATCH_SIZE) >= len(epoch_samples))
                )

                did_optimizer_step = False
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    optimizer_step += 1
                    did_optimizer_step = True

                global_step += 1

                running_loss_window.append(batch_loss)
                running_avg = float(sum(running_loss_window) / len(running_loss_window))
                running_std = float(np.std(running_loss_window)) if len(running_loss_window) > 1 else 0.0
                current_lr = float(scheduler.get_last_lr()[0])

                logger.log_train_step(
                    global_step=global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    batch_loss=batch_loss,
                    running_avg=running_avg,
                    running_std=running_std,
                    lr=current_lr,
                    samples_seen=min(start + len(batch_samples), len(epoch_samples)),
                    valid_tokens=valid_tokens_in_batch,
                )

                print("\n" + "-" * 60)
                print(f"Batch {batch_idx} | Global Step {global_step} | Optimizer Step {optimizer_step}")
                print("Files:")
                for name in file_names:
                    print(f"  {name}")
                print(f"Batch Loss: {batch_loss:.4f}")
                print(f"Running Avg Loss: {running_avg:.4f} ± {running_std:.4f}")
                print(
                    "Per-sample loss: "
                    + ", ".join(
                        f"{batch_samples[j]['lang_name']}={sample_losses[j]:.3f}({sample_tokens[j]}t)"
                        for j in range(len(sample_losses))
                    )
                )
                print(f"Valid Tokens: {valid_tokens_in_batch}")
                print(f"LR: {current_lr:.8e}")

                alloc = torch.cuda.memory_allocated() / 1e9
                reserv = torch.cuda.memory_reserved() / 1e9
                print(f"GPU Memory: {alloc:.2f} GB allocated / {reserv:.2f} GB reserved")

                progress.update(len(batch_samples))
                progress.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{current_lr:.2e}")

                if did_optimizer_step and (optimizer_step % LOG_PER_LANG_EVERY == 0):
                    logger.flush_per_language(global_step, epoch)

                if did_optimizer_step and (optimizer_step % VAL_EVERY_N_OPT_STEPS == 0):
                    run_validation(
                        model=model,
                        processor=processor,
                        val_samples=val_samples,
                        batch_size=BATCH_SIZE,
                        prefix_text=prefix_text,
                        eos_token=eos_token,
                        model_dtype=model_dtype,
                        logger=logger,
                        global_step=global_step,
                        epoch=epoch,
                    )

                if global_step % PLOT_EVERY_N_STEPS == 0:
                    save_training_plots(OUTPUT_DIR, logger, suffix=f"_step{global_step}")

                if did_optimizer_step and (optimizer_step % CHECKPOINT_EVERY_N_OPT_STEPS == 0):
                    save_checkpoint(
                        output_dir=OUTPUT_DIR,
                        tag=f"checkpoint_step_{global_step}",
                        model=model,
                        processor=processor,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        global_step=global_step,
                        epoch=epoch,
                    )

            except Exception as e:
                print(f"\n[Skip batch] {e}")
                progress.update(len(batch_samples))
                continue

        progress.close()

        logger.flush_per_language(global_step, epoch)

        epoch_avg_loss = (epoch_loss_times_tokens / epoch_tokens) if epoch_tokens > 0 else float("nan")

        print(f"\n[Epoch {epoch} summary]")
        print(f"  Token-weighted train loss: {epoch_avg_loss:.4f}")
        print(f"  Global token-weighted train loss: {logger.token_weighted_train_avg():.4f}")

        run_validation(
            model=model,
            processor=processor,
            val_samples=val_samples,
            batch_size=BATCH_SIZE,
            prefix_text=prefix_text,
            eos_token=eos_token,
            model_dtype=model_dtype,
            logger=logger,
            global_step=global_step,
            epoch=epoch,
        )

        save_training_plots(OUTPUT_DIR, logger, suffix=f"_epoch{epoch}")

        save_checkpoint(
            output_dir=OUTPUT_DIR,
            tag=f"epoch_{epoch}",
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            epoch=epoch,
        )

    final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    with open(os.path.join(final_dir, "base_model_name.txt"), "w", encoding="utf-8") as f:
        f.write(MODEL_NAME + "\n")

    save_training_plots(OUTPUT_DIR, logger, suffix="_final")

    print("\n" + "=" * 80)
    print("Training complete.")
    print("=" * 80)
    print(f"Final adapter saved to: {final_dir}")
    print(f"Training log CSV:       {logger.train_csv}")
    print(f"Validation log CSV:     {logger.val_csv}")
    print(f"Per-language loss CSV:  {logger.lang_csv}")
    print(f"Final plots saved in:   {OUTPUT_DIR}")


if __name__ == "__main__":
    main()