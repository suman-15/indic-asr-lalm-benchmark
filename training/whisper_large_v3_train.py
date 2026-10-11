"""
Whisper Large-v3 LoRA Fine-tuning for Indic Languages (Assamese, Bengali, Hindi, Odia)
VERSION 6 — Optimized for 40k dataset

Key Design Decisions:
- LoRA r=64 applied to decoder attention + MLP layers (encoder frozen entirely)
- 4-bit NF4 quantization to fit in 24GB VRAM
- Uses native Whisper language tokens for Assamese, Bengali, and Hindi, with Bengali token fallback for Odia.
- Effective batch size 16 (batch 2 × grad accum 8) for stable gradients
- Cosine LR with warmup and min floor (never decays below 10% of peak)
- Label smoothing 0.1 to reduce repetition loops

Changes from v5 (small dataset version):
- LoRA rank 32→64, alpha 64→128 (more capacity for 40k samples)
- Grad accum 4→8 (effective batch 16 for stability)
- LR 1e-4→5e-5 (lower for larger dataset + higher rank)
- Cosine decay min floor 10% (prevents LR going to zero)
- Validation/logging intervals scaled for ~5000 optimizer steps/epoch

OUTPUT: training_log.csv, validation_log.csv, per_language_loss.csv,
        training_curves_*.png, per_language_loss_*.png in OUTPUT_DIR
"""

import os
import csv
import time
import math
import random
import torch
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F

# =========================
# CONFIG
# =========================

MODEL_NAME = "openai/whisper-large-v3"

# Dataset directory — set this to your actual path
# Works on both Linux and Windows
DATASET_DIR = os.path.join(
    os.path.expanduser("~"),
    "datasets",
    "IndicVoices_subset_4lang_combined",
)
# Override: uncomment and set your actual path below
DATASET_DIR = "/path/to/data/IndicVoices_subset_4lang_combined"  # Linux
# DATASET_DIR = r"/path/to/data/indicvocies_finetune"                  # Windows

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

OUTPUT_DIR = os.path.join(DATASET_DIR, "whisper_lora_indic")

BATCH_SIZE = 4  # GPU batch size (sends 2 random audio files per batch)
EPOCHS = 5
LR = 5e-5  # Lower than small-dataset (1e-4) — more data + higher rank = lower LR
WARMUP_RATIO = 0.05  # 5% warmup (longer training = shorter warmup ratio)
LABEL_SMOOTHING = 0.1  # Prevents overconfident predictions → reduces repetition loops
MIN_LR_RATIO = 0.1  # Cosine decay floor: never go below 10% of peak LR
RANDOM_SEED = 42
MAX_AUDIO_SECONDS = 30  # Whisper hard limit is 30 seconds

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = "cuda"
GRAD_ACCUM_STEPS = 8  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 16

# --- Logging & Validation Config ---
# With 40k samples: ~20k micro-steps/epoch → ~2500 optimizer steps/epoch → ~10000 total
VAL_EVERY_N_STEPS = 100  # Run micro-validation every 100 optimizer steps
PLOT_EVERY_N_STEPS = 100  # Save training curve plot every 100 steps
LOG_PER_LANG_EVERY = 500  # Log per-language breakdown every 500 steps

# Disable CPU multithreading (as requested)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =========================
# LANGUAGE CONFIGURATION
# =========================

LANG_CODE_TO_NAME = {
    "as": "Assamese",
    "bn": "Bengali",
    "hi": "Hindi",
    "or": "Odia",
}

ALLOWED_LANG_CODES = set(LANG_CODE_TO_NAME.keys())

LANG_CODE_TO_WHISPER = {
    "as": "assamese",
    "bn": "bengali",
    "hi": "hindi",
    "or": "bengali",  # Whisper tokenizer workaround for Odia
}


def get_language_name(lang_code):
    code = str(lang_code).lower().strip()
    return LANG_CODE_TO_NAME.get(code, code.capitalize())


def get_whisper_language(lang_code):
    code = str(lang_code).lower().strip()
    return LANG_CODE_TO_WHISPER.get(code, "hindi")


def resolve_audio_path(raw_path, audio_dir, split_dir):
    raw = str(raw_path).strip()
    normalized = raw.replace("\\", "/")

    candidates = [raw, normalized]

    filename = os.path.basename(normalized)
    if filename:
        candidates.append(os.path.join(audio_dir, filename))
        candidates.append(os.path.join(split_dir, filename))

    parts = normalized.split("/")
    if "audio" in parts:
        audio_idx = len(parts) - 1 - parts[::-1].index("audio")
        rel_after_audio = "/".join(parts[audio_idx + 1 :]).strip()
        if rel_after_audio:
            candidates.append(os.path.join(audio_dir, rel_after_audio))

    seen = set()
    for cand in candidates:
        c = cand.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.exists(c):
            return c

    return None


# =========================
# LOAD TRAINING METADATA
# =========================


def load_samples_from_dir(data_dir, split_name):
    csv_path = os.path.join(data_dir, f"{split_name}.csv")
    audio_dir = os.path.join(data_dir, "audio")

    print(f"Loading {split_name} CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"CSV columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    if "lang" in df.columns:
        print(f"Unique language codes: {df['lang'].unique()}")

    samples = []
    skipped = 0
    skipped_unsupported_lang = 0

    for _, row in df.iterrows():
        lang_code = str(row["lang"]).strip().lower()
        if lang_code not in ALLOWED_LANG_CODES:
            skipped_unsupported_lang += 1
            continue

        resolved_path = resolve_audio_path(
            str(row["path"]),
            audio_dir=audio_dir,
            split_dir=data_dir,
        )

        if resolved_path is None:
            skipped += 1
            continue

        samples.append(
            {
                "audio": resolved_path,
                "text": str(row["text"]),
                "lang": get_language_name(lang_code),
                "lang_code": lang_code,
                "whisper_lang": get_whisper_language(lang_code),
            }
        )

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} samples (audio files not found)")
    if skipped_unsupported_lang > 0:
        print(f"  ⚠️  Skipped {skipped_unsupported_lang} samples (unsupported language)")
    print(f"  Loaded {len(samples)} samples")
    return samples


samples = load_samples_from_dir(TRAIN_DIR, "train")
random.shuffle(samples)

print(f"\nTotal training samples: {len(samples)}")
train_lang_counts = defaultdict(int)
for s in samples:
    train_lang_counts[s["lang"]] += 1
print("Training language distribution:")
for lang, count in sorted(train_lang_counts.items()):
    print(f"  {lang}: {count}")

val_samples = load_samples_from_dir(VAL_DIR, "validation")
random.shuffle(val_samples)

print(f"\nTotal validation samples: {len(val_samples)}")
val_lang_counts = defaultdict(int)
for s in val_samples:
    val_lang_counts[s["lang"]] += 1
print("Validation language distribution:")
for lang, count in sorted(val_lang_counts.items()):
    print(f"  {lang}: {count}")

# =========================
# LOAD MODEL IN 4-BIT
# =========================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading Whisper processor...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)

TARGET_SR = 16000
print(f"Target sampling rate: {TARGET_SR}")

print("Loading Whisper model in 4-bit...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# =========================
# FREEZE ENCODER ENTIRELY
# =========================

model.model.encoder.requires_grad_(False)
print("✅ Encoder frozen — only decoder will be fine-tuned via LoRA")

# =========================
# APPLY LORA TO DECODER
# =========================
# r=32 is needed to give the adapter enough capacity to learn new scripts
# for Indic scripts that differ from the base model's strongest languages.
# Lower rank (e.g. r=16) causes undertrained adapters that can't override
# Whisper's English/Hindi bias → outputs in wrong script.
# For 40k dataset: r=64 gives more capacity without overfitting risk.

lora_config = LoraConfig(
    r=128,  # Higher rank for 40k dataset — more capacity
    lora_alpha=256,  # alpha/r ratio = 2 (standard)
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ],
    lora_dropout=0.05,  # Lower dropout for larger dataset (less regularization needed)
    bias="none",
    # NOTE: task_type intentionally omitted — see v1 notes
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# =========================
# OPTIMIZER + LR SCHEDULER
# =========================

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")

# Compute total optimizer steps for scheduler
steps_per_epoch = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
total_micro_steps = steps_per_epoch * EPOCHS
total_optimizer_steps = (total_micro_steps + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
warmup_steps = max(1, int(total_optimizer_steps * WARMUP_RATIO))

print(f"\n📋 Scheduler Info:")
print(f"   Steps per epoch: {steps_per_epoch}")
print(f"   Total optimizer steps: {total_optimizer_steps}")
print(f"   Warmup steps: {warmup_steps}")


def cosine_with_warmup(current_step):
    """Linear warmup then cosine decay to MIN_LR_RATIO floor."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(
        max(1, total_optimizer_steps - warmup_steps)
    )
    # Cosine decay with floor — never go below MIN_LR_RATIO of peak LR
    cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(MIN_LR_RATIO, cosine_value)


scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_warmup)

# =========================
# LABEL SMOOTHING LOSS
# =========================
# Prevents the model from becoming too confident about individual tokens,
# which is a major cause of repetition loops in autoregressive decoding.


def compute_label_smoothed_loss_stats(logits, labels):
    """Compute token-level label-smoothed CE stats and mean loss."""
    vocab_size = logits.size(-1)
    per_token_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
        label_smoothing=LABEL_SMOOTHING,
    ).view(labels.shape)

    valid_token_mask = labels.ne(-100)
    per_sample_loss_sum = (per_token_loss * valid_token_mask).sum(dim=1)
    per_sample_token_count = valid_token_mask.sum(dim=1)
    total_token_count = per_sample_token_count.sum()
    total_loss_sum = per_sample_loss_sum.sum()

    if total_token_count.item() == 0:
        mean_loss = logits.new_tensor(0.0)
    else:
        mean_loss = total_loss_sum / total_token_count

    return mean_loss, per_sample_loss_sum, per_sample_token_count


# =========================
# AUDIO LOADING
# =========================


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32")

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // g, sr // g).astype(np.float32)

    max_len = int(TARGET_SR * MAX_AUDIO_SECONDS)
    if len(wav) > max_len:
        wav = wav[:max_len]

    return wav


# =========================
# LOGGING INFRASTRUCTURE
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_CSV_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")
VAL_CSV_PATH = os.path.join(OUTPUT_DIR, "validation_log.csv")
LANG_CSV_PATH = os.path.join(OUTPUT_DIR, "per_language_loss.csv")

with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
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

with open(VAL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["global_step", "epoch", "val_loss", "val_loss_by_lang", "elapsed_sec"]
    )

all_langs = sorted(set(s["lang"] for s in samples))
with open(LANG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["global_step", "epoch"]
        + [f"loss_{lang}" for lang in all_langs]
        + [f"count_{lang}" for lang in all_langs]
    )

train_steps_log = []
train_avg_log = []
val_log = []
per_lang_log = []

lang_loss_token_sum = defaultdict(float)
lang_loss_token_count = defaultdict(int)
lang_loss_sample_count = defaultdict(int)

training_start_time = time.time()
RUNNING_AVG_WINDOW = 50


def get_gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return alloc, reserved
    return 0.0, 0.0


def log_training_step(
    global_step, epoch, batch_idx, batch_loss, running_avg, lr, samples_seen
):
    elapsed = time.time() - training_start_time
    alloc, reserved = get_gpu_mem()

    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
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


def log_per_language_loss(global_step, epoch):
    if not lang_loss_sample_count:
        return

    row = [global_step, epoch]
    lang_avgs = {}

    for lang in all_langs:
        if lang_loss_token_count[lang] > 0:
            avg = lang_loss_token_sum[lang] / lang_loss_token_count[lang]
            lang_avgs[lang] = avg
            row.append(f"{avg:.6f}")
        else:
            row.append("")

    for lang in all_langs:
        row.append(lang_loss_sample_count.get(lang, 0))

    with open(LANG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    per_lang_log.append((global_step, dict(lang_avgs)))

    print("\n  📊 Per-language loss breakdown:")
    for lang in all_langs:
        if lang_loss_token_count[lang] > 0:
            avg = lang_loss_token_sum[lang] / lang_loss_token_count[lang]
            print(f"    {lang}: {avg:.4f} (n={lang_loss_sample_count[lang]})")

    lang_loss_token_sum.clear()
    lang_loss_token_count.clear()
    lang_loss_sample_count.clear()


# =========================
# PREPARE BATCH FUNCTION
# =========================


def prepare_batch(batch_samples):
    audio_arrays = []
    texts = []
    whisper_langs = []
    file_names = []
    batch_langs = []

    for s in batch_samples:
        wav = load_audio(s["audio"])
        audio_arrays.append(wav)
        texts.append(s["text"])
        whisper_langs.append(s["whisper_lang"])
        file_names.append(os.path.basename(s["audio"]))
        batch_langs.append(s["lang"])

    input_features_list = []
    for wav in audio_arrays:
        feats = processor.feature_extractor(
            wav, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features
        input_features_list.append(feats.squeeze(0))

    input_features = torch.stack(input_features_list)

    MAX_LABEL_TOKENS = 448
    labels_list = []
    for i, text in enumerate(texts):
        processor.tokenizer.set_prefix_tokens(
            language=whisper_langs[i],
            task="transcribe",
        )
        label_ids = processor.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LABEL_TOKENS,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels_list.append(label_ids)

    max_label_len = max(l.shape[0] for l in labels_list)
    padded_labels = torch.full(
        (len(labels_list), max_label_len),
        -100,
        dtype=torch.long,
    )
    for i, l in enumerate(labels_list):
        padded_labels[i, : l.shape[0]] = l

    return {
        "input_features": input_features,
        "labels": padded_labels,
        "file_names": file_names,
        "batch_langs": batch_langs,
    }


# =========================
# MICRO-VALIDATION FUNCTION
# =========================


def run_micro_validation(global_step, epoch):
    model.eval()
    val_loss_total = 0.0
    val_token_total = 0
    val_lang_loss = defaultdict(float)
    val_lang_token_count = defaultdict(int)

    with torch.no_grad():
        for vi in range(0, len(val_samples), BATCH_SIZE):
            vbatch = val_samples[vi : vi + BATCH_SIZE]

            try:
                batch_data = prepare_batch(vbatch)
            except Exception as e:
                print(f"  [VAL] Skipping batch at {vi}: {e}")
                continue

            input_features = batch_data["input_features"].to(model.device)
            labels = batch_data["labels"].to(model.device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(
                    input_features=input_features,
                    labels=labels,
                )
                _, per_sample_loss_sum, per_sample_token_count = (
                    compute_label_smoothed_loss_stats(outputs.logits, labels)
                )

            val_loss_total += per_sample_loss_sum.sum().item()
            val_token_total += int(per_sample_token_count.sum().item())

            for idx, lang in enumerate(batch_data["batch_langs"]):
                lang_token_count = int(per_sample_token_count[idx].item())
                if lang_token_count <= 0:
                    continue
                val_lang_loss[lang] += float(per_sample_loss_sum[idx].item())
                val_lang_token_count[lang] += lang_token_count

    model.train()

    if val_token_total == 0:
        return float("nan")

    avg_val_loss = val_loss_total / val_token_total

    lang_breakdown = {}
    for lang in sorted(val_lang_loss.keys()):
        if val_lang_token_count[lang] > 0:
            lang_breakdown[lang] = val_lang_loss[lang] / val_lang_token_count[lang]

    elapsed = time.time() - training_start_time
    with open(VAL_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                global_step,
                epoch,
                f"{avg_val_loss:.6f}",
                str(lang_breakdown),
                f"{elapsed:.1f}",
            ]
        )

    val_log.append((global_step, avg_val_loss))

    print(f"\n  🔍 Validation Loss: {avg_val_loss:.4f}")
    for lang, loss in sorted(lang_breakdown.items()):
        print(f"     {lang}: {loss:.4f}")

    return avg_val_loss


# =========================
# PLOTTING
# =========================


def save_training_plots(suffix=""):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if train_steps_log:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))

            steps, losses = zip(*train_steps_log)
            ax.plot(
                steps,
                losses,
                alpha=0.3,
                linewidth=0.5,
                label="Batch Loss",
                color="blue",
            )

            if train_avg_log:
                avg_steps, avg_losses = zip(*train_avg_log)
                ax.plot(
                    avg_steps,
                    avg_losses,
                    linewidth=2,
                    label=f"Running Avg ({RUNNING_AVG_WINDOW})",
                    color="darkblue",
                )

            if val_log:
                val_steps, val_losses = zip(*val_log)
                ax.plot(
                    val_steps,
                    val_losses,
                    "ro-",
                    markersize=4,
                    linewidth=1.5,
                    label="Validation Loss",
                )

            ax.set_xlabel("Global Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training Progress")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, f"training_curves{suffix}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  📈 Training curves saved to: {plot_path}")

        if per_lang_log:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))

            lang_series = defaultdict(lambda: ([], []))
            for step, lang_dict in per_lang_log:
                for lang, loss_val in lang_dict.items():
                    lang_series[lang][0].append(step)
                    lang_series[lang][1].append(loss_val)

            for lang in sorted(lang_series.keys()):
                steps_l, losses_l = lang_series[lang]
                ax.plot(
                    steps_l,
                    losses_l,
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=lang,
                )

            ax.set_xlabel("Global Step")
            ax.set_ylabel("Average Loss")
            ax.set_title("Per-Language Training Loss")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            lang_plot_path = os.path.join(OUTPUT_DIR, f"per_language_loss{suffix}.png")
            plt.savefig(lang_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  📈 Per-language loss plot saved to: {lang_plot_path}")

    except ImportError:
        print("  ⚠️ matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"  ⚠️ Plot generation failed: {e}")


# =========================
# TRAINING LOOP
# =========================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(
    f"Batch size: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM_STEPS} | Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}"
)
print(f"Learning rate: {LR} (cosine decay + {warmup_steps}-step warmup)")
print(f"Label smoothing: {LABEL_SMOOTHING}")
print(
    f"LoRA rank: {lora_config.r} | LoRA alpha: {lora_config.lora_alpha} | Dropout: {lora_config.lora_dropout}"
)
print(f"Epochs: {EPOCHS}")
print(f"Training samples: {len(samples)}")
print(f"Validation samples: {len(val_samples)}")
print(f"Total optimizer steps: {total_optimizer_steps}")
print("=" * 60)

model.train()

global_step = 0
optimizer_step = 0
running_loss_window = []
best_val_loss = float("inf")
last_validated_optimizer_step = -1

for epoch in range(EPOCHS):
    print(f"\n{'=' * 40}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'=' * 40}")

    random.shuffle(samples)

    epoch_loss = 0.0
    step_count = 0
    accum_counter = 0

    progress = tqdm(
        total=len(samples),
        desc=f"Epoch {epoch + 1}",
        unit="samples",
        dynamic_ncols=True,
    )

    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i : i + BATCH_SIZE]

        try:
            batch_data = prepare_batch(batch)
        except Exception as e:
            print(f"\n⚠️ Skipping batch at index {i}: {e}")
            progress.update(len(batch))
            continue

        input_features = batch_data["input_features"].to(model.device)
        labels = batch_data["labels"].to(model.device)
        file_names = batch_data["file_names"]
        batch_langs = batch_data["batch_langs"]

        # =========================
        # FORWARD PASS (with label smoothing)
        # =========================

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(
                input_features=input_features,
                labels=labels,
            )

            # Recompute loss with label smoothing
            # IMPORTANT: Whisper is encoder-decoder — it already aligns logits[i] → labels[i]
            # internally via decoder_input_ids shifting. Do NOT shift again here.
            # The old shift_logits/shift_labels caused off-by-one misalignment that
            # corrupted ALL training (model learns wrong token → falls back to English).
            mean_loss, per_sample_loss_sum, per_sample_token_count = (
                compute_label_smoothed_loss_stats(outputs.logits, labels)
            )
            loss = mean_loss
            loss = loss / GRAD_ACCUM_STEPS

        # =========================
        # BACKPROP (with gradient accumulation)
        # =========================

        scaler.scale(loss).backward()
        accum_counter += 1

        should_step = accum_counter == GRAD_ACCUM_STEPS or (i + BATCH_SIZE) >= len(
            samples
        )
        if should_step:
            scaler.unscale_(optimizer)
            if accum_counter < GRAD_ACCUM_STEPS:
                rescale = GRAD_ACCUM_STEPS / float(accum_counter)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(rescale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            accum_counter = 0

        # =========================
        # LOGGING
        # =========================

        batch_loss = loss.item() * GRAD_ACCUM_STEPS
        epoch_loss += batch_loss
        step_count += 1
        global_step += 1

        running_loss_window.append(batch_loss)
        if len(running_loss_window) > RUNNING_AVG_WINDOW:
            running_loss_window.pop(0)
        running_avg = sum(running_loss_window) / len(running_loss_window)

        for idx, lang in enumerate(batch_langs):
            lang_loss_sample_count[lang] += 1
            lang_token_count = int(per_sample_token_count[idx].item())
            if lang_token_count <= 0:
                continue
            lang_loss_token_sum[lang] += float(per_sample_loss_sum[idx].item())
            lang_loss_token_count[lang] += lang_token_count

        samples_seen = i + len(batch)

        current_lr = optimizer.param_groups[0]["lr"]
        log_training_step(
            global_step,
            epoch + 1,
            step_count,
            batch_loss,
            running_avg,
            current_lr,
            samples_seen,
        )

        print("\n------------------------------------")
        print(
            f"Batch {step_count} | Global Step {global_step} | Optimizer Step {optimizer_step}"
        )
        print(f"LR: {current_lr:.2e}")
        print("Files:")
        for idx, f in enumerate(file_names):
            print(f"  [{batch_langs[idx]}] {f}")
        print(f"Batch Loss: {batch_loss:.4f} | Running Avg: {running_avg:.4f}")

        alloc, reserved = get_gpu_mem()
        print(f"GPU Memory: {alloc:.1f}GB alloc / {reserved:.1f}GB reserved")

        if global_step % LOG_PER_LANG_EVERY == 0:
            log_per_language_loss(global_step, epoch + 1)

        if (
            optimizer_step > 0
            and optimizer_step % VAL_EVERY_N_STEPS == 0
            and optimizer_step != last_validated_optimizer_step
        ):
            val_loss = run_micro_validation(global_step, epoch + 1)
            last_validated_optimizer_step = optimizer_step
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_save_path = os.path.join(OUTPUT_DIR, "best")
                os.makedirs(best_save_path, exist_ok=True)
                model.save_pretrained(best_save_path)
                processor.save_pretrained(best_save_path)
                print(f"  💾 New best model saved (val_loss={best_val_loss:.4f})")

        if global_step % PLOT_EVERY_N_STEPS == 0:
            save_training_plots(suffix=f"_step{global_step}")

        progress.update(len(batch))

    progress.close()

    if lang_loss_sample_count:
        log_per_language_loss(global_step, epoch + 1)

    avg_loss = epoch_loss / max(step_count, 1)
    print(f"\n✅ Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    print(f"\n--- End-of-Epoch Validation ---")
    val_loss = run_micro_validation(global_step, epoch + 1)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_save_path = os.path.join(OUTPUT_DIR, "best")
        os.makedirs(best_save_path, exist_ok=True)
        model.save_pretrained(best_save_path)
        processor.save_pretrained(best_save_path)
        print(f"  💾 New best model saved (val_loss={best_val_loss:.4f})")

    save_training_plots(suffix=f"_epoch{epoch + 1}")

    save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving LoRA adapter to: {save_path}")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)


# =========================
# SAVE FINAL MODEL + FINAL PLOTS
# =========================

print("\n" + "=" * 60)
print("Training complete. Saving final model.")
print("=" * 60)

model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

save_training_plots(suffix="_final")

elapsed_total = time.time() - training_start_time
print(f"\n📊 TRAINING SUMMARY")
print(f"{'=' * 50}")
print(f"Total time: {elapsed_total / 3600:.2f} hours ({elapsed_total:.0f} seconds)")
print(f"Total optimizer steps: {optimizer_step}")
print(f"Total batches processed: {global_step}")
print(f"Samples trained: {len(samples)} x {EPOCHS} epochs")
print(f"Validation samples: {len(val_samples)}")
print(f"Best validation loss: {best_val_loss:.4f}")

if val_log:
    first_val = val_log[0][1]
    last_val = val_log[-1][1]
    print(f"First val loss: {first_val:.4f}")
    print(f"Last val loss: {last_val:.4f}")
    print(f"Val loss change: {last_val - first_val:+.4f}")
    if last_val > first_val * 1.1:
        print(f"⚠️  WARNING: Validation loss increased — possible overfitting!")
    else:
        print(f"✅ No significant overfitting detected.")

print(f"\nOutput files:")
print(f"  📄 {LOG_CSV_PATH}")
print(f"  📄 {VAL_CSV_PATH}")
print(f"  📄 {LANG_CSV_PATH}")
print(f"  📈 {os.path.join(OUTPUT_DIR, 'training_curves_final.png')}")
print(f"  📈 {os.path.join(OUTPUT_DIR, 'per_language_loss_final.png')}")

print(f"\n✅ Final LoRA adapter saved to: {OUTPUT_DIR}")
print(f"✅ Best LoRA adapter saved to: {os.path.join(OUTPUT_DIR, 'best')}")
print(f"   To load best model for inference:")
print(
    f"   model = PeftModel.from_pretrained(base_model, '{os.path.join(OUTPUT_DIR, 'best')}')"
)
