"""
MMS-1b-all LoRA Fine-tuning for Multilingual Indic ASR (Assamese, Bengali, Hindi)
==================================================================================

Key Design Decisions:
- LoRA r=64 applied to encoder attention layers (q_proj, k_proj, v_proj, out_proj)
- NO quantization — full FP32 model with FP16 mixed precision for speed
- Unified vocabulary built from all 3 languages' training text
- CTC loss (Wav2Vec2ForCTC native)
- Batch size 2, gradient accumulation 8 → effective batch 16
- Cosine LR with warmup and 10% floor
- Label smoothing via CTC (not applicable — CTC has its own blank handling)

Dataset: IndicVoices 15k (5000 each: Assamese, Bengali, Hindi)
Output: LoRA adapter + unified tokenizer + training_log.csv, validation_log.csv,
        per_language_loss.csv, training_curves_*.png, per_language_loss_*.png

MMS Language codes (ISO 639-3): asm (Assamese), ben (Bengali), hin (Hindi)
"""

import os
import csv
import sys
import time
import json
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
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

# =========================
# CONFIG
# =========================

MODEL_NAME = "facebook/mms-1b-all"

# Dataset directory — set this to your actual path
DATASET_DIR = os.path.join(
    os.path.expanduser("~"), "datasets", "indicvocies_finetune_15k"
)
# Override: uncomment and set your actual path below
DATASET_DIR = "/path/to/data/IndicVoices_finetune_15k"  # Linux
# DATASET_DIR = r"/path/to/data/indicvocies_finetune_15k_mini"                # Windows

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

OUTPUT_DIR = os.path.join(DATASET_DIR, "mms_lora_indic")

BATCH_SIZE = 2  # GPU batch size — sends 2 random audio files per batch
EPOCHS = 3
LR = 3e-4  # Higher than Whisper because LoRA on Wav2Vec2 encoder needs it
WARMUP_RATIO = 0.05
MIN_LR_RATIO = 0.1  # Cosine decay floor: never go below 10% of peak LR
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = "cuda"
GRAD_ACCUM_STEPS = 8  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 16

# --- Logging & Validation Config ---
# 15k samples / batch 2 = 7500 micro-steps/epoch → ~937 optimizer steps/epoch
VAL_EVERY_N_STEPS = 200  # Run validation every 200 optimizer steps
PLOT_EVERY_N_STEPS = 200  # Save training curve plot every 200 steps
LOG_PER_LANG_EVERY = 500  # Log per-language breakdown every 500 global steps

# Disable CPU multithreading (as requested)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Faster convolutions for consistent input sizes
torch.backends.cudnn.deterministic = False  # Speed over determinism

TARGET_SR = 16000  # MMS expects 16kHz audio

# =========================
# LANGUAGE CONFIGURATION
# =========================

# CSV 'lang' column values → human-readable names
LANG_CODE_TO_NAME = {
    "as": "Assamese",
    "bn": "Bengali",
    "hi": "Hindi",
}

ALLOWED_LANG_CODES = set(LANG_CODE_TO_NAME.keys())

# CSV lang codes → MMS ISO 639-3 codes
LANG_CODE_TO_MMS = {
    "as": "asm",
    "bn": "ben",
    "hi": "hin",
}


def get_language_name(lang_code):
    code = str(lang_code).lower().strip()
    return LANG_CODE_TO_NAME.get(code, code.capitalize())


def get_mms_lang(lang_code):
    code = str(lang_code).lower().strip()
    return LANG_CODE_TO_MMS.get(code, "hin")


def resolve_audio_path(raw_path, audio_dir, split_dir):
    """Resolve audio file path from CSV (handles Windows/Linux paths)."""
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

        text = str(row.get("text", row.get("normalized", ""))).strip()
        if not text:
            skipped += 1
            continue

        samples.append(
            {
                "audio": resolved_path,
                "text": text,
                "lang": get_language_name(lang_code),
                "lang_code": lang_code,
                "mms_lang": get_mms_lang(lang_code),
            }
        )

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} samples (audio not found or empty text)")
    if skipped_unsupported_lang > 0:
        print(f"  ⚠️  Skipped {skipped_unsupported_lang} samples (unsupported language)")
    print(f"  ✅ Loaded {len(samples)} samples")
    return samples


print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

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
# BUILD UNIFIED VOCABULARY
# =========================

print("\n" + "=" * 60)
print("BUILDING UNIFIED VOCABULARY")
print("=" * 60)

# Collect all unique characters from training + validation text
all_text = [s["text"] for s in samples] + [s["text"] for s in val_samples]
vocab_chars = set()
for text in all_text:
    vocab_chars.update(list(text))

# Remove special characters that shouldn't be in vocab
vocab_chars.discard(" ")

# Build vocab dict: character → id
# Reserve special tokens: <pad>=0, <s>=1, </s>=2, <unk>=3, |=4 (word boundary/space)
vocab_dict = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
for idx, char in enumerate(sorted(vocab_chars)):
    vocab_dict[char] = idx + 5

print(f"Vocabulary size: {len(vocab_dict)} tokens")
print(f"Unique characters: {len(vocab_chars)}")

# Save vocabulary as JSON (nested format required by MMS tokenizer)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MMS uses a nested vocab with target_lang as key
# We create a unified "multilingual" vocab
unified_vocab = {"multilingual": vocab_dict}
vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(unified_vocab, f, ensure_ascii=False, indent=2)
print(f"Vocabulary saved to: {vocab_path}")


# =========================
# LOAD MODEL & BUILD PROCESSOR
# =========================

print("\n" + "=" * 60)
print("LOADING MMS MODEL")
print("=" * 60)

# Create tokenizer from our unified vocabulary
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path,
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|",
    target_lang="multilingual",
)

# Create feature extractor (normalizes audio)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=TARGET_SR,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

# Combine into processor
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

print("Loading MMS-1b-all model (full precision, no quantization)...")
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.0,  # Disable SpecAugment masking during fine-tuning
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(vocab_dict),
    ignore_mismatched_sizes=True,  # Required: new vocab size differs from pretrained
)

# Freeze the feature extractor (CNN frontend) — keep encoder trainable via LoRA
model.freeze_feature_encoder()
print("✅ Feature extractor (CNN) frozen")

# Move to GPU
model = model.to(DEVICE)
print(f"✅ Model loaded on {DEVICE}")

# =========================
# APPLY LORA TO ENCODER
# =========================

print("\n" + "=" * 60)
print("APPLYING LORA")
print("=" * 60)

# Target the attention projection layers in the Wav2Vec2 encoder
# These are the standard attention modules in each transformer layer
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["lm_head"],  # Also train the new CTC output head
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =========================
# OPTIMIZER + LR SCHEDULER
# =========================

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=0.01,
)
scaler = torch.amp.GradScaler("cuda")

# Compute total optimizer steps for scheduler
steps_per_epoch = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
total_micro_steps = steps_per_epoch * EPOCHS
total_optimizer_steps = (total_micro_steps + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
warmup_steps = max(1, int(total_optimizer_steps * WARMUP_RATIO))

print(f"\n📋 Scheduler Info:")
print(f"   Steps per epoch (micro): {steps_per_epoch}")
print(f"   Total optimizer steps: {total_optimizer_steps}")
print(f"   Warmup steps: {warmup_steps}")


def cosine_with_warmup(current_step):
    """Linear warmup then cosine decay to MIN_LR_RATIO floor."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(
        max(1, total_optimizer_steps - warmup_steps)
    )
    cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(MIN_LR_RATIO, cosine_value)


scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_warmup)


# =========================
# AUDIO LOADING
# =========================


def load_audio(path):
    """Load audio file, convert to mono, resample to 16kHz. Use full clip."""
    wav, sr = sf.read(path, dtype="float32")

    # Convert stereo to mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // g, sr // g).astype(np.float32)

    return wav


# =========================
# LOGGING INFRASTRUCTURE
# =========================

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

lang_loss_sum = defaultdict(float)
lang_loss_count = defaultdict(int)
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
        if lang_loss_count[lang] > 0:
            avg = lang_loss_sum[lang] / lang_loss_count[lang]
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
        if lang_loss_count[lang] > 0:
            avg = lang_loss_sum[lang] / lang_loss_count[lang]
            print(f"    {lang}: {avg:.4f} (n={lang_loss_sample_count[lang]})")

    # Reset accumulators
    lang_loss_sum.clear()
    lang_loss_count.clear()
    lang_loss_sample_count.clear()


# =========================
# PREPARE BATCH FUNCTION
# =========================


def prepare_batch(batch_samples):
    """Prepare a batch of audio samples for Wav2Vec2ForCTC.

    Returns input_values (normalized waveforms), attention_mask, labels (token ids),
    and metadata (file_names, batch_langs).
    """
    audio_arrays = []
    texts = []
    file_names = []
    batch_langs = []

    for s in batch_samples:
        wav = load_audio(s["audio"])
        audio_arrays.append(wav)
        texts.append(s["text"])
        file_names.append(os.path.basename(s["audio"]))
        batch_langs.append(s["lang"])

    # Process audio through feature extractor (normalizes + pads)
    inputs = processor.feature_extractor(
        audio_arrays,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_values = inputs.input_values
    attention_mask = inputs.attention_mask

    # Tokenize text labels
    # Replace spaces with | (word delimiter) for CTC
    labels_list = []
    for text in texts:
        # CTC tokenization: character-level with | as word boundary
        text_with_delim = text.replace(" ", "|")
        encoded = processor.tokenizer(
            text_with_delim,
            return_tensors="pt",
            padding=False,
        )
        labels_list.append(encoded.input_ids.squeeze(0))

    # Pad labels to same length (pad with -100 for CTC loss ignoring)
    max_label_len = max(l.shape[0] for l in labels_list)
    padded_labels = torch.full(
        (len(labels_list), max_label_len),
        -100,
        dtype=torch.long,
    )
    for i, l in enumerate(labels_list):
        padded_labels[i, : l.shape[0]] = l

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": padded_labels,
        "file_names": file_names,
        "batch_langs": batch_langs,
    }


# =========================
# MICRO-VALIDATION FUNCTION
# =========================


def run_micro_validation(global_step, epoch):
    """Run validation on all validation samples and return average loss."""
    model.eval()
    val_loss_total = 0.0
    val_count = 0
    val_lang_loss = defaultdict(float)
    val_lang_count = defaultdict(int)

    with torch.no_grad():
        for vi in range(0, len(val_samples), BATCH_SIZE):
            vbatch = val_samples[vi : vi + BATCH_SIZE]

            try:
                batch_data = prepare_batch(vbatch)
            except Exception as e:
                print(f"  [VAL] Skipping batch at {vi}: {e}")
                continue

            input_values = batch_data["input_values"].to(DEVICE)
            attention_mask = batch_data["attention_mask"].to(DEVICE)
            labels = batch_data["labels"].to(DEVICE)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            if loss is not None and not torch.isnan(loss):
                batch_loss_val = loss.item()
                batch_size_actual = len(vbatch)
                val_loss_total += batch_loss_val * batch_size_actual
                val_count += batch_size_actual

                # Per-language tracking
                for idx, lang in enumerate(batch_data["batch_langs"]):
                    val_lang_loss[lang] += batch_loss_val
                    val_lang_count[lang] += 1

    model.train()

    if val_count == 0:
        return float("nan")

    avg_val_loss = val_loss_total / val_count

    lang_breakdown = {}
    for lang in sorted(val_lang_loss.keys()):
        if val_lang_count[lang] > 0:
            lang_breakdown[lang] = val_lang_loss[lang] / val_lang_count[lang]

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
    for lang, loss_val in sorted(lang_breakdown.items()):
        print(f"     {lang}: {loss_val:.4f}")

    return avg_val_loss


# =========================
# PLOTTING
# =========================


def save_training_plots(suffix=""):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- Training Curves Plot ---
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
            ax.set_title("MMS-1b-all LoRA Training Progress")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, f"training_curves{suffix}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  📈 Training curves saved to: {plot_path}")

        # --- Per-Language Loss Plot ---
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
            ax.set_title("MMS-1b-all Per-Language Training Loss")
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
    f"Batch size: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM_STEPS} | "
    f"Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}"
)
print(f"Learning rate: {LR} (cosine decay + {warmup_steps}-step warmup)")
print(
    f"LoRA rank: {lora_config.r} | LoRA alpha: {lora_config.lora_alpha} | "
    f"Dropout: {lora_config.lora_dropout}"
)
print(f"Epochs: {EPOCHS}")
print(f"Training samples: {len(samples)}")
print(f"Validation samples: {len(val_samples)}")
print(f"Total optimizer steps: {total_optimizer_steps}")
print(f"Vocabulary size: {len(vocab_dict)}")
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

    # Shuffle training data each epoch
    random.shuffle(samples)

    epoch_loss = 0.0
    step_count = 0
    accum_counter = 0

    # tqdm progress bar showing overall file progress
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

        input_values = batch_data["input_values"].to(DEVICE)
        attention_mask = batch_data["attention_mask"].to(DEVICE)
        labels = batch_data["labels"].to(DEVICE)
        file_names = batch_data["file_names"]
        batch_langs = batch_data["batch_langs"]

        # =========================
        # FORWARD PASS (CTC Loss)
        # =========================

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            if loss is None or torch.isnan(loss):
                print(f"\n⚠️ NaN/None loss at batch {step_count}, skipping")
                progress.update(len(batch))
                continue

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
            # Rescale gradients if partial accumulation at end of epoch
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

        # Per-language loss tracking
        for idx, lang in enumerate(batch_langs):
            lang_loss_sample_count[lang] += 1
            lang_loss_sum[lang] += batch_loss
            lang_loss_count[lang] += 1

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

        # Print batch info
        print("\n------------------------------------")
        print(
            f"Batch {step_count} | Global Step {global_step} | "
            f"Optimizer Step {optimizer_step}"
        )
        print(f"LR: {current_lr:.2e}")
        print("Files:")
        for idx, f in enumerate(file_names):
            print(f"  [{batch_langs[idx]}] {f}")
        print(f"Batch Loss: {batch_loss:.4f} | Running Avg: {running_avg:.4f}")

        alloc, reserved = get_gpu_mem()
        print(f"GPU Memory: {alloc:.1f}GB alloc / {reserved:.1f}GB reserved")

        # Per-language logging at intervals
        if global_step % LOG_PER_LANG_EVERY == 0:
            log_per_language_loss(global_step, epoch + 1)

        # Validation at intervals
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
                model.save_pretrained(
                    best_save_path,
                    save_embedding_layers=False,  # Wav2Vec2ForCTC has no input token embeddings.
                )
                processor.save_pretrained(best_save_path)
                print(f"  💾 New best model saved (val_loss={best_val_loss:.4f})")

        # Plot at intervals
        if global_step % PLOT_EVERY_N_STEPS == 0:
            save_training_plots(suffix=f"_step{global_step}")

        progress.update(len(batch))

    progress.close()

    # End-of-epoch per-language logging
    if lang_loss_sample_count:
        log_per_language_loss(global_step, epoch + 1)

    avg_loss = epoch_loss / max(step_count, 1)
    print(f"\n✅ Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # End-of-epoch validation
    print(f"\n--- End-of-Epoch Validation ---")
    val_loss = run_micro_validation(global_step, epoch + 1)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_save_path = os.path.join(OUTPUT_DIR, "best")
        os.makedirs(best_save_path, exist_ok=True)
        model.save_pretrained(
            best_save_path,
            save_embedding_layers=False,  # Wav2Vec2ForCTC has no input token embeddings.
        )
        processor.save_pretrained(best_save_path)
        print(f"  💾 New best model saved (val_loss={best_val_loss:.4f})")

    save_training_plots(suffix=f"_epoch{epoch + 1}")

    # Save epoch checkpoint
    save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving LoRA adapter to: {save_path}")
    model.save_pretrained(
        save_path,
        save_embedding_layers=False,  # Wav2Vec2ForCTC has no input token embeddings.
    )
    processor.save_pretrained(save_path)


# =========================
# SAVE FINAL MODEL + FINAL PLOTS
# =========================

print("\n" + "=" * 60)
print("Training complete. Saving final model.")
print("=" * 60)

model.save_pretrained(
    OUTPUT_DIR,
    save_embedding_layers=False,  # Wav2Vec2ForCTC has no input token embeddings.
)
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
print(f"   To load for inference:")
print(f"   base_model = Wav2Vec2ForCTC.from_pretrained('{MODEL_NAME}', ...)")
print(
    f"   model = PeftModel.from_pretrained(base_model, '{os.path.join(OUTPUT_DIR, 'best')}')"
)
