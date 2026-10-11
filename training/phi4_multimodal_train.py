"""
Phi-4-Multimodal-Instruct LoRA Fine-tuning for Indic Languages

This script fine-tunes the Microsoft Phi-4-multimodal-instruct model on audio samples
from 4 Indic languages: Assamese, Bengali, Hindi, and Odia.

Key Features:
- Official full-precision/bf16 loading path for stable fine-tuning
- LoRA fine-tuning on Speech adapter
- Batch processing with batch size 2
- Random audio file selection
- Complete audio clip usage (no chunking)
- Loss logging with CSV and PNG plots
- Validation loss evaluation
- tqdm progress bar

Dataset Structure:
    indicvoices_finetune_40k_mini/
    |__test/
    |   |__audio/
    |   |__test.csv
    |__train/
    |   |__audio/
    |   |__train.csv
    |__validation/
        |__audio/
        |__validation.csv

Output:
    - training_log.csv: Step-level training losses
    - validation_log.csv: Validation losses
    - per_language_loss.csv: Per-language loss breakdown
    - training_curves.png: Training/validation loss plots
    - Fine-tuned model saved for inference
"""

import os
import gc
import csv
import time
import random
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Compatibility shim: SlidingWindowCache was removed/restructured in transformers 5.x
# The Phi-4-multimodal model's remote code imports it, so we inject a stub if missing.
import transformers.cache_utils as _cache_utils

if not hasattr(_cache_utils, "SlidingWindowCache"):
    from transformers import DynamicCache as _DynCache

    class _SlidingWindowCache(_DynCache):
        """Fallback stub for transformers versions that removed SlidingWindowCache."""

        def __init__(self, *args, **kwargs):
            kwargs.pop("sliding_window", None)
            super().__init__(*args, **kwargs)

    _cache_utils.SlidingWindowCache = _SlidingWindowCache
    import transformers as _tf

    _tf.SlidingWindowCache = _SlidingWindowCache

# =========================
# CONFIGURATION
# =========================

MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"

# Dataset paths - UPDATE THESE TO YOUR ACTUAL PATHS
DATA_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/train"
VAL_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/validation"

OUTPUT_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/phi4_multimodal_lora_indic"

# Training parameters
BATCH_SIZE = 1  # Reduced from 2 to avoid OOM with long audio clips
EPOCHS = 5
LR = 2e-4
RANDOM_SEED = 42
MAX_AUDIO_SECONDS = 30  # Maximum audio duration in seconds
USE_FULL_AUDIO_CLIP = True

# Gradient accumulation for effective larger batch
GRAD_ACCUM_STEPS = 8  # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 8

# Logging config
VAL_EVERY_N_STEPS = 100
PLOT_EVERY_N_STEPS = 100
LOG_PER_LANG_EVERY = 500
RUN_VALIDATION_EVERY_N_STEPS = VAL_EVERY_N_STEPS
RUN_VALIDATION_EACH_EPOCH = True
TRAIN_PROGRESS_UNIT = "samples"

# Disable CPU multithreading
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LANGUAGE CODE MAPPING
# =========================

LANG_CODE_TO_NAME = {
    "as": "Assamese",
    "bn": "Bengali",
    "hi": "Hindi",
    "or": "Odia",
}

# Language-specific prompts for transcription
LANG_PROMPTS = {
    "Assamese": "Transcribe the audio clip into Assamese text.",
    "Bengali": "Transcribe the audio clip into Bengali text.",
    "Hindi": "Transcribe the audio clip into Hindi text.",
    "Odia": "Transcribe the audio clip into Odia text.",
}

ANSWER_SUFFIX = "<|end|>"
_IGNORE_INDEX = -100


def get_language_name(lang_code):
    """Convert language code to full name."""
    code = str(lang_code).lower().strip()
    return LANG_CODE_TO_NAME.get(code, code.capitalize())


# =========================
# AUDIO LOADING
# =========================


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio using soundfile and resample if needed."""
    wav, sr = sf.read(path, dtype="float32")

    # Stereo to mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # Resample only if needed
    if sr != target_sr:
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)

    # Keep full clip by default; optionally cap length if explicitly requested.
    if not USE_FULL_AUDIO_CLIP:
        max_len = int(target_sr * MAX_AUDIO_SECONDS)
        if len(wav) > max_len:
            wav = wav[:max_len]

    return wav, target_sr


# =========================
# DATASET CLASS
# =========================


class IndicVoiceDataset(torch.utils.data.Dataset):
    """Dataset for Indic voice transcription."""

    def __init__(self, samples: List[Dict], processor, training: bool = True):
        self.samples = samples
        self.processor = processor
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio_path = sample["audio"]
        text = sample["text"]
        lang = sample["lang"]

        try:
            wav, sr = load_audio(audio_path)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return a dummy sample
            wav = np.zeros(16000, dtype=np.float32)
            sr = 16000
            text = ""

        # Build prompt
        instruction = LANG_PROMPTS.get(
            lang, f"Transcribe the audio clip into {lang} text."
        )

        user_message = {
            "role": "user",
            "content": f"<|audio_1|>\n{instruction}",
        }

        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(text=prompt, audios=[(wav, sr)], return_tensors="pt")

        # Build answer
        answer = f"{text}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids

        # For both training and validation-loss computation, labels must align
        # with full sequence length (prompt + answer) to avoid shape mismatch.
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1] :] = answer_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
            "audio_path": audio_path,
            "lang": lang,
        }


# =========================
# COLLATE FUNCTION
# =========================


def pad_sequence(sequences, padding_side="right", padding_value=0):
    """Pad a list of sequences to the same length."""
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """Concatenate tensors along dim, padding to max for other dims."""
    ndim = tensors[0].dim()
    assert all(t.dim() == ndim for t in tensors[1:]), (
        "All tensors must have same dimensions"
    )

    out_size = list(max(t.shape[i] for t in tensors) for i in range(ndim))
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[tuple(slices)] = t
        index += t.shape[dim]

    return output


def indic_collate_fn(batch):
    """Custom collate function for Indic voice dataset."""
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    audio_paths = []
    langs = []

    for inputs in batch:
        input_ids_list.append(inputs["input_ids"][0])
        labels_list.append(inputs["labels"][0])
        input_audio_embeds_list.append(inputs["input_audio_embeds"])
        audio_embed_sizes_list.append(inputs["audio_embed_sizes"])
        audio_attention_mask_list.append(
            inputs["input_audio_embeds"].new_full(
                (inputs["input_audio_embeds"].size(1),), True, dtype=torch.bool
            )
        )
        audio_paths.append(inputs["audio_path"])
        langs.append(inputs["lang"])

    try:
        input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=0)
        labels = pad_sequence(labels_list, padding_side="left", padding_value=0)
        audio_attention_mask = (
            pad_sequence(
                audio_attention_mask_list, padding_side="right", padding_value=False
            )
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        print(f"Collate error: {e}")
        raise

    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "input_mode": 2,  # speech mode
            "audio_paths": audio_paths,
            "langs": langs,
        }
    )


# =========================
# STOPPING CRITERIA
# =========================


class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria for multiple stop-tokens with batch support."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1):
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(
            batch_size, dtype=torch.long, device=stop_tokens.device
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        generated_inputs = torch.eq(
            input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens
        )
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)


# =========================
# LOAD TRAINING METADATA
# =========================


def load_samples(data_dir: str) -> List[Dict]:
    """Load samples from CSV file."""
    samples = []
    skipped_missing = 0
    missing_examples = []
    dir_name = os.path.basename(data_dir)
    csv_path = os.path.join(data_dir, f"{dir_name}.csv")

    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"CSV columns: {list(df.columns)}")
    print(f"Unique language codes: {df['lang'].unique()}")

    audio_dir = os.path.join(data_dir, "audio")

    for _, row in df.iterrows():
        # Handle path robustly across Windows/Linux CSVs.
        raw_path = str(row["path"]).strip()
        normalized_path = raw_path.replace("\\", "/")

        path_candidates = []

        # Candidate 1: path as-is (works when CSV already has Linux absolute paths).
        path_candidates.append(raw_path)

        # Candidate 2: slash-normalized form.
        path_candidates.append(normalized_path)

        # Candidate 3: local dataset audio dir + filename from path (critical for Windows paths on Linux).
        filename = os.path.basename(normalized_path)
        if filename:
            path_candidates.append(os.path.join(audio_dir, filename))
            path_candidates.append(os.path.join(data_dir, filename))

        # Candidate 4: if path includes ".../audio/...", keep subpath after audio/.
        parts = normalized_path.split("/")
        if "audio" in parts:
            audio_idx = len(parts) - 1 - parts[::-1].index("audio")
            rel_after_audio = "/".join(parts[audio_idx + 1 :]).strip()
            if rel_after_audio:
                path_candidates.append(os.path.join(audio_dir, rel_after_audio))

        # Deduplicate while preserving order.
        deduped_candidates = []
        seen = set()
        for candidate in path_candidates:
            key = candidate.strip()
            if key and key not in seen:
                seen.add(key)
                deduped_candidates.append(key)

        audio_path = None
        for candidate in deduped_candidates:
            if os.path.exists(candidate):
                audio_path = candidate
                break

        if audio_path is None:
            skipped_missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(raw_path)
            continue  # Skip missing files

        lang_code = str(row["lang"])
        lang_name = get_language_name(lang_code)

        # Skip languages not in our target set
        if lang_name not in LANG_PROMPTS:
            continue

        samples.append(
            {
                "audio": audio_path,
                "text": str(row["text"]),
                "lang": lang_name,
            }
        )

    print(
        f"Loaded {len(samples)} samples from {csv_path}. "
        f"Skipped missing files: {skipped_missing}"
    )
    if missing_examples:
        print("Missing path examples (first 5):")
        for p in missing_examples:
            print(f"  - {p}")

    return samples


# =========================
# LOGGING INFRASTRUCTURE
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_CSV_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")
VAL_CSV_PATH = os.path.join(OUTPUT_DIR, "validation_log.csv")
LANG_CSV_PATH = os.path.join(OUTPUT_DIR, "per_language_loss.csv")
EPOCH_SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "epoch_summary.csv")


# Initialize CSV files
def init_csv_files(all_langs):
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

    with open(LANG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["global_step", "epoch"]
            + [f"loss_{lang}" for lang in all_langs]
            + [f"count_{lang}" for lang in all_langs]
        )

    with open(EPOCH_SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_train_loss", "num_batches", "elapsed_sec"])


# In-memory tracking for plotting
train_steps_log = []
train_avg_log = []
val_log = []
per_lang_log = []
epoch_summary_log = []

# Per-language loss accumulator
lang_loss_accum = defaultdict(float)
lang_loss_count = defaultdict(int)

training_start_time = time.time()


def get_gpu_mem():
    """Return (allocated_gb, reserved_gb)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return alloc, reserved
    return 0.0, 0.0


def log_training_step(
    global_step, epoch, batch_idx, batch_loss, running_avg, lr, samples_seen
):
    """Log training step to CSV."""
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


def log_per_language_loss(global_step, epoch, all_langs):
    """Write per-language loss averages to CSV and reset accumulators."""
    if not lang_loss_count:
        return

    row = [global_step, epoch]
    lang_avgs = {}

    for lang in all_langs:
        if lang_loss_count[lang] > 0:
            avg = lang_loss_accum[lang] / lang_loss_count[lang]
            lang_avgs[lang] = avg
            row.append(f"{avg:.6f}")
        else:
            row.append("")

    for lang in all_langs:
        row.append(lang_loss_count.get(lang, 0))

    with open(LANG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    per_lang_log.append((global_step, dict(lang_avgs)))

    # Print summary
    print("\n  📊 Per-language loss breakdown:")
    for lang in all_langs:
        if lang_loss_count[lang] > 0:
            avg = lang_loss_accum[lang] / lang_loss_count[lang]
            print(f"    {lang}: {avg:.4f} (n={lang_loss_count[lang]})")

    # Reset accumulators
    lang_loss_accum.clear()
    lang_loss_count.clear()


def log_epoch_summary(epoch, avg_train_loss, num_batches):
    """Write epoch-level train summary."""
    elapsed = time.time() - training_start_time
    with open(EPOCH_SUMMARY_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{avg_train_loss:.6f}", num_batches, f"{elapsed:.1f}"])
    epoch_summary_log.append((epoch, avg_train_loss, num_batches))


# =========================
# PLOTTING FUNCTION
# =========================


def save_training_plots(suffix=""):
    """Generate and save training curve plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: Training loss curve
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        if train_steps_log:
            steps = [x[0] for x in train_steps_log]
            losses = [x[1] for x in train_steps_log]
            axes[0].plot(
                steps,
                losses,
                alpha=0.3,
                color="blue",
                linewidth=0.5,
                label="Batch Loss",
            )

        if train_avg_log:
            avg_steps = [x[0] for x in train_avg_log]
            avg_losses = [x[1] for x in train_avg_log]
            axes[0].plot(
                avg_steps, avg_losses, color="red", linewidth=1.5, label="Running Avg"
            )

        axes[0].set_xlabel("Global Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Validation loss
        if val_log:
            val_steps = [x[0] for x in val_log]
            val_losses = [x[1] for x in val_log]
            axes[1].plot(
                val_steps,
                val_losses,
                color="green",
                marker="o",
                linewidth=2,
                markersize=5,
                label="Val Loss",
            )

            if train_avg_log:
                axes[1].plot(
                    avg_steps,
                    avg_losses,
                    color="red",
                    linewidth=1,
                    alpha=0.5,
                    label="Train Running Avg",
                )

            axes[1].set_xlabel("Global Step")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Train vs Validation Loss")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No validation data yet",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_title("Validation Loss")

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"training_curves{suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📈 Training curves saved to: {plot_path}")

        # Plot 3: Per-language loss trends
        if per_lang_log:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

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
# VALIDATION FUNCTION
# =========================


def run_validation(model, processor, val_dataset, global_step, epoch, all_langs):
    """Run validation and return average loss."""
    model.eval()
    val_loss_total = 0.0
    val_steps = 0
    val_lang_loss = defaultdict(float)
    val_lang_count = defaultdict(int)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=indic_collate_fn,
        shuffle=False,
        drop_last=False,
    )

    with torch.no_grad():
        for inputs in tqdm(val_dataloader, desc="Validation", leave=False):
            try:
                inputs = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                labels = inputs.pop("labels")
                audio_paths = inputs.pop("audio_paths", [])
                langs = inputs.pop("langs", [])

                # Forward pass
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs, labels=labels)

                loss = outputs.loss.item()
                val_loss_total += loss
                val_steps += 1

                for lang in langs:
                    val_lang_loss[lang] += loss
                    val_lang_count[lang] += 1

            except Exception as e:
                print(f"  [VAL] Error: {e}")
                continue

    model.train()
    gc.collect()
    torch.cuda.empty_cache()

    if val_steps == 0:
        return float("nan")

    avg_val_loss = val_loss_total / val_steps

    # Per-language val loss
    lang_breakdown = {}
    for lang in sorted(val_lang_loss.keys()):
        if val_lang_count[lang] > 0:
            lang_breakdown[lang] = val_lang_loss[lang] / val_lang_count[lang]

    # Log to CSV
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

    # Print
    print(f"\n  🔍 VALIDATION @ step {global_step}:")
    print(f"     Val Loss: {avg_val_loss:.4f}")
    for lang, loss_val in sorted(lang_breakdown.items()):
        print(f"     {lang}: {loss_val:.4f}")

    return avg_val_loss


class IndicVoiceTrainer(Trainer):
    """Custom trainer with per-batch logging."""

    def __init__(
        self, *args, all_langs=None, processor=None, val_dataset=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.global_step = 0
        self.running_loss_window = []
        self.RUNNING_AVG_WINDOW = 50
        self.all_langs = all_langs or []
        self.processor = processor
        self.val_dataset = val_dataset
        self.samples_processed = 0
        self.progress_bar = None
        self.total_samples_target = 0
        self._current_epoch = None
        self._epoch_batch_idx = 0
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def _init_progress_bar_if_needed(self):
        if self.progress_bar is not None:
            return
        num_epochs = max(1, int(np.ceil(float(self.args.num_train_epochs))))
        self.total_samples_target = len(self.train_dataset) * num_epochs
        self.progress_bar = tqdm(
            total=self.total_samples_target,
            desc=f"Training Progress ({TRAIN_PROGRESS_UNIT})",
            unit="sample",
            leave=True,
        )

    def _get_epoch_idx(self):
        epoch_float = self.state.epoch if self.state.epoch is not None else 0.0
        return int(epoch_float) + 1

    def _flush_epoch_summary_if_needed(self, current_epoch):
        if self._current_epoch is None:
            self._current_epoch = current_epoch
            return

        if current_epoch == self._current_epoch:
            return

        self._flush_current_epoch_summary(
            run_epoch_validation=RUN_VALIDATION_EACH_EPOCH
        )
        self._current_epoch = current_epoch
        self._epoch_batch_idx = 0

    def _flush_current_epoch_summary(self, run_epoch_validation=False):
        if self._current_epoch is None or self._epoch_loss_count == 0:
            return

        avg_epoch_loss = self._epoch_loss_sum / self._epoch_loss_count
        print(
            f"\n✅ Epoch {self._current_epoch} Average Train Loss: {avg_epoch_loss:.4f} "
            f"(from {self._epoch_loss_count} batches)"
        )
        log_epoch_summary(self._current_epoch, avg_epoch_loss, self._epoch_loss_count)

        if (
            run_epoch_validation
            and self.val_dataset is not None
            and self.processor is not None
        ):
            print(f"\n📊 End-of-epoch validation (epoch {self._current_epoch})...")
            run_validation(
                self.model,
                self.processor,
                self.val_dataset,
                self.global_step,
                self._current_epoch,
                self.all_langs,
            )

        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0
        self._epoch_batch_idx = 0

    def finalize_epoch_loss_logging(self):
        """Flush remaining epoch summary and close progress bar."""
        self._flush_current_epoch_summary(
            run_epoch_validation=RUN_VALIDATION_EACH_EPOCH
        )
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step for custom logging."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        self._init_progress_bar_if_needed()

        # Get audio paths and languages for logging
        audio_paths = inputs.pop("audio_paths", [])
        langs = inputs.pop("langs", [])
        current_epoch = self._get_epoch_idx()
        self._flush_epoch_summary_if_needed(current_epoch)

        with self.compute_loss_context_manager():
            try:
                loss = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch
                )
            except TypeError:
                # Backward compatibility with older Trainer.compute_loss signatures.
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        batch_loss = loss.item()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Use accelerator for backward pass (handles bf16/fp16 automatically)
        self.accelerator.backward(loss)

        # Log batch info
        self.running_loss_window.append(batch_loss)
        if len(self.running_loss_window) > self.RUNNING_AVG_WINDOW:
            self.running_loss_window.pop(0)
        running_avg = sum(self.running_loss_window) / len(self.running_loss_window)
        self._epoch_loss_sum += batch_loss
        self._epoch_loss_count += 1

        # Accumulate per-language loss
        for lang in langs:
            lang_loss_accum[lang] += batch_loss
            lang_loss_count[lang] += 1

        # Print batch info
        batch_idx_for_logs = self._epoch_batch_idx + 1
        print(f"\n------------------------------------")
        print(f"Batch {batch_idx_for_logs} | Global Step {self.global_step}")
        print("Files:")
        for i, (path, lang) in enumerate(zip(audio_paths, langs)):
            print(f"  [{lang}] {os.path.basename(path)}")
        print(f"Batch Loss: {batch_loss:.4f} | Running Avg: {running_avg:.4f}")

        alloc, reserved = get_gpu_mem()
        print(f"GPU Memory: {alloc:.1f}GB alloc / {reserved:.1f}GB reserved")

        # Log to CSV
        current_lr = self.optimizer.param_groups[0]["lr"]
        log_training_step(
            self.global_step,
            self.state.epoch,
            batch_idx_for_logs,
            batch_loss,
            running_avg,
            current_lr,
            self.samples_processed,
        )

        self._epoch_batch_idx += 1
        self.global_step += 1
        self.samples_processed += len(audio_paths)
        if self.progress_bar is not None:
            self.progress_bar.update(len(audio_paths))
            self.progress_bar.set_postfix(
                {
                    "processed": f"{self.samples_processed}/{self.total_samples_target}",
                    "batch_loss": f"{batch_loss:.4f}",
                }
            )

        # Per-language breakdown
        if self.global_step % LOG_PER_LANG_EVERY == 0:
            log_per_language_loss(self.global_step, self.state.epoch, self.all_langs)

        # Periodic validation during training
        if (
            self.val_dataset is not None
            and self.processor is not None
            and RUN_VALIDATION_EVERY_N_STEPS > 0
            and self.global_step % RUN_VALIDATION_EVERY_N_STEPS == 0
        ):
            print(f"\n📊 Periodic validation @ step {self.global_step}...")
            run_validation(
                model,
                self.processor,
                self.val_dataset,
                self.global_step,
                current_epoch,
                self.all_langs,
            )

        # Save plots periodically
        if self.global_step % PLOT_EVERY_N_STEPS == 0:
            save_training_plots(suffix=f"_step{self.global_step}")

        return loss.detach()


# =========================
# MAIN TRAINING FUNCTION
# =========================


def main():
    print("=" * 60)
    print("Phi-4-Multimodal-Instruct LoRA Fine-tuning")
    print("Indic Languages: Assamese, Bengali, Hindi, Odia")
    print("=" * 60)

    # Load samples
    print("\n📦 Loading training samples...")
    samples = load_samples(DATA_DIR)
    random.shuffle(samples)
    print(f"Total training samples: {len(samples)}")

    print("\n📦 Loading validation samples...")
    val_samples = load_samples(VAL_DIR)
    random.shuffle(val_samples)
    print(f"Total validation samples: {len(val_samples)}")

    # Get all languages
    all_langs = sorted(set(s["lang"] for s in samples))
    print(f"Languages: {all_langs}")

    # Initialize CSV files
    init_csv_files(all_langs)

    # Load processor
    print("\n🔧 Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"Processor loaded successfully")

    # Load model using the official full-precision/bf16 fine-tuning path
    print("\n🔧 Loading model (official bf16 path)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).to("cuda")
    print("Model loaded successfully")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Set LoRA adapter for speech
    print("\n🔧 Setting up Speech LoRA adapter...")
    try:
        model.set_lora_adapter("speech")
    except Exception as exc:
        raise RuntimeError(
            "Failed to set LoRA adapter 'speech'. Ensure the loaded model exposes "
            "set_lora_adapter and supports the speech adapter."
        ) from exc

    # Print trainable parameters
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable parameters: {trainable_params:,} / {all_params:,} "
        f"({100 * trainable_params / all_params:.4f}%)"
    )

    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = IndicVoiceDataset(samples, processor, training=True)
    val_dataset = IndicVoiceDataset(val_samples, processor, training=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,  # No CPU multithreading
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
    )

    # Create trainer
    trainer = IndicVoiceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=indic_collate_fn,
        all_langs=all_langs,
        processor=processor,
        val_dataset=val_dataset,
    )

    # Progress bar setup
    total_steps = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS)
    print(f"\n🚀 Starting training...")
    print(f"Total estimated steps per epoch: {total_steps}")

    # Train
    trainer.train()
    trainer.finalize_epoch_loss_logging()

    # Flush any remaining per-language accumulators
    if lang_loss_count:
        log_per_language_loss(trainer.global_step, EPOCHS, all_langs)

    # Run final validation
    print("\n📊 Final validation...")
    run_validation(
        model, processor, val_dataset, trainer.global_step, EPOCHS, all_langs
    )

    # Save model
    print(f"\n💾 Saving model to: {OUTPUT_DIR}")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)

    # Final plots
    save_training_plots(suffix="_final")

    # Training summary
    elapsed_total = time.time() - training_start_time
    print(f"\n📊 TRAINING SUMMARY")
    print("=" * 50)
    print(f"Total time: {elapsed_total / 3600:.2f} hours ({elapsed_total:.0f} seconds)")
    print(f"Total steps: {trainer.global_step}")
    print(f"Samples trained: {len(samples)}")
    print(f"Validation samples: {len(val_samples)}")

    if val_log:
        first_val = val_log[0][1]
        last_val = val_log[-1][1]
        print(f"First val loss: {first_val:.4f}")
        print(f"Last val loss: {last_val:.4f}")
        print(f"Val loss change: {last_val - first_val:+.4f}")
        if last_val > first_val * 1.1:
            print(f"⚠️  WARNING: Validation loss increased - possible overfitting!")
        else:
            print(f"✅ No significant overfitting detected.")

    print(f"\nOutput files:")
    print(f"  📄 {LOG_CSV_PATH}")
    print(f"  📄 {VAL_CSV_PATH}")
    print(f"  📄 {LANG_CSV_PATH}")
    print(f"  📄 {EPOCH_SUMMARY_CSV_PATH}")
    print(f"  📈 {os.path.join(OUTPUT_DIR, 'training_curves_final.png')}")
    print(f"  📈 {os.path.join(OUTPUT_DIR, 'per_language_loss_final.png')}")
    print(f"\n✅ Final model saved to: {OUTPUT_DIR}")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
