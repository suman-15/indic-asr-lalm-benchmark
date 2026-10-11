"""
MMS-1b-all LoRA Inference Script for Indic Languages (Assamese, Bengali, Hindi)
================================================================================

This script performs inference on test data using the fine-tuned MMS model with
LoRA adapter. It evaluates test samples for 3 languages and computes comprehensive
ASR metrics with 95% confidence intervals for research publication.

Key Features:
- Loads fine-tuned LoRA adapter with unified multilingual vocabulary
- Batch GPU processing (batch_size=2, randomly shuffled)
- Complete audio clips (no chunking)
- Per-sample metrics: CER, WER, char P/R/F1, word P/R/F1
- Per-language and overall aggregate metrics
- 95% Confidence Intervals using bootstrap resampling
- Visualization plots (box plot, bar chart with CI, scatter, forest plot)
- Results saved to CSV and JSON
- tqdm progress bar

Matches training setup from mms_1b_all_lora_indic_train.py:
- Wav2Vec2ForCTC with CTC decoding (argmax)
- Unified multilingual vocabulary with | as word delimiter
- Wav2Vec2Processor (feature_extractor + tokenizer)
- Full precision model (no quantization) with FP16 autocast

Usage:
    python mms_1b_lora_indic_inference.py

Requires:
    pip install transformers peft soundfile scipy jiwer tqdm pandas numpy matplotlib
"""

import os
import csv
import time
import json
import random
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import jiwer

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from peft import PeftModel

# =========================
# CONFIGURATION
# =========================

# Model paths
BASE_MODEL_NAME = "facebook/mms-1b-all"

# Dataset directory — UPDATE THIS to your actual path
DATASET_DIR = "/path/to/data/IndicVoices_finetune_15k"  # Linux
# DATASET_DIR = r"/path/to/data/indicvocies_finetune_15k_mini"        # Windows

TEST_DIR = os.path.join(DATASET_DIR, "test")

# LoRA adapter path — uses the best checkpoint saved during training
FINETUNED_MODEL_PATH = os.path.join(DATASET_DIR, "mms_lora_indic", "best")
if not os.path.exists(FINETUNED_MODEL_PATH):
    FINETUNED_MODEL_PATH = os.path.join(DATASET_DIR, "mms_lora_indic")

# Output directory
OUTPUT_DIR = os.path.join(DATASET_DIR, "mms_inference_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Audio configuration
TARGET_SR = 16000  # MMS expects 16kHz audio
MAX_AUDIO_SECONDS = 30

# Inference configuration
BATCH_SIZE = 2  # Process 2 random audio files at a time

# Bootstrap configuration for confidence intervals
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95

# Random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Disable CPU multithreading (as requested)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Speed over determinism

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LANGUAGE CONFIGURATION
# =========================

LANG_CODE_TO_NAME = {
    "as": "Assamese",
    "bn": "Bengali",
    "hi": "Hindi",
}

ALLOWED_LANG_CODES = set(LANG_CODE_TO_NAME.keys())


def get_language_name(lang_code: str) -> str:
    """Convert language code to language name."""
    code = str(lang_code).lower().strip()
    return LANG_CODE_TO_NAME.get(code, code.capitalize())


def resolve_audio_path(raw_path: str, audio_dir: str, split_dir: str) -> Optional[str]:
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
# AUDIO LOADING
# =========================


def load_audio(path: str) -> np.ndarray:
    """Load audio file, convert to mono, resample to 16kHz. Use full clip."""
    wav, sr = sf.read(path, dtype="float32")

    # Convert stereo to mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // g, sr // g).astype(np.float32)

    # Truncate to max duration
    max_len = int(TARGET_SR * MAX_AUDIO_SECONDS)
    if len(wav) > max_len:
        wav = wav[:max_len]

    return wav


# =========================
# METRICS COMPUTATION
# =========================


def compute_asr_metrics(reference: str, hypothesis: str) -> Dict:
    """
    Compute comprehensive ASR metrics between reference and hypothesis strings.

    Returns dict with:
      - cer, char_precision, char_recall, char_f1
      - wer, word_precision, word_recall, word_f1
    """
    ref = reference.strip()
    hyp = hypothesis.strip()

    if not ref and not hyp:
        return {
            "cer": 0.0,
            "char_precision": 1.0,
            "char_recall": 1.0,
            "char_f1": 1.0,
            "wer": 0.0,
            "word_precision": 1.0,
            "word_recall": 1.0,
            "word_f1": 1.0,
        }
    if not ref:
        return {
            "cer": 1.0,
            "char_precision": 0.0,
            "char_recall": 0.0,
            "char_f1": 0.0,
            "wer": 1.0,
            "word_precision": 0.0,
            "word_recall": 0.0,
            "word_f1": 0.0,
        }
    if not hyp:
        return {
            "cer": 1.0,
            "char_precision": 0.0,
            "char_recall": 0.0,
            "char_f1": 0.0,
            "wer": 1.0,
            "word_precision": 0.0,
            "word_recall": 0.0,
            "word_f1": 0.0,
        }

    # --- Character-level metrics ---
    char_out = jiwer.process_characters(ref, hyp)
    cer_val = char_out.cer

    c_hits = char_out.hits
    c_subs = char_out.substitutions
    c_ins = char_out.insertions
    c_dels = char_out.deletions

    c_pred_total = c_hits + c_subs + c_ins
    c_ref_total = c_hits + c_subs + c_dels

    char_precision = c_hits / c_pred_total if c_pred_total > 0 else 0.0
    char_recall = c_hits / c_ref_total if c_ref_total > 0 else 0.0
    char_f1 = (
        2 * char_precision * char_recall / (char_precision + char_recall)
        if (char_precision + char_recall) > 0
        else 0.0
    )

    # --- Word-level metrics ---
    word_out = jiwer.process_words(ref, hyp)
    wer_val = word_out.wer

    w_hits = word_out.hits
    w_subs = word_out.substitutions
    w_ins = word_out.insertions
    w_dels = word_out.deletions

    w_pred_total = w_hits + w_subs + w_ins
    w_ref_total = w_hits + w_subs + w_dels

    word_precision = w_hits / w_pred_total if w_pred_total > 0 else 0.0
    word_recall = w_hits / w_ref_total if w_ref_total > 0 else 0.0
    word_f1 = (
        2 * word_precision * word_recall / (word_precision + word_recall)
        if (word_precision + word_recall) > 0
        else 0.0
    )

    return {
        "cer": cer_val,
        "char_precision": char_precision,
        "char_recall": char_recall,
        "char_f1": char_f1,
        "wer": wer_val,
        "word_precision": word_precision,
        "word_recall": word_recall,
        "word_f1": word_f1,
    }


def compute_bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a list of values.

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    values = np.array(values)
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(n, size=n, replace=True)
        bootstrap_sample = values[sample_indices]
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    mean = np.mean(values)

    return mean, lower, upper


# =========================
# CORPUS-LEVEL AGGREGATE METRICS
# =========================


def compute_corpus_metrics(results_list: List[Dict]) -> Dict:
    """
    Compute corpus-level aggregate metrics using jiwer on concatenated refs/hyps.
    Also computes sample-averaged metrics.
    """
    if not results_list:
        return {}

    all_refs = [r["reference"] for r in results_list]
    all_hyps = [r["prediction"] for r in results_list]

    # Filter out empty pairs
    pairs = [
        (ref, hyp) for ref, hyp in zip(all_refs, all_hyps) if ref.strip() or hyp.strip()
    ]
    if not pairs:
        return {
            "count": len(results_list),
            "corpus_cer": 0.0,
            "corpus_wer": 0.0,
            "corpus_char_precision": 1.0,
            "corpus_char_recall": 1.0,
            "corpus_char_f1": 1.0,
            "corpus_word_precision": 1.0,
            "corpus_word_recall": 1.0,
            "corpus_word_f1": 1.0,
            "avg_cer": 0.0,
            "avg_wer": 0.0,
            "avg_char_precision": 1.0,
            "avg_char_recall": 1.0,
            "avg_char_f1": 1.0,
            "avg_word_precision": 1.0,
            "avg_word_recall": 1.0,
            "avg_word_f1": 1.0,
        }

    refs, hyps = zip(*pairs)

    # Corpus-level CER/WER
    try:
        corpus_cer = jiwer.cer(list(refs), list(hyps))
    except Exception:
        corpus_cer = float("nan")
    try:
        corpus_wer = jiwer.wer(list(refs), list(hyps))
    except Exception:
        corpus_wer = float("nan")

    # Corpus-level char P/R/F1
    try:
        char_out = jiwer.process_characters(list(refs), list(hyps))
        c_hits, c_subs, c_ins, c_dels = (
            char_out.hits,
            char_out.substitutions,
            char_out.insertions,
            char_out.deletions,
        )
        c_pred = c_hits + c_subs + c_ins
        c_ref = c_hits + c_subs + c_dels
        corpus_char_prec = c_hits / c_pred if c_pred > 0 else 0.0
        corpus_char_rec = c_hits / c_ref if c_ref > 0 else 0.0
        corpus_char_f1 = (
            2
            * corpus_char_prec
            * corpus_char_rec
            / (corpus_char_prec + corpus_char_rec)
            if (corpus_char_prec + corpus_char_rec) > 0
            else 0.0
        )
    except Exception:
        corpus_char_prec = corpus_char_rec = corpus_char_f1 = float("nan")

    # Corpus-level word P/R/F1
    try:
        word_out = jiwer.process_words(list(refs), list(hyps))
        w_hits, w_subs, w_ins, w_dels = (
            word_out.hits,
            word_out.substitutions,
            word_out.insertions,
            word_out.deletions,
        )
        w_pred = w_hits + w_subs + w_ins
        w_ref = w_hits + w_subs + w_dels
        corpus_word_prec = w_hits / w_pred if w_pred > 0 else 0.0
        corpus_word_rec = w_hits / w_ref if w_ref > 0 else 0.0
        corpus_word_f1 = (
            2
            * corpus_word_prec
            * corpus_word_rec
            / (corpus_word_prec + corpus_word_rec)
            if (corpus_word_prec + corpus_word_rec) > 0
            else 0.0
        )
    except Exception:
        corpus_word_prec = corpus_word_rec = corpus_word_f1 = float("nan")

    # Sample-averaged metrics
    n = len(results_list)
    avg_cer = sum(r["cer"] for r in results_list) / n
    avg_wer = sum(r["wer"] for r in results_list) / n
    avg_char_prec = sum(r["char_precision"] for r in results_list) / n
    avg_char_rec = sum(r["char_recall"] for r in results_list) / n
    avg_char_f1 = sum(r["char_f1"] for r in results_list) / n
    avg_word_prec = sum(r["word_precision"] for r in results_list) / n
    avg_word_rec = sum(r["word_recall"] for r in results_list) / n
    avg_word_f1 = sum(r["word_f1"] for r in results_list) / n

    return {
        "count": n,
        "corpus_cer": corpus_cer,
        "corpus_char_precision": corpus_char_prec,
        "corpus_char_recall": corpus_char_rec,
        "corpus_char_f1": corpus_char_f1,
        "corpus_wer": corpus_wer,
        "corpus_word_precision": corpus_word_prec,
        "corpus_word_recall": corpus_word_rec,
        "corpus_word_f1": corpus_word_f1,
        "avg_cer": avg_cer,
        "avg_char_precision": avg_char_prec,
        "avg_char_recall": avg_char_rec,
        "avg_char_f1": avg_char_f1,
        "avg_wer": avg_wer,
        "avg_word_precision": avg_word_prec,
        "avg_word_recall": avg_word_rec,
        "avg_word_f1": avg_word_f1,
    }


# =========================
# VISUALIZATION PLOTS
# =========================


def generate_plots(results: List[Dict], ci_metrics: Dict, output_dir: str):
    """Generate visualization plots for research publication."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Collect per-language WER lists
        lang_wers = defaultdict(list)
        for r in results:
            lang_wers[r["lang"]].append(r["wer"])

        langs = sorted(lang_wers.keys())
        wer_data = [lang_wers[lang] for lang in langs]
        colors = {"Assamese": "blue", "Bengali": "green", "Hindi": "red"}

        # --- Plot 1: WER box plot + WER/CER bar chart with CI ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot
        axes[0].boxplot(wer_data, labels=langs)
        axes[0].set_xlabel("Language")
        axes[0].set_ylabel("Word Error Rate (WER)")
        axes[0].set_title("WER Distribution by Language")
        axes[0].grid(True, alpha=0.3)

        # Bar plot with CI
        x = range(len(langs))
        avg_wers = [ci_metrics["per_language"][lang]["wer"] for lang in langs]
        wer_err_lo = [
            ci_metrics["per_language"][lang]["wer"]
            - ci_metrics["per_language"][lang]["wer_ci_lower"]
            for lang in langs
        ]
        wer_err_hi = [
            ci_metrics["per_language"][lang]["wer_ci_upper"]
            - ci_metrics["per_language"][lang]["wer"]
            for lang in langs
        ]
        avg_cers = [ci_metrics["per_language"][lang]["cer"] for lang in langs]
        cer_err_lo = [
            ci_metrics["per_language"][lang]["cer"]
            - ci_metrics["per_language"][lang]["cer_ci_lower"]
            for lang in langs
        ]
        cer_err_hi = [
            ci_metrics["per_language"][lang]["cer_ci_upper"]
            - ci_metrics["per_language"][lang]["cer"]
            for lang in langs
        ]

        width = 0.35
        axes[1].bar(
            [i - width / 2 for i in x],
            avg_wers,
            width,
            yerr=[wer_err_lo, wer_err_hi],
            label="WER",
            color="steelblue",
            capsize=5,
        )
        axes[1].bar(
            [i + width / 2 for i in x],
            avg_cers,
            width,
            yerr=[cer_err_lo, cer_err_hi],
            label="CER",
            color="coral",
            capsize=5,
        )
        axes[1].set_xlabel("Language")
        axes[1].set_ylabel("Error Rate")
        axes[1].set_title("Average WER and CER by Language (with 95% CI)")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(langs)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "inference_metrics.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Plot saved to: {plot_path}")

        # --- Plot 2: WER vs CER scatter ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for lang in langs:
            lr = [r for r in results if r["lang"] == lang]
            ax.scatter(
                [r["wer"] for r in lr],
                [r["cer"] for r in lr],
                alpha=0.5,
                label=lang,
                c=colors.get(lang, "gray"),
            )
        ax.set_xlabel("Word Error Rate (WER)")
        ax.set_ylabel("Character Error Rate (CER)")
        ax.set_title("WER vs CER by Sample")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        scatter_path = os.path.join(output_dir, "wer_vs_cer_scatter.png")
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Scatter plot saved to: {scatter_path}")

        # --- Plot 3: Confidence interval forest plot ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        y_pos = range(len(langs) + 1)

        ov = ci_metrics["overall"]
        ax.errorbar(
            ov["wer"],
            0,
            xerr=[[ov["wer"] - ov["wer_ci_lower"]], [ov["wer_ci_upper"] - ov["wer"]]],
            fmt="o",
            color="black",
            capsize=5,
            markersize=8,
            label="OVERALL",
        )

        for i, lang in enumerate(langs):
            lm = ci_metrics["per_language"][lang]
            ax.errorbar(
                lm["wer"],
                i + 1,
                xerr=[
                    [lm["wer"] - lm["wer_ci_lower"]],
                    [lm["wer_ci_upper"] - lm["wer"]],
                ],
                fmt="o",
                color=colors.get(lang, "gray"),
                capsize=5,
                markersize=8,
            )

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(["OVERALL"] + langs)
        ax.set_xlabel("Word Error Rate (WER)")
        ax.set_title("WER with 95% Confidence Intervals")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        ci_path = os.path.join(output_dir, "confidence_intervals.png")
        plt.savefig(ci_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Confidence interval plot saved to: {ci_path}")

    except ImportError:
        print("  ⚠️ matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"  ⚠️ Plot generation failed: {e}")


# =========================
# LOAD TEST METADATA
# =========================


def load_test_samples(test_dir: str) -> List[Dict]:
    """Load test samples from test.csv and resolve audio paths."""
    csv_path = os.path.join(test_dir, "test.csv")
    audio_dir = os.path.join(test_dir, "audio")

    print(f"Loading test CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

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
            split_dir=test_dir,
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
                "filename": os.path.basename(resolved_path),
            }
        )

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} samples (audio not found or empty text)")
    if skipped_unsupported_lang > 0:
        print(f"  ⚠️  Skipped {skipped_unsupported_lang} samples (unsupported language)")
    print(f"  ✅ Loaded {len(samples)} test samples")
    return samples


# =========================
# MAIN FUNCTION
# =========================


def main():
    print("=" * 60)
    print("MMS-1b-all LoRA Inference for Indic Languages")
    print("=" * 60)
    print(f"Base Model: {BASE_MODEL_NAME}")
    print(f"Fine-tuned Model: {FINETUNED_MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Bootstrap iterations for CI: {BOOTSTRAP_ITERATIONS}")
    print(f"Confidence level: {CONFIDENCE_LEVEL * 100}%")
    print("=" * 60)

    # Check paths
    if not os.path.exists(FINETUNED_MODEL_PATH):
        raise FileNotFoundError(
            f"Fine-tuned model not found at: {FINETUNED_MODEL_PATH}"
        )

    # =========================
    # LOAD MODEL
    # =========================
    print("\n🔧 Loading model...")

    # Load processor from saved adapter dir (has unified vocabulary)
    print(f"Loading processor from: {FINETUNED_MODEL_PATH}")
    processor = Wav2Vec2Processor.from_pretrained(FINETUNED_MODEL_PATH)

    # Get vocab size from the loaded processor
    vocab_size = len(processor.tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # Load base model with matching vocab size
    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL_NAME,
        vocab_size=vocab_size,
        ignore_mismatched_sizes=True,  # Required: vocab size differs from pretrained
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        torch_dtype=torch.float32,  # Match training: full precision model
    )

    # Move to device
    base_model = base_model.to(DEVICE)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {FINETUNED_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model loaded: {total_params:,} total params, {trainable_params:,} trainable"
    )

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {alloc:.1f}GB alloc / {reserved:.1f}GB reserved")

    print("✅ Model loaded successfully")

    # =========================
    # LOAD TEST DATA
    # =========================
    print("\n📂 Loading test data...")
    test_samples = load_test_samples(TEST_DIR)

    if not test_samples:
        print("No test samples found. Exiting.")
        return

    # Shuffle randomly
    random.shuffle(test_samples)

    # Print language distribution
    lang_counts = defaultdict(int)
    for s in test_samples:
        lang_counts[s["lang"]] += 1
    print("\nTest language distribution:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")

    # =========================
    # RESULTS CSV SETUP
    # =========================
    results_csv_path = os.path.join(OUTPUT_DIR, "inference_results.csv")
    csv_headers = [
        "sample_idx",
        "filename",
        "language",
        "reference_text",
        "predicted_text",
        "CER",
        "char_precision",
        "char_recall",
        "char_f1",
        "WER",
        "word_precision",
        "word_recall",
        "word_f1",
    ]
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_headers)

    # =========================
    # INFERENCE LOOP
    # =========================
    print(f"\n{'=' * 60}")
    print("STARTING INFERENCE")
    print(f"{'=' * 60}")
    print(f"Total test samples: {len(test_samples)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Decoding: CTC greedy (argmax)")
    print(f"LoRA adapter: {FINETUNED_MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    all_results = []
    per_lang_results = defaultdict(list)

    start_time = time.time()

    progress = tqdm(
        total=len(test_samples),
        desc="Inference",
        unit="samples",
        dynamic_ncols=True,
    )

    batch_num = 0

    for i in range(0, len(test_samples), BATCH_SIZE):
        batch = test_samples[i : i + BATCH_SIZE]
        batch_num += 1

        # Load audio for all samples in batch
        audio_arrays = []
        for s in batch:
            wav = load_audio(s["audio"])
            audio_arrays.append(wav)

        # Process audio through feature extractor (normalizes + pads)
        inputs = processor.feature_extractor(
            audio_arrays,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        input_values = inputs.input_values.to(DEVICE)
        attention_mask = inputs.attention_mask.to(DEVICE)

        # Forward pass with FP16 autocast for speed
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                ).logits

        # CTC greedy decoding: argmax → batch_decode
        pred_ids = torch.argmax(logits, dim=-1)
        predictions = processor.batch_decode(pred_ids)

        # Process each sample in the batch
        print(f"\n{'─' * 60}")
        print(
            f"Batch {batch_num} | Samples {i + 1}-{i + len(batch)} of {len(test_samples)}"
        )
        print(f"{'─' * 60}")

        for j, sample in enumerate(batch):
            pred_text = predictions[j].strip()
            ref_text = sample["text"].strip()
            lang_name = sample["lang"]
            filename = sample["filename"]

            # Compute metrics
            metrics = compute_asr_metrics(ref_text, pred_text)

            # Store result
            result = {
                "sample_idx": i + j + 1,
                "filename": filename,
                "lang": lang_name,
                "lang_code": sample["lang_code"],
                "reference": ref_text,
                "prediction": pred_text,
                **metrics,
            }
            all_results.append(result)
            per_lang_results[lang_name].append(result)

            # Write to CSV immediately (streaming)
            with open(results_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        result["sample_idx"],
                        result["filename"],
                        result["lang"],
                        result["reference"],
                        result["prediction"],
                        f"{metrics['cer']:.4f}",
                        f"{metrics['char_precision']:.4f}",
                        f"{metrics['char_recall']:.4f}",
                        f"{metrics['char_f1']:.4f}",
                        f"{metrics['wer']:.4f}",
                        f"{metrics['word_precision']:.4f}",
                        f"{metrics['word_recall']:.4f}",
                        f"{metrics['word_f1']:.4f}",
                    ]
                )

            # Print per-sample details
            print(f"\n  📄 File: {filename}")
            print(f"     Language:   {lang_name}")
            print(
                f"     Reference:  {ref_text[:120]}{'...' if len(ref_text) > 120 else ''}"
            )
            print(
                f"     Predicted:  {pred_text[:120]}{'...' if len(pred_text) > 120 else ''}"
            )
            print(f"     ── Character-level ──")
            print(
                f"        CER: {metrics['cer']:.4f} | "
                f"Precision: {metrics['char_precision']:.4f} | "
                f"Recall: {metrics['char_recall']:.4f} | "
                f"F1: {metrics['char_f1']:.4f}"
            )
            print(f"     ── Word-level ──")
            print(
                f"        WER: {metrics['wer']:.4f} | "
                f"Precision: {metrics['word_precision']:.4f} | "
                f"Recall: {metrics['word_recall']:.4f} | "
                f"F1: {metrics['word_f1']:.4f}"
            )

        progress.update(len(batch))

    progress.close()
    elapsed = time.time() - start_time

    # =========================
    # OVERALL SUMMARY
    # =========================
    overall_agg = compute_corpus_metrics(all_results)

    print(f"\n{'=' * 70}")
    print(f"OVERALL RESULTS ({overall_agg['count']} samples)")
    print(f"{'=' * 70}")
    print(
        f"Time elapsed: {elapsed:.1f}s ({elapsed / max(len(all_results), 1):.2f}s per sample)"
    )
    print(f"\n  ── Corpus-level Character Metrics ──")
    print(f"     CER:       {overall_agg['corpus_cer']:.4f}")
    print(f"     Precision: {overall_agg['corpus_char_precision']:.4f}")
    print(f"     Recall:    {overall_agg['corpus_char_recall']:.4f}")
    print(f"     F1:        {overall_agg['corpus_char_f1']:.4f}")
    print(f"\n  ── Corpus-level Word Metrics ──")
    print(f"     WER:       {overall_agg['corpus_wer']:.4f}")
    print(f"     Precision: {overall_agg['corpus_word_precision']:.4f}")
    print(f"     Recall:    {overall_agg['corpus_word_recall']:.4f}")
    print(f"     F1:        {overall_agg['corpus_word_f1']:.4f}")
    print(f"\n  ── Sample-averaged Character Metrics ──")
    print(f"     CER:       {overall_agg['avg_cer']:.4f}")
    print(f"     Precision: {overall_agg['avg_char_precision']:.4f}")
    print(f"     Recall:    {overall_agg['avg_char_recall']:.4f}")
    print(f"     F1:        {overall_agg['avg_char_f1']:.4f}")
    print(f"\n  ── Sample-averaged Word Metrics ──")
    print(f"     WER:       {overall_agg['avg_wer']:.4f}")
    print(f"     Precision: {overall_agg['avg_word_precision']:.4f}")
    print(f"     Recall:    {overall_agg['avg_word_recall']:.4f}")
    print(f"     F1:        {overall_agg['avg_word_f1']:.4f}")

    # Save overall summary CSV
    overall_csv_path = os.path.join(OUTPUT_DIR, "overall_summary.csv")
    with open(overall_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in overall_agg.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])

    # =========================
    # PER-LANGUAGE SUMMARY
    # =========================
    print(f"\n{'=' * 70}")
    print("PER-LANGUAGE RESULTS")
    print(f"{'=' * 70}")

    lang_summary_rows = []
    for lang in sorted(per_lang_results.keys()):
        lang_res = per_lang_results[lang]
        lang_agg = compute_corpus_metrics(lang_res)

        print(f"\n  📌 {lang} ({lang_agg['count']} samples)")
        print(f"     ── Corpus-level Character Metrics ──")
        print(f"        CER:       {lang_agg['corpus_cer']:.4f}")
        print(f"        Precision: {lang_agg['corpus_char_precision']:.4f}")
        print(f"        Recall:    {lang_agg['corpus_char_recall']:.4f}")
        print(f"        F1:        {lang_agg['corpus_char_f1']:.4f}")
        print(f"     ── Corpus-level Word Metrics ──")
        print(f"        WER:       {lang_agg['corpus_wer']:.4f}")
        print(f"        Precision: {lang_agg['corpus_word_precision']:.4f}")
        print(f"        Recall:    {lang_agg['corpus_word_recall']:.4f}")
        print(f"        F1:        {lang_agg['corpus_word_f1']:.4f}")
        print(f"     ── Sample-averaged Character Metrics ──")
        print(f"        CER:       {lang_agg['avg_cer']:.4f}")
        print(f"        Precision: {lang_agg['avg_char_precision']:.4f}")
        print(f"        Recall:    {lang_agg['avg_char_recall']:.4f}")
        print(f"        F1:        {lang_agg['avg_char_f1']:.4f}")
        print(f"     ── Sample-averaged Word Metrics ──")
        print(f"        WER:       {lang_agg['avg_wer']:.4f}")
        print(f"        Precision: {lang_agg['avg_word_precision']:.4f}")
        print(f"        Recall:    {lang_agg['avg_word_recall']:.4f}")
        print(f"        F1:        {lang_agg['avg_word_f1']:.4f}")

        lang_summary_rows.append({"language": lang, **lang_agg})

    # Save per-language summary CSV
    per_lang_csv_path = os.path.join(OUTPUT_DIR, "per_language_summary.csv")
    with open(per_lang_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "language",
            "count",
            "corpus_cer",
            "corpus_char_precision",
            "corpus_char_recall",
            "corpus_char_f1",
            "corpus_wer",
            "corpus_word_precision",
            "corpus_word_recall",
            "corpus_word_f1",
            "avg_cer",
            "avg_char_precision",
            "avg_char_recall",
            "avg_char_f1",
            "avg_wer",
            "avg_word_precision",
            "avg_word_recall",
            "avg_word_f1",
        ]
        writer.writerow(header)
        for row in lang_summary_rows:
            writer.writerow(
                [
                    row["language"],
                    row["count"],
                    f"{row['corpus_cer']:.6f}",
                    f"{row['corpus_char_precision']:.6f}",
                    f"{row['corpus_char_recall']:.6f}",
                    f"{row['corpus_char_f1']:.6f}",
                    f"{row['corpus_wer']:.6f}",
                    f"{row['corpus_word_precision']:.6f}",
                    f"{row['corpus_word_recall']:.6f}",
                    f"{row['corpus_word_f1']:.6f}",
                    f"{row['avg_cer']:.6f}",
                    f"{row['avg_char_precision']:.6f}",
                    f"{row['avg_char_recall']:.6f}",
                    f"{row['avg_char_f1']:.6f}",
                    f"{row['avg_wer']:.6f}",
                    f"{row['avg_word_precision']:.6f}",
                    f"{row['avg_word_recall']:.6f}",
                    f"{row['avg_word_f1']:.6f}",
                ]
            )

    # =========================
    # BOOTSTRAP CI METRICS
    # =========================
    print(f"\n📊 Computing metrics with {BOOTSTRAP_ITERATIONS} bootstrap iterations...")

    ci_metrics = {"overall": {}, "per_language": {}}

    # Overall CI
    all_wers = [r["wer"] for r in all_results]
    all_cers = [r["cer"] for r in all_results]
    wer_mean, wer_lower, wer_upper = compute_bootstrap_ci(
        all_wers, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS
    )
    cer_mean, cer_lower, cer_upper = compute_bootstrap_ci(
        all_cers, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS
    )

    ci_metrics["overall"] = {
        "wer": wer_mean,
        "wer_ci_lower": wer_lower,
        "wer_ci_upper": wer_upper,
        "cer": cer_mean,
        "cer_ci_lower": cer_lower,
        "cer_ci_upper": cer_upper,
        "total_samples": len(all_results),
    }

    # Per-language CI
    for lang_name, lang_list in per_lang_results.items():
        lw = [r["wer"] for r in lang_list]
        lc = [r["cer"] for r in lang_list]
        lwm, lwl, lwu = compute_bootstrap_ci(lw, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS)
        lcm, lcl, lcu = compute_bootstrap_ci(lc, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS)
        ci_metrics["per_language"][lang_name] = {
            "wer": lwm,
            "wer_ci_lower": lwl,
            "wer_ci_upper": lwu,
            "cer": lcm,
            "cer_ci_lower": lcl,
            "cer_ci_upper": lcu,
            "total_samples": len(lang_list),
        }

    # Print CI results
    print(f"\n{'=' * 60}")
    print("RESULTS WITH 95% CONFIDENCE INTERVALS")
    print(f"{'=' * 60}")
    print(f"\n📊 Overall Metrics (95% CI):")
    print(
        f"   WER: {ci_metrics['overall']['wer']:.4f} [{ci_metrics['overall']['wer_ci_lower']:.4f}, {ci_metrics['overall']['wer_ci_upper']:.4f}]"
    )
    print(
        f"   CER: {ci_metrics['overall']['cer']:.4f} [{ci_metrics['overall']['cer_ci_lower']:.4f}, {ci_metrics['overall']['cer_ci_upper']:.4f}]"
    )
    print(f"   Total samples: {ci_metrics['overall']['total_samples']}")

    print(f"\n📊 Per-Language Metrics (95% CI):")
    for lang_name in sorted(ci_metrics["per_language"].keys()):
        lm = ci_metrics["per_language"][lang_name]
        print(f"   {lang_name}:")
        print(
            f"      WER: {lm['wer']:.4f} [{lm['wer_ci_lower']:.4f}, {lm['wer_ci_upper']:.4f}]"
        )
        print(
            f"      CER: {lm['cer']:.4f} [{lm['cer_ci_lower']:.4f}, {lm['cer_ci_upper']:.4f}]"
        )
        print(f"      Samples: {lm['total_samples']}")

    # Save CI metrics CSV
    ci_csv_path = os.path.join(OUTPUT_DIR, "metrics_summary_with_ci.csv")
    with open(ci_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "language",
                "wer",
                "wer_ci_lower",
                "wer_ci_upper",
                "cer",
                "cer_ci_lower",
                "cer_ci_upper",
                "num_samples",
            ]
        )
        w.writerow(
            [
                "OVERALL",
                f"{ci_metrics['overall']['wer']:.4f}",
                f"{ci_metrics['overall']['wer_ci_lower']:.4f}",
                f"{ci_metrics['overall']['wer_ci_upper']:.4f}",
                f"{ci_metrics['overall']['cer']:.4f}",
                f"{ci_metrics['overall']['cer_ci_lower']:.4f}",
                f"{ci_metrics['overall']['cer_ci_upper']:.4f}",
                ci_metrics["overall"]["total_samples"],
            ]
        )
        for lang_name in sorted(ci_metrics["per_language"].keys()):
            lm = ci_metrics["per_language"][lang_name]
            w.writerow(
                [
                    lang_name,
                    f"{lm['wer']:.4f}",
                    f"{lm['wer_ci_lower']:.4f}",
                    f"{lm['wer_ci_upper']:.4f}",
                    f"{lm['cer']:.4f}",
                    f"{lm['cer_ci_lower']:.4f}",
                    f"{lm['cer_ci_upper']:.4f}",
                    lm["total_samples"],
                ]
            )

    # Save CI metrics JSON
    ci_json_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(ci_json_path, "w", encoding="utf-8") as f:
        json.dump(ci_metrics, f, indent=2)

    # =========================
    # GENERATE PLOTS
    # =========================
    print(f"\n📈 Generating visualization plots...")
    generate_plots(all_results, ci_metrics, OUTPUT_DIR)

    # =========================
    # SAMPLE PREDICTIONS
    # =========================
    print(f"\n📝 Sample Predictions (first 5 per language):")
    for lang in sorted(per_lang_results.keys()):
        lang_res = per_lang_results[lang][:5]
        print(f"\n{lang}:")
        for idx, r in enumerate(lang_res):
            ref_preview = (
                r["reference"][:60] + "..."
                if len(r["reference"]) > 60
                else r["reference"]
            )
            pred_preview = (
                r["prediction"][:60] + "..."
                if len(r["prediction"]) > 60
                else r["prediction"]
            )
            print(f"   Sample {idx + 1}:")
            print(f"      Reference:  {ref_preview}")
            print(f"      Prediction: {pred_preview}")
            print(f"      WER: {r['wer']:.4f} | CER: {r['cer']:.4f}")

    # =========================
    # FINAL SUMMARY
    # =========================
    print(f"\n{'=' * 70}")
    print("OUTPUT FILES")
    print(f"{'=' * 70}")
    print(f"  📄 Per-sample results:       {results_csv_path}")
    print(f"  📄 Per-language summary:     {per_lang_csv_path}")
    print(f"  📄 Overall summary:          {overall_csv_path}")
    print(f"  📄 CI Metrics CSV:           {ci_csv_path}")
    print(f"  📄 CI Metrics JSON:          {ci_json_path}")
    print(
        f"  📈 Inference metrics plot:   {os.path.join(OUTPUT_DIR, 'inference_metrics.png')}"
    )
    print(
        f"  📈 WER vs CER scatter:       {os.path.join(OUTPUT_DIR, 'wer_vs_cer_scatter.png')}"
    )
    print(
        f"  📈 Confidence intervals:     {os.path.join(OUTPUT_DIR, 'confidence_intervals.png')}"
    )

    print(
        f"\n✅ Inference complete. {len(all_results)} samples processed in {elapsed:.1f}s."
    )
    print("Results are ready for research publication.")
    print("=" * 70)

    return all_results, ci_metrics


if __name__ == "__main__":
    main()
