"""
Whisper Large-v3 LoRA Inference for Indic Languages (Assamese, Bengali, Hindi, Odia)

Loads the locally saved LoRA adapter from training and performs ASR inference
on the test split. Computes per-sample and aggregate CER, WER, precision,
recall, F1 at both character and word level. Exports all results to CSV.

Key points:
- 4-bit NF4 quantization for memory efficiency
- Processes audio files in display-batches of 2 (randomly shuffled)
- Complete audio clips (no chunking), max 30s
- Per-language aggregate metrics at the end

Version 3 changes:
- Restored greedy decoding (num_beams=1). Beam search was pulling
  undertrained languages towards English output.
- Removed repetition_penalty — it suppressed natural Indic script patterns
  and pushed outputs towards English/Latin characters.
- Kept no_repeat_ngram_size=3 as the SOLE anti-repetition measure.
  This directly blocks the exact repetition loops seen in v1 (e.g. কিকিকিকি)
  without any side effects on script selection.
- Loads 'best' checkpoint by default (saved during training by val loss)

Requires: pip install transformers peft bitsandbytes soundfile scipy jiwer tqdm pandas matplotlib
"""

import os
import csv
import time
import json
import random
import torch
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple
import jiwer

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

# =========================
# CONFIG
# =========================

BASE_MODEL_NAME = "openai/whisper-large-v3"

# Path to your dataset and LoRA adapter (adjust these)
DATASET_DIR = os.path.join(
    os.path.expanduser("~"),
    "datasets",
    "IndicVoices_subset_4lang_combined",
)
# Override: uncomment and set your actual path below
DATASET_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined"
# DATASET_DIR = r"/path/to/data/indicvocies_finetune"                  # Windows

TEST_DIR = os.path.join(DATASET_DIR, "test")

# Use 'best' checkpoint by default (saved during training based on val loss)
# Fall back to main output dir if 'best' doesn't exist
LORA_ADAPTER_DIR = os.path.join(
    r"/path/to/data/IndicVoices_subset_4lang_combined",
    "whisper_lora_indic",
    "best",
)
if not os.path.exists(LORA_ADAPTER_DIR):
    LORA_ADAPTER_DIR = os.path.join(DATASET_DIR, "whisper_lora_indic")

OUTPUT_DIR = os.path.join(DATASET_DIR, "whisper_inference_results")

BATCH_SIZE = 4  # Display-batch: process 2 random audio files at a time
RANDOM_SEED = 42
MAX_AUDIO_SECONDS = 30  # Whisper hard limit

# --- Generation Config ---
# Greedy decoding (num_beams=1) is critical — beam search pulls undertrained
# languages towards English. Only no_repeat_ngram_size is used for anti-repetition.
MAX_NEW_TOKENS = 225  # Most transcriptions are <100 tokens
NO_REPEAT_NGRAM_SIZE = 3  # Blocks any 3-gram from repeating (fixes কিকিকিকি loops)

# ✅ NEW: Bootstrap configuration for confidence intervals
BOOTSTRAP_ITERATIONS = 1000  # Number of bootstrap samples for CI
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Disable CPU multithreading
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

TARGET_SR = 16000

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
    "or": "bengali",
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
# METRICS COMPUTATION
# =========================


def compute_asr_metrics(reference, hypothesis):
    """
    Compute comprehensive ASR metrics between reference and hypothesis strings.

    Returns dict with:
      - CER, char_precision, char_recall, char_f1
      - WER, word_precision, word_recall, word_f1
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

    char_hits = char_out.hits
    char_subs = char_out.substitutions
    char_ins = char_out.insertions
    char_dels = char_out.deletions

    char_pred_total = char_hits + char_subs + char_ins
    char_ref_total = char_hits + char_subs + char_dels

    char_precision = char_hits / char_pred_total if char_pred_total > 0 else 0.0
    char_recall = char_hits / char_ref_total if char_ref_total > 0 else 0.0
    char_f1 = (
        2 * char_precision * char_recall / (char_precision + char_recall)
        if (char_precision + char_recall) > 0
        else 0.0
    )

    # --- Word-level metrics ---
    word_out = jiwer.process_words(ref, hyp)
    wer_val = word_out.wer

    word_hits = word_out.hits
    word_subs = word_out.substitutions
    word_ins = word_out.insertions
    word_dels = word_out.deletions

    word_pred_total = word_hits + word_subs + word_ins
    word_ref_total = word_hits + word_subs + word_dels

    word_precision = word_hits / word_pred_total if word_pred_total > 0 else 0.0
    word_recall = word_hits / word_ref_total if word_ref_total > 0 else 0.0
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


# =========================================================
# ✅ NEW: BOOTSTRAP CONFIDENCE INTERVAL
# =========================================================
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


# =========================================================
# ✅ NEW: VISUALIZATION PLOTS
# =========================================================
def generate_plots(results: List[Dict], metrics: Dict, output_dir: str):
    """Generate visualization plots for research publication."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- Plot 1: WER box plot + WER/CER bar chart with CI ---
        lang_wers = defaultdict(list)
        for r in results:
            lang_wers[r["lang"]].append(r["wer"])

        langs = sorted(lang_wers.keys())
        wer_data = [lang_wers[lang] for lang in langs]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot of WER by language
        axes[0].boxplot(wer_data, labels=langs)
        axes[0].set_xlabel("Language")
        axes[0].set_ylabel("Word Error Rate (WER)")
        axes[0].set_title("WER Distribution by Language")
        axes[0].grid(True, alpha=0.3)

        # Bar plot of average WER/CER with confidence intervals
        x = range(len(langs))
        avg_wers = [metrics["per_language"][lang]["wer"] for lang in langs]
        wer_errors_lower = [
            metrics["per_language"][lang]["wer"]
            - metrics["per_language"][lang]["wer_ci_lower"]
            for lang in langs
        ]
        wer_errors_upper = [
            metrics["per_language"][lang]["wer_ci_upper"]
            - metrics["per_language"][lang]["wer"]
            for lang in langs
        ]

        avg_cers = [metrics["per_language"][lang]["cer"] for lang in langs]
        cer_errors_lower = [
            metrics["per_language"][lang]["cer"]
            - metrics["per_language"][lang]["cer_ci_lower"]
            for lang in langs
        ]
        cer_errors_upper = [
            metrics["per_language"][lang]["cer_ci_upper"]
            - metrics["per_language"][lang]["cer"]
            for lang in langs
        ]

        width = 0.35
        axes[1].bar(
            [i - width / 2 for i in x],
            avg_wers,
            width,
            yerr=[wer_errors_lower, wer_errors_upper],
            label="WER",
            color="steelblue",
            capsize=5,
        )
        axes[1].bar(
            [i + width / 2 for i in x],
            avg_cers,
            width,
            yerr=[cer_errors_lower, cer_errors_upper],
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

        # --- Plot 2: WER vs CER scatter plot ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        colors = {
            "Assamese": "blue",
            "Bengali": "green",
            "Hindi": "red",
            "Odia": "purple",
        }
        for lang in langs:
            lang_results = [r for r in results if r["lang"] == lang]
            wers_list = [r["wer"] for r in lang_results]
            cers_list = [r["cer"] for r in lang_results]
            ax.scatter(
                wers_list, cers_list, alpha=0.5, label=lang, c=colors.get(lang, "gray")
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

        y_pos = range(len(langs) + 1)  # +1 for overall

        # Overall
        overall_wer = metrics["overall"]["wer"]
        overall_lower = metrics["overall"]["wer_ci_lower"]
        overall_upper = metrics["overall"]["wer_ci_upper"]
        ax.errorbar(
            overall_wer,
            0,
            xerr=[[overall_wer - overall_lower], [overall_upper - overall_wer]],
            fmt="o",
            color="black",
            capsize=5,
            markersize=8,
            label="OVERALL",
        )

        # Per-language
        for i, lang in enumerate(langs):
            lang_wer = metrics["per_language"][lang]["wer"]
            lang_lower = metrics["per_language"][lang]["wer_ci_lower"]
            lang_upper = metrics["per_language"][lang]["wer_ci_upper"]
            ax.errorbar(
                lang_wer,
                i + 1,
                xerr=[[lang_wer - lang_lower], [lang_upper - lang_wer]],
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


def load_test_samples(test_dir):
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

        filename = os.path.basename(resolved_path)
        samples.append(
            {
                "audio": resolved_path,
                "text": str(row["text"]),
                "lang": get_language_name(lang_code),
                "lang_code": lang_code,
                "whisper_lang": get_whisper_language(lang_code),
                "filename": filename,
            }
        )

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} samples (audio files not found)")
    if skipped_unsupported_lang > 0:
        print(f"  ⚠️  Skipped {skipped_unsupported_lang} samples (unsupported language)")
    print(f"  Loaded {len(samples)} test samples")
    return samples


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
# MAIN INFERENCE
# =========================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_samples = load_test_samples(TEST_DIR)
    if not test_samples:
        print("No test samples found. Exiting.")
        return

    random.shuffle(test_samples)

    lang_counts = defaultdict(int)
    for s in test_samples:
        lang_counts[s["lang"]] += 1
    print("\nTest language distribution:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")

    # =========================
    # LOAD MODEL + LORA
    # =========================

    print(f"\n{'=' * 60}")
    print("Loading model and LoRA adapter...")
    print(f"{'=' * 60}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading processor from: {BASE_MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME)

    print(f"Loading base model in 4-bit: {BASE_MODEL_NAME}")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from: {LORA_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    model.eval()

    print("✅ Model loaded successfully")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {alloc:.1f}GB alloc / {reserved:.1f}GB reserved")

    # =========================
    # RESULTS CSV SETUP
    # =========================

    results_csv_path = os.path.join(OUTPUT_DIR, "inference_results.csv")
    summary_csv_path = os.path.join(OUTPUT_DIR, "per_language_summary.csv")
    overall_csv_path = os.path.join(OUTPUT_DIR, "overall_summary.csv")

    csv_headers = [
        "sample_idx",
        "filename",
        "actual_language",
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
    print(f"Batch display size: {BATCH_SIZE}")
    print(f"Decoding: greedy (no beam search)")
    print(f"No-repeat n-gram size: {NO_REPEAT_NGRAM_SIZE}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"LoRA adapter: {LORA_ADAPTER_DIR}")
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

        print(f"\n{'─' * 60}")
        print(
            f"Batch {batch_num} | Samples {i + 1}-{i + len(batch)} of {len(test_samples)}"
        )
        print(f"{'─' * 60}")

        for j, sample in enumerate(batch):
            filename = sample["filename"]
            actual_lang = sample["lang"]
            whisper_lang = sample["whisper_lang"]
            reference = sample["text"]

            # Load and process audio
            wav = load_audio(sample["audio"])

            input_features = processor.feature_extractor(
                wav, sampling_rate=TARGET_SR, return_tensors="pt"
            ).input_features.to(model.device)

            # Get forced decoder IDs for this language
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=whisper_lang, task="transcribe"
            )

            # Generate transcription
            # Greedy decoding + no_repeat_ngram_size only.
            # DO NOT use beam search or repetition_penalty — they pull
            # undertrained languages towards English output.
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                predicted_ids = model.generate(
                    input_features=input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                )

            # Decode transcription
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            # Compute metrics
            metrics = compute_asr_metrics(reference, transcription)

            # Store result
            result = {
                "sample_idx": i + j + 1,
                "filename": filename,
                "actual_language": actual_lang,
                "reference_text": reference,
                "predicted_text": transcription,
                **metrics,
            }
            all_results.append(result)
            per_lang_results[actual_lang].append(result)

            # Write to CSV immediately
            with open(results_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        result["sample_idx"],
                        result["filename"],
                        result["actual_language"],
                        result["reference_text"],
                        result["predicted_text"],
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
            print(f"     Language:   {actual_lang}")
            print(
                f"     Reference:  {reference[:120]}{'...' if len(reference) > 120 else ''}"
            )
            print(
                f"     Predicted:  {transcription[:120]}{'...' if len(transcription) > 120 else ''}"
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
    # AGGREGATE METRICS
    # =========================

    def aggregate_metrics(results_list):
        if not results_list:
            return {}

        all_refs = [r["reference_text"] for r in results_list]
        all_hyps = [r["predicted_text"] for r in results_list]

        pairs = [
            (ref, hyp)
            for ref, hyp in zip(all_refs, all_hyps)
            if ref.strip() or hyp.strip()
        ]
        if not pairs:
            return {
                "avg_cer": 0.0,
                "avg_char_precision": 1.0,
                "avg_char_recall": 1.0,
                "avg_char_f1": 1.0,
                "avg_wer": 0.0,
                "avg_word_precision": 1.0,
                "avg_word_recall": 1.0,
                "avg_word_f1": 1.0,
                "corpus_cer": 0.0,
                "corpus_wer": 0.0,
                "count": len(results_list),
            }

        refs, hyps = zip(*pairs)

        try:
            corpus_cer = jiwer.cer(list(refs), list(hyps))
        except Exception:
            corpus_cer = float("nan")
        try:
            corpus_wer = jiwer.wer(list(refs), list(hyps))
        except Exception:
            corpus_wer = float("nan")

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

        n = len(results_list)
        avg_cer = sum(r["cer"] for r in results_list) / n
        avg_char_prec = sum(r["char_precision"] for r in results_list) / n
        avg_char_rec = sum(r["char_recall"] for r in results_list) / n
        avg_char_f1 = sum(r["char_f1"] for r in results_list) / n
        avg_wer = sum(r["wer"] for r in results_list) / n
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
    # OVERALL SUMMARY
    # =========================

    overall = aggregate_metrics(all_results)

    print(f"\n{'=' * 70}")
    print(f"OVERALL RESULTS ({overall['count']} samples)")
    print(f"{'=' * 70}")
    print(
        f"Time elapsed: {elapsed:.1f}s ({elapsed / max(len(all_results), 1):.2f}s per sample)"
    )
    print(f"\n  ── Corpus-level Character Metrics ──")
    print(f"     CER:       {overall['corpus_cer']:.4f}")
    print(f"     Precision: {overall['corpus_char_precision']:.4f}")
    print(f"     Recall:    {overall['corpus_char_recall']:.4f}")
    print(f"     F1:        {overall['corpus_char_f1']:.4f}")
    print(f"\n  ── Corpus-level Word Metrics ──")
    print(f"     WER:       {overall['corpus_wer']:.4f}")
    print(f"     Precision: {overall['corpus_word_precision']:.4f}")
    print(f"     Recall:    {overall['corpus_word_recall']:.4f}")
    print(f"     F1:        {overall['corpus_word_f1']:.4f}")
    print(f"\n  ── Sample-averaged Character Metrics ──")
    print(f"     CER:       {overall['avg_cer']:.4f}")
    print(f"     Precision: {overall['avg_char_precision']:.4f}")
    print(f"     Recall:    {overall['avg_char_recall']:.4f}")
    print(f"     F1:        {overall['avg_char_f1']:.4f}")
    print(f"\n  ── Sample-averaged Word Metrics ──")
    print(f"     WER:       {overall['avg_wer']:.4f}")
    print(f"     Precision: {overall['avg_word_precision']:.4f}")
    print(f"     Recall:    {overall['avg_word_recall']:.4f}")
    print(f"     F1:        {overall['avg_word_f1']:.4f}")

    with open(overall_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in overall.items():
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
        lang_agg = aggregate_metrics(lang_res)

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

        lang_summary_rows.append(
            {
                "language": lang,
                **lang_agg,
            }
        )

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
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
    # OUTPUT FILES (original)
    # =========================

    print(f"\n{'=' * 70}")
    print("OUTPUT FILES")
    print(f"{'=' * 70}")
    print(f"  📄 Per-sample results:     {results_csv_path}")
    print(f"  📄 Per-language summary:   {summary_csv_path}")
    print(f"  📄 Overall summary:        {overall_csv_path}")

    # =========================================================
    # ✅ NEW: BOOTSTRAP CI METRICS + JSON EXPORT
    # =========================================================
    print(f"\n📊 Computing metrics with {BOOTSTRAP_ITERATIONS} bootstrap iterations...")

    # Build per-sample list in the format expected by bootstrap/plot functions
    ci_sample_results = []
    for r in all_results:
        ci_sample_results.append(
            {
                "audio": r["filename"],
                "reference": r["reference_text"],
                "prediction": r["predicted_text"],
                "wer": r["wer"],
                "cer": r["cer"],
                "lang": r["actual_language"],
            }
        )

    ci_metrics = {
        "overall": {},
        "per_language": {},
    }

    # Overall bootstrap CI
    all_wers = [r["wer"] for r in ci_sample_results]
    all_cers = [r["cer"] for r in ci_sample_results]

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
        "total_samples": len(ci_sample_results),
    }

    # Per-language bootstrap CI
    lang_results_grouped = defaultdict(list)
    for r in ci_sample_results:
        lang_results_grouped[r["lang"]].append(r)

    for lang_name, lang_list in lang_results_grouped.items():
        lang_wers = [r["wer"] for r in lang_list]
        lang_cers = [r["cer"] for r in lang_list]

        lang_wer_mean, lang_wer_lower, lang_wer_upper = compute_bootstrap_ci(
            lang_wers, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS
        )
        lang_cer_mean, lang_cer_lower, lang_cer_upper = compute_bootstrap_ci(
            lang_cers, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS
        )

        ci_metrics["per_language"][lang_name] = {
            "wer": lang_wer_mean,
            "wer_ci_lower": lang_wer_lower,
            "wer_ci_upper": lang_wer_upper,
            "cer": lang_cer_mean,
            "cer_ci_lower": lang_cer_lower,
            "cer_ci_upper": lang_cer_upper,
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

    # Save CI metrics summary CSV
    ci_metrics_csv_path = os.path.join(OUTPUT_DIR, "metrics_summary_with_ci.csv")
    with open(ci_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
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
        # Overall
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
        # Per-language
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
    print(f"  📄 CI Metrics CSV:         {ci_metrics_csv_path}")

    # Save metrics as JSON
    ci_metrics_json_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(ci_metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(ci_metrics, f, indent=2)
    print(f"  📄 CI Metrics JSON:        {ci_metrics_json_path}")

    # =========================================================
    # ✅ NEW: GENERATE VISUALIZATION PLOTS
    # =========================================================
    print(f"\n📈 Generating visualization plots...")
    generate_plots(ci_sample_results, ci_metrics, OUTPUT_DIR)

    print(
        f"\n✅ Inference complete. {len(all_results)} samples processed in {elapsed:.1f}s."
    )
    print("Results are ready for research publication.")


if __name__ == "__main__":
    main()
