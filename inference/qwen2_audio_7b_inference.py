"""
Qwen2-Audio-7B-Instruct LoRA Inference for Indic Languages - v8 (FIXED)

Critical Fixes from v7:
1. CORRECT WER/CER AGGREGATION: Uses aggregate method (total errors / total reference words)
   per NIST, Mozilla DeepSpeech, and HuggingFace standards.
   Reference: WER = (S + D + I) / N where N = total reference words
   NOT average of per-sample WERs!

2. BOOTSTRAP CONFIDENCE INTERVALS: Uses bootstrap resampling (B=2000) for 95% CI
   per Bisani & Ney (ICASSP 2004), cited 314+ times, the standard for ASR evaluation.
   Reference: https://www.researchgate.net/publication/4087402

3. UPDATED JIWER API: Uses new jiwer API (process_words, process_characters)
   instead of deprecated compute_measures.

Other features from v7:
- Consistent audio resampling with training script
- Data validation and file existence checks
- Deterministic mode option for reproducibility
- Inference time tracking
- Error analysis output
- Per-language statistics

OUTPUT: results_detailed.csv, per_language_results.csv, error_analysis.csv,
        inference_summary.txt, metrics_summary_with_ci.csv, metrics_summary.json,
        inference_metrics.png, wer_vs_cer_scatter.png, confidence_intervals.png
        in OUTPUT_DIR
"""

import os
import time
import random
import json
import math
import torch
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from scipy import stats as scipy_stats
from math import gcd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import warnings
import csv

from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    BitsAndBytesConfig,
)

from peft import PeftModel
import jiwer

# =========================
# CONFIG
# =========================

BASE_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
LORA_PATH = r"/path/to/data/IndicVoices_subset_4lang_combined/qwen2_audio_lora_indic"

# Path to test data directory
DATA_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/test"

# Output directory for all results
OUTPUT_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/inference_results"

BATCH_SIZE = 4
RANDOM_SEED = 42
MAX_AUDIO_SECONDS = 60

# --- Inference Options ---
USE_TORCH_COMPILE = False    # Set True for faster inference (may fail on some setups)
DETERMINISTIC = False        # Set True for reproducible inference (slower)
NORMALIZE_TEXT = True        # Normalize text before computing metrics

# --- Bootstrap Config ---
N_BOOTSTRAP = 2000           # Number of bootstrap samples (standard: 2000)
CONFIDENCE_LEVEL = 0.95      # 95% confidence interval

# --- Error Analysis ---
CER_THRESHOLD = 0.5          # Flag samples with CER > threshold for error analysis
WER_THRESHOLD = 1.0          # Flag samples with WER > threshold for error analysis

DEVICE = "cuda"

# =========================
# SET SEED AND REPRODUCIBILITY
# =========================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

if DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("⚠️  Deterministic mode enabled. Inference will be reproducible.")
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# =========================
# LANGUAGE CODE MAPPING
# =========================

LANG_CODE_TO_NAME = {
    "as": "Assamese",
    "bn": "Bengali",
    "hi": "Hindi",
    "or": "Odia",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
}


def get_language_name(lang_code):
    code = str(lang_code).lower().strip()
    return LANG_CODE_TO_NAME.get(code, code.capitalize())


# =========================
# TEXT NORMALIZATION
# =========================

def normalize_text(text):
    """Normalize text for fair metric computation."""
    if not NORMALIZE_TEXT:
        return text
    
    text = str(text).strip()
    text = ' '.join(text.split())
    text = text.strip('.,!?;:"\'-')
    
    return text


# =========================
# METRIC FUNCTIONS (Using new jiwer API)
# =========================

def compute_wer_measures(references, hypotheses):
    """
    Compute aggregate WER using jiwer's process_words (new API).
    This is the CORRECT way: total errors / total reference words.
    
    Returns dict with: wer, mer, wil, wip, hits, substitutions, deletions, insertions
    """
    if isinstance(references, str):
        references = [references]
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    
    # Use new jiwer API
    output = jiwer.process_words(references, hypotheses)
    
    return {
        'wer': output.wer,
        'mer': output.mer,
        'wil': output.wil,
        'wip': output.wip,
        'hits': output.hits,
        'substitutions': output.substitutions,
        'deletions': output.deletions,
        'insertions': output.insertions,
    }


def compute_cer_measures(references, hypotheses):
    """
    Compute aggregate CER using jiwer's process_characters (new API).
    This is the CORRECT way: total char errors / total reference chars.
    
    Returns: cer, substitutions, deletions, insertions, hits
    """
    if isinstance(references, str):
        references = [references]
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    
    # Use new jiwer API
    output = jiwer.process_characters(references, hypotheses)
    
    return output.cer, output.substitutions, output.deletions, output.insertions, output.hits


def char_precision_recall_f1(pred, ref):
    """Calculate character-level precision, recall, F1."""
    pred = list(pred)
    ref = list(ref)

    if len(pred) == 0 and len(ref) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0:
        return 0.0, 0.0, 0.0

    pred_c = Counter(pred)
    ref_c = Counter(ref)

    tp = sum((pred_c & ref_c).values())
    fp = sum((pred_c - ref_c).values())
    fn = sum((ref_c - pred_c).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def word_precision_recall_f1(pred, ref):
    """Calculate word-level precision, recall, F1."""
    pred_words = pred.split()
    ref_words = ref.split()

    if len(pred_words) == 0 and len(ref_words) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_words) == 0:
        return 0.0, 0.0, 0.0

    pred_c = Counter(pred_words)
    ref_c = Counter(ref_words)

    tp = sum((pred_c & ref_c).values())
    fp = sum((pred_c - ref_c).values())
    fn = sum((ref_c - pred_c).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


# =========================
# BOOTSTRAP CONFIDENCE INTERVAL
# =========================

def bootstrap_wer_ci(references, hypotheses, n_bootstrap=N_BOOTSTRAP, 
                     confidence=CONFIDENCE_LEVEL, seed=RANDOM_SEED):
    """
    Compute bootstrap confidence interval for WER.
    
    Per Bisani & Ney (ICASSP 2004):
    - Resample utterances with replacement
    - Compute WER for each bootstrap sample
    - Take percentiles for CI
    
    Returns: (wer_mean, wer_std, ci_low, ci_high)
    """
    np.random.seed(seed)
    n = len(references)
    
    if n < 2:
        measures = compute_wer_measures(references, hypotheses)
        return measures['wer'], 0.0, measures['wer'], measures['wer']
    
    wer_samples = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        boot_refs = [references[i] for i in indices]
        boot_hyps = [hypotheses[i] for i in indices]
        
        measures = compute_wer_measures(boot_refs, boot_hyps)
        wer_samples.append(measures['wer'])
    
    wer_samples = np.array(wer_samples)
    wer_mean = np.mean(wer_samples)
    wer_std = np.std(wer_samples, ddof=1)
    
    # Percentile method for CI
    alpha = 1 - confidence
    ci_low = np.percentile(wer_samples, 100 * alpha / 2)
    ci_high = np.percentile(wer_samples, 100 * (1 - alpha / 2))
    
    return wer_mean, wer_std, ci_low, ci_high


def bootstrap_cer_ci(references, hypotheses, n_bootstrap=N_BOOTSTRAP,
                     confidence=CONFIDENCE_LEVEL, seed=RANDOM_SEED):
    """
    Compute bootstrap confidence interval for CER.
    
    Returns: (cer_mean, cer_std, ci_low, ci_high)
    """
    np.random.seed(seed)
    n = len(references)
    
    if n < 2:
        cer, _, _, _, _ = compute_cer_measures(references, hypotheses)
        return cer, 0.0, cer, cer
    
    cer_samples = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_refs = [references[i] for i in indices]
        boot_hyps = [hypotheses[i] for i in indices]
        
        cer, _, _, _, _ = compute_cer_measures(boot_refs, boot_hyps)
        cer_samples.append(cer)
    
    cer_samples = np.array(cer_samples)
    cer_mean = np.mean(cer_samples)
    cer_std = np.std(cer_samples, ddof=1)
    
    alpha = 1 - confidence
    ci_low = np.percentile(cer_samples, 100 * alpha / 2)
    ci_high = np.percentile(cer_samples, 100 * (1 - alpha / 2))
    
    return cer_mean, cer_std, ci_low, ci_high


def bootstrap_metric_ci(values, n_bootstrap=N_BOOTSTRAP,
                        confidence=CONFIDENCE_LEVEL, seed=RANDOM_SEED):
    """
    Compute bootstrap CI for any metric from per-sample values.
    
    Returns: (mean, std, ci_low, ci_high)
    """
    np.random.seed(seed)
    values = np.array(values)
    n = len(values)
    
    if n < 2:
        return np.mean(values), 0.0, np.mean(values), np.mean(values)
    
    boot_means = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_means.append(np.mean(values[indices]))
    
    boot_means = np.array(boot_means)
    mean = np.mean(boot_means)
    std = np.std(boot_means, ddof=1)
    
    alpha = 1 - confidence
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    
    return mean, std, ci_low, ci_high


# =========================
# DATA VALIDATION
# =========================

def validate_csv(df, csv_path, required_cols=['path', 'lang', 'text']):
    """Validate CSV has required columns and no critical missing values."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV {csv_path} missing required columns: {missing_cols}")
    
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        warnings.warn(f"CSV {csv_path} has NaN values:\n{nan_counts[nan_counts > 0]}")
        original_len = len(df)
        df = df.dropna(subset=required_cols)
        print(f"  Dropped {original_len - len(df)} rows with NaN values")
    
    return df


def check_audio_exists(samples, name="samples"):
    """Check that audio files exist and report missing ones."""
    missing = []
    for s in samples:
        if not os.path.exists(s['audio']):
            missing.append(s['audio'])
    
    if missing:
        print(f"⚠️  {len(missing)} audio files not found in {name}:")
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    
    return len(missing)


# =========================
# AUDIO LOADER
# =========================

def load_audio(path, target_sr, max_seconds):
    """Load audio with improved resampling for high sample rates."""
    try:
        wav, sr = sf.read(path, dtype='float32')
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {path}: {e}")
    
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    
    if len(wav) == 0:
        raise RuntimeError(f"Audio file {path} is empty")
    
    current_sr = sr
    while current_sr > 2 * target_sr:
        intermediate_sr = current_sr // 2
        g = gcd(current_sr, intermediate_sr)
        wav = resample_poly(wav, intermediate_sr // g, current_sr // g).astype(np.float32)
        current_sr = intermediate_sr
    
    if current_sr != target_sr:
        g = gcd(current_sr, target_sr)
        wav = resample_poly(wav, target_sr // g, current_sr // g).astype(np.float32)
    
    max_len = int(target_sr * max_seconds)
    if len(wav) > max_len:
        wav = wav[:max_len]
    
    return wav


# =========================================================
# VISUALIZATION PLOTS (ADDED)
# =========================================================

def generate_plots(results_rows: List[Dict], ci_metrics: Dict, output_dir: str):
    """Generate 3 visualization plots for research publication."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Collect per-language WER lists
        lang_wers = defaultdict(list)
        for r in results_rows:
            lang_wers[r["language"]].append(r["wer"])

        langs = sorted(lang_wers.keys())
        wer_data = [lang_wers[lang] for lang in langs]

        # Color map for languages
        base_colors = {
            "Assamese": "blue", "Bengali": "green", "Hindi": "red",
            "Odia": "purple", "Urdu": "orange", "Tamil": "brown",
            "Telugu": "pink", "Malayalam": "cyan", "Kannada": "olive",
            "Gujarati": "teal", "Marathi": "navy", "Punjabi": "magenta",
        }

        # --- Plot 1: WER box plot + WER/CER bar chart with CI ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].boxplot(wer_data, labels=langs)
        axes[0].set_xlabel("Language")
        axes[0].set_ylabel("Word Error Rate (WER)")
        axes[0].set_title("WER Distribution by Language")
        axes[0].grid(True, alpha=0.3)
        if len(langs) > 4:
            axes[0].tick_params(axis='x', rotation=45)

        x = range(len(langs))
        avg_wers = [ci_metrics["per_language"][lang]["wer"] for lang in langs]
        wer_err_lo = [ci_metrics["per_language"][lang]["wer"] - ci_metrics["per_language"][lang]["wer_ci_lower"] for lang in langs]
        wer_err_hi = [ci_metrics["per_language"][lang]["wer_ci_upper"] - ci_metrics["per_language"][lang]["wer"] for lang in langs]
        avg_cers = [ci_metrics["per_language"][lang]["cer"] for lang in langs]
        cer_err_lo = [ci_metrics["per_language"][lang]["cer"] - ci_metrics["per_language"][lang]["cer_ci_lower"] for lang in langs]
        cer_err_hi = [ci_metrics["per_language"][lang]["cer_ci_upper"] - ci_metrics["per_language"][lang]["cer"] for lang in langs]

        width = 0.35
        axes[1].bar([i - width / 2 for i in x], avg_wers, width, yerr=[wer_err_lo, wer_err_hi],
                    label="WER", color="steelblue", capsize=5)
        axes[1].bar([i + width / 2 for i in x], avg_cers, width, yerr=[cer_err_lo, cer_err_hi],
                    label="CER", color="coral", capsize=5)
        axes[1].set_xlabel("Language")
        axes[1].set_ylabel("Error Rate")
        axes[1].set_title("Average WER and CER by Language (with 95% CI)")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(langs, rotation=45 if len(langs) > 4 else 0)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "inference_metrics.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Plot saved to: {plot_path}")

        # --- Plot 2: WER vs CER scatter plot ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for lang in langs:
            lr = [r for r in results_rows if r["language"] == lang]
            ax.scatter([r["wer"] for r in lr], [r["cer"] for r in lr],
                       alpha=0.5, label=lang, c=base_colors.get(lang, "gray"))
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
        fig, ax = plt.subplots(1, 1, figsize=(10, max(6, 2 + len(langs) * 0.8)))
        y_pos = range(len(langs) + 1)

        ov = ci_metrics["overall"]
        ax.errorbar(ov["wer"], 0,
                    xerr=[[ov["wer"] - ov["wer_ci_lower"]], [ov["wer_ci_upper"] - ov["wer"]]],
                    fmt='o', color='black', capsize=5, markersize=8, label='OVERALL')

        for i, lang in enumerate(langs):
            lm = ci_metrics["per_language"][lang]
            ax.errorbar(lm["wer"], i + 1,
                        xerr=[[lm["wer"] - lm["wer_ci_lower"]], [lm["wer_ci_upper"] - lm["wer"]]],
                        fmt='o', color=base_colors.get(lang, "gray"), capsize=5, markersize=8)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(["OVERALL"] + langs)
        ax.set_xlabel("Word Error Rate (WER)")
        ax.set_title("WER with 95% Confidence Intervals")
        ax.grid(True, alpha=0.3, axis='x')
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
# MAIN
# =========================

def main():
    inference_start_time = time.time()
    
    print("=" * 60)
    print("Qwen2-Audio LoRA Inference for Indic Languages v8 (FIXED)")
    print("=" * 60)
    print(f"Base Model: {BASE_MODEL}")
    print(f"LoRA Path: {LORA_PATH}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Text Normalization: {NORMALIZE_TEXT}")
    print(f"Bootstrap Samples: {N_BOOTSTRAP}")
    print(f"Deterministic Mode: {DETERMINISTIC}")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================
    # LOAD ALL FILES
    # =========================
    
    samples = []
    
    dir_name = os.path.basename(DATA_DIR)
    csv_path = os.path.join(DATA_DIR, f"{dir_name}.csv")
    
    print(f"\nLoading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = validate_csv(df, csv_path)
    
    print(f"CSV columns: {list(df.columns)}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Unique language codes: {df['lang'].unique()}")
    
    for _, row in df.iterrows():
        audio_path = os.path.join(DATA_DIR, str(row["path"]))
        lang_code = str(row["lang"])
        lang_name = get_language_name(lang_code)
        
        samples.append({
            "audio": audio_path,
            "text": str(row["text"]),
            "lang": lang_name,
            "lang_code": lang_code,
        })
    
    print(f"Total inference samples: {len(samples)}")
    
    missing = check_audio_exists(samples, "test set")
    if missing > 0:
        samples = [s for s in samples if os.path.exists(s['audio'])]
        print(f"  Remaining samples after removing missing files: {len(samples)}")
    
    samples.sort(key=lambda x: os.path.getsize(x["audio"]))
    
    lang_counts = defaultdict(int)
    for s in samples:
        lang_counts[s["lang"]] += 1
    print(f"\nLanguage distribution:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")
    
    # =========================
    # LOAD MODEL (4-bit)
    # =========================
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    target_sr = processor.feature_extractor.sampling_rate
    print(f"Target sampling rate: {target_sr}")
    
    print("Loading base model in 4-bit...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    
    if USE_TORCH_COMPILE:
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"⚠️  torch.compile failed, continuing without: {e}")
    
    model_name_print = f"{BASE_MODEL} + LoRA({os.path.basename(LORA_PATH)})"
    
    # =========================
    # STORAGE
    # =========================
    
    results_rows = []
    error_analysis_rows = []
    
    # Store all predictions and references for aggregate metric computation
    all_references = []
    all_hypotheses = []
    
    # Per-language storage
    per_lang_refs = defaultdict(list)
    per_lang_hyps = defaultdict(list)
    
    # Per-sample metrics (for F1, precision, recall)
    global_char_prec = []
    global_char_rec = []
    global_char_f1 = []
    global_word_prec = []
    global_word_rec = []
    global_word_f1 = []
    
    per_lang_sample_metrics = defaultdict(lambda: {
        'char_p': [], 'char_r': [], 'char_f1': [],
        'word_p': [], 'word_r': [], 'word_f1': [],
        'latency': []
    })
    
    latencies = []
    
    # =========================
    # WARMUP
    # =========================
    
    print("\nWarming up model...")
    dummy_wav = torch.randn(target_sr).numpy()
    dummy_conv = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": "Test"},
            ],
        }
    ]
    dummy_text = processor.apply_chat_template(
        dummy_conv, add_generation_prompt=True, tokenize=False
    )
    try:
        dummy_inputs = processor(
            text=dummy_text, audio=dummy_wav,
            sampling_rate=target_sr, return_tensors="pt", padding=True,
        )
    except TypeError:
        dummy_inputs = processor(
            text=dummy_text, audios=dummy_wav,
            sampling_rate=target_sr, return_tensors="pt", padding=True,
        )
    dummy_inputs = {k: v.to(model.device) for k, v in dummy_inputs.items()}
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        _ = model.generate(**dummy_inputs, max_new_tokens=1, use_cache=True)
    print("Warmup complete.")
    
    # =========================
    # PRE-COMPUTE CHAT TEMPLATES
    # =========================
    
    print("\nPre-computing chat templates...")
    for s in tqdm(samples, desc="Building templates"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "placeholder"},
                    {
                        "type": "text",
                        "text": f"Transcribe the following audio in {s['lang']} language.",
                    },
                ],
            }
        ]
        s["full_text"] = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
    
    # =========================
    # INFERENCE LOOP (BATCHED)
    # =========================
    
    progress = tqdm(total=len(samples), desc="Inference")
    
    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch_samples = samples[batch_start : batch_start + BATCH_SIZE]
        
        batch_wavs = []
        batch_valid_samples = []
        
        for sample in batch_samples:
            try:
                wav = load_audio(sample["audio"], target_sr, MAX_AUDIO_SECONDS)
                batch_wavs.append(wav)
                batch_valid_samples.append(sample)
            except Exception as e:
                print(f"\n⚠️  Error loading {sample['audio']}: {e}")
                progress.update(1)
        
        if not batch_valid_samples:
            continue
        
        batch_texts = [s["full_text"] for s in batch_valid_samples]
        
        try:
            inputs = processor(
                text=batch_texts, audio=batch_wavs,
                sampling_rate=target_sr, return_tensors="pt", padding=True,
            )
        except TypeError:
            inputs = processor(
                text=batch_texts, audios=batch_wavs,
                sampling_rate=target_sr, return_tensors="pt", padding=True,
            )
        
        input_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        batch_start_time = time.time()
        
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        batch_end_time = time.time()
        batch_latency = (batch_end_time - batch_start_time) / len(batch_valid_samples)
        
        for i, sample in enumerate(batch_valid_samples):
            lang = sample["lang"]
            lang_code = sample["lang_code"]
            gt_text = sample["text"]
            fname = os.path.basename(sample["audio"])
            
            generated_tokens = generated_ids[i, input_len:]
            raw_pred = processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            
            if "assistant" in raw_pred.lower():
                last_assistant_idx = raw_pred.lower().rfind("assistant")
                pred = raw_pred[last_assistant_idx + len("assistant") :].strip()
            else:
                pred = raw_pred
            
            gt = normalize_text(gt_text)
            pred_norm = normalize_text(pred)
            
            # Store for aggregate metrics
            all_references.append(gt)
            all_hypotheses.append(pred_norm)
            per_lang_refs[lang].append(gt)
            per_lang_hyps[lang].append(pred_norm)
            
            # Per-sample F1 metrics
            char_p, char_r, char_f1 = char_precision_recall_f1(pred_norm, gt)
            word_p, word_r, word_f1 = word_precision_recall_f1(pred_norm, gt)
            
            global_char_prec.append(char_p)
            global_char_rec.append(char_r)
            global_char_f1.append(char_f1)
            global_word_prec.append(word_p)
            global_word_rec.append(word_r)
            global_word_f1.append(word_f1)
            
            per_lang_sample_metrics[lang]['char_p'].append(char_p)
            per_lang_sample_metrics[lang]['char_r'].append(char_r)
            per_lang_sample_metrics[lang]['char_f1'].append(char_f1)
            per_lang_sample_metrics[lang]['word_p'].append(word_p)
            per_lang_sample_metrics[lang]['word_r'].append(word_r)
            per_lang_sample_metrics[lang]['word_f1'].append(word_f1)
            per_lang_sample_metrics[lang]['latency'].append(batch_latency)
            
            latencies.append(batch_latency)
            
            # Per-sample WER/CER for error analysis (using new jiwer API)
            sample_measures = compute_wer_measures(gt, pred_norm)
            sample_cer, _, _, _, _ = compute_cer_measures(gt, pred_norm)
            sample_wer = sample_measures['wer']
            
            if sample_cer > CER_THRESHOLD or sample_wer > WER_THRESHOLD:
                error_analysis_rows.append({
                    "language": lang,
                    "lang_code": lang_code,
                    "file": fname,
                    "prediction": pred,
                    "ground_truth": gt,
                    "cer": sample_cer,
                    "wer": sample_wer,
                    "cer_threshold_exceeded": sample_cer > CER_THRESHOLD,
                    "wer_threshold_exceeded": sample_wer > WER_THRESHOLD,
                })
            
            sample_idx = len(all_references)
            if sample_idx <= 50 or sample_idx % 100 == 0:
                print(f"\n{'='*50}")
                print(f"Sample {sample_idx}")
                print(f"Model: {model_name_print}")
                print(f"Language: {lang} ({lang_code})")
                print(f"File: {fname}")
                print(f"Prediction: {pred[:100]}..." if len(pred) > 100 else f"Prediction: {pred}")
                print(f"Ground Truth: {gt[:100]}..." if len(gt) > 100 else f"Ground Truth: {gt}")
                print(f"CER: {sample_cer:.4f} | WER: {sample_wer:.4f}")
                print(f"Char F1: {char_f1:.4f} | Word F1: {word_f1:.4f}")
            
            results_rows.append({
                "sample_idx": sample_idx,
                "language": lang,
                "lang_code": lang_code,
                "file": fname,
                "prediction": pred,
                "ground_truth": gt,
                "cer": sample_cer,
                "char_precision": char_p,
                "char_recall": char_r,
                "char_f1": char_f1,
                "wer": sample_wer,
                "word_precision": word_p,
                "word_recall": word_r,
                "word_f1": word_f1,
                "latency_sec": batch_latency,
            })
            
            progress.update(1)
    
    progress.close()
    
    total_inference_time = time.time() - inference_start_time
    
    # =========================
    # COMPUTE AGGREGATE METRICS (CORRECT WAY)
    # =========================
    
    print("\n" + "=" * 60)
    print("COMPUTING AGGREGATE METRICS (Bootstrap CI)")
    print("=" * 60)
    
    # Overall WER and CER (aggregate, not average of per-sample)
    overall_wer_measures = compute_wer_measures(all_references, all_hypotheses)
    overall_cer, overall_cer_S, overall_cer_D, overall_cer_I, overall_cer_C = \
        compute_cer_measures(all_references, all_hypotheses)
    
    # Bootstrap CIs
    print("  Computing bootstrap confidence intervals (this may take a moment)...")
    wer_mean, wer_std, wer_ci_low, wer_ci_high = bootstrap_wer_ci(
        all_references, all_hypotheses, N_BOOTSTRAP, CONFIDENCE_LEVEL, RANDOM_SEED
    )
    cer_mean, cer_std, cer_ci_low, cer_ci_high = bootstrap_cer_ci(
        all_references, all_hypotheses, N_BOOTSTRAP, CONFIDENCE_LEVEL, RANDOM_SEED
    )
    
    # Bootstrap CI for F1 metrics
    char_f1_mean, char_f1_std, char_f1_ci_low, char_f1_ci_high = bootstrap_metric_ci(global_char_f1)
    word_f1_mean, word_f1_std, word_f1_ci_low, word_f1_ci_high = bootstrap_metric_ci(global_word_f1)
    latency_mean, latency_std, latency_ci_low, latency_ci_high = bootstrap_metric_ci(latencies)
    
    # =========================
    # OVERALL RESULTS
    # =========================
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS (Bootstrap 95% CI)")
    print("=" * 60)
    print(f"\n--- Aggregate WER/CER ---")
    print(f"WER: {overall_wer_measures['wer']:.4f} (S={overall_wer_measures['substitutions']}, "
          f"D={overall_wer_measures['deletions']}, I={overall_wer_measures['insertions']})")
    print(f"WER Bootstrap: {wer_mean:.4f} ± {wer_std:.4f} (95% CI: [{wer_ci_low:.4f}, {wer_ci_high:.4f}])")
    print(f"\nCER: {overall_cer:.4f} (S={overall_cer_S}, D={overall_cer_D}, I={overall_cer_I})")
    print(f"CER Bootstrap: {cer_mean:.4f} ± {cer_std:.4f} (95% CI: [{cer_ci_low:.4f}, {cer_ci_high:.4f}])")
    
    print(f"\n--- F1 Metrics (Bootstrap 95% CI) ---")
    print(f"Char F1: {char_f1_mean:.4f} ± {char_f1_std:.4f} (95% CI: [{char_f1_ci_low:.4f}, {char_f1_ci_high:.4f}])")
    print(f"Word F1: {word_f1_mean:.4f} ± {word_f1_std:.4f} (95% CI: [{word_f1_ci_low:.4f}, {word_f1_ci_high:.4f}])")
    
    print(f"\n--- Latency ---")
    print(f"Per-sample: {latency_mean:.4f}s ± {latency_std:.4f}s (95% CI: [{latency_ci_low:.4f}, {latency_ci_high:.4f}])")
    print(f"Total inference time: {total_inference_time/3600:.2f} hours ({total_inference_time:.1f}s)")
    print(f"Throughput: {len(samples)/total_inference_time:.2f} samples/sec")
    
    # =========================
    # PER-LANGUAGE METRICS
    # =========================
    
    per_lang_rows = []
    
    print("\n" + "=" * 60)
    print("PER-LANGUAGE RESULTS (Bootstrap 95% CI)")
    print("=" * 60)
    
    for lang in sorted(per_lang_refs.keys()):
        refs = per_lang_refs[lang]
        hyps = per_lang_hyps[lang]
        sample_stats = per_lang_sample_metrics[lang]
        
        # Aggregate WER/CER
        lang_wer_measures = compute_wer_measures(refs, hyps)
        lang_cer, _, _, _, _ = compute_cer_measures(refs, hyps)
        
        # Bootstrap CI
        lang_wer_mean, lang_wer_std, lang_wer_ci_l, lang_wer_ci_h = bootstrap_wer_ci(refs, hyps)
        lang_cer_mean, lang_cer_std, lang_cer_ci_l, lang_cer_ci_h = bootstrap_cer_ci(refs, hyps)
        
        # F1 bootstrap
        cf_m, cf_s, cf_ci_l, cf_ci_h = bootstrap_metric_ci(sample_stats['char_f1'])
        wf_m, wf_s, wf_ci_l, wf_ci_h = bootstrap_metric_ci(sample_stats['word_f1'])
        lat_m, lat_s, lat_ci_l, lat_ci_h = bootstrap_metric_ci(sample_stats['latency'])
        
        print(f"\n--- {lang} (n={len(refs)}) ---")
        print(f"WER: {lang_wer_measures['wer']:.4f} (aggregate)")
        print(f"WER Bootstrap: {lang_wer_mean:.4f} ± {lang_wer_std:.4f} (95% CI: [{lang_wer_ci_l:.4f}, {lang_wer_ci_h:.4f}])")
        print(f"CER: {lang_cer:.4f} (aggregate)")
        print(f"CER Bootstrap: {lang_cer_mean:.4f} ± {lang_cer_std:.4f} (95% CI: [{lang_cer_ci_l:.4f}, {lang_cer_ci_h:.4f}])")
        print(f"Char F1: {cf_m:.4f} ± {cf_s:.4f} (95% CI: [{cf_ci_l:.4f}, {cf_ci_h:.4f}])")
        print(f"Word F1: {wf_m:.4f} ± {wf_s:.4f} (95% CI: [{wf_ci_l:.4f}, {wf_ci_h:.4f}])")
        
        per_lang_rows.append({
            "language": lang,
            "num_samples": len(refs),
            "aggregate_wer": lang_wer_measures['wer'],
            "wer_mean": lang_wer_mean,
            "wer_std": lang_wer_std,
            "wer_ci_low": lang_wer_ci_l,
            "wer_ci_high": lang_wer_ci_h,
            "aggregate_cer": lang_cer,
            "cer_mean": lang_cer_mean,
            "cer_std": lang_cer_std,
            "cer_ci_low": lang_cer_ci_l,
            "cer_ci_high": lang_cer_ci_h,
            "char_f1_mean": cf_m,
            "char_f1_std": cf_s,
            "char_f1_ci_low": cf_ci_l,
            "char_f1_ci_high": cf_ci_h,
            "word_f1_mean": wf_m,
            "word_f1_std": wf_s,
            "word_f1_ci_low": wf_ci_l,
            "word_f1_ci_high": wf_ci_h,
            "latency_mean_sec": lat_m,
            "latency_std_sec": lat_s,
        })
    
    # =========================
    # SAVE FILES
    # =========================
    
    df_results = pd.DataFrame(results_rows)
    results_path = os.path.join(OUTPUT_DIR, "results_detailed.csv")
    df_results.to_csv(results_path, index=False, encoding="utf-8")
    
    df_lang = pd.DataFrame(per_lang_rows)
    lang_path = os.path.join(OUTPUT_DIR, "per_language_results.csv")
    df_lang.to_csv(lang_path, index=False, encoding="utf-8")
    
    if error_analysis_rows:
        df_errors = pd.DataFrame(error_analysis_rows)
        error_path = os.path.join(OUTPUT_DIR, "error_analysis.csv")
        df_errors.to_csv(error_path, index=False, encoding="utf-8")
        print(f"\n⚠️  {len(error_analysis_rows)} samples flagged for error analysis")
    
    # Summary
    summary_path = os.path.join(OUTPUT_DIR, "inference_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Qwen2-Audio LoRA Inference Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_name_print}\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Bootstrap samples: {N_BOOTSTRAP}\n\n")
        
        f.write("Overall Metrics (Aggregate):\n")
        f.write("-" * 40 + "\n")
        f.write(f"WER: {overall_wer_measures['wer']:.4f}\n")
        f.write(f"WER Bootstrap: {wer_mean:.4f} ± {wer_std:.4f} (95% CI: [{wer_ci_low:.4f}, {wer_ci_high:.4f}])\n")
        f.write(f"CER: {overall_cer:.4f}\n")
        f.write(f"CER Bootstrap: {cer_mean:.4f} ± {cer_std:.4f} (95% CI: [{cer_ci_low:.4f}, {cer_ci_high:.4f}])\n")
        f.write(f"Char F1: {char_f1_mean:.4f} ± {char_f1_std:.4f} (95% CI: [{char_f1_ci_low:.4f}, {char_f1_ci_high:.4f}])\n")
        f.write(f"Word F1: {word_f1_mean:.4f} ± {word_f1_std:.4f} (95% CI: [{word_f1_ci_low:.4f}, {word_f1_ci_high:.4f}])\n\n")
        
        f.write("Per-Language Results:\n")
        f.write("-" * 40 + "\n")
        for row in per_lang_rows:
            f.write(f"\n{row['language']} (n={row['num_samples']}):\n")
            f.write(f"  WER: {row['aggregate_wer']:.4f} (aggregate)\n")
            f.write(f"  CER: {row['aggregate_cer']:.4f} (aggregate)\n")
            f.write(f"  Char F1: {row['char_f1_mean']:.4f} ± {row['char_f1_std']:.4f}\n")
            f.write(f"  Word F1: {row['word_f1_mean']:.4f} ± {row['word_f1_std']:.4f}\n")
        
        f.write(f"\nInference Time:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total: {total_inference_time/3600:.2f} hours\n")
        f.write(f"Per-sample latency: {latency_mean:.4f}s ± {latency_std:.4f}s\n")
        f.write(f"Throughput: {len(samples)/total_inference_time:.2f} samples/sec\n")
    
    # =========================================================
    # STANDARDIZED CI METRICS CSV + JSON + VISUALIZATION PLOTS
    # (ADDED — everything above this line is unchanged)
    # =========================================================

    # Build ci_metrics dict for plots and standardized export
    ci_metrics = {"overall": {}, "per_language": {}}

    ci_metrics["overall"] = {
        "wer": wer_mean,
        "wer_ci_lower": wer_ci_low,
        "wer_ci_upper": wer_ci_high,
        "cer": cer_mean,
        "cer_ci_lower": cer_ci_low,
        "cer_ci_upper": cer_ci_high,
        "total_samples": len(all_references),
    }

    for row in per_lang_rows:
        ci_metrics["per_language"][row["language"]] = {
            "wer": row["wer_mean"],
            "wer_ci_lower": row["wer_ci_low"],
            "wer_ci_upper": row["wer_ci_high"],
            "cer": row["cer_mean"],
            "cer_ci_lower": row["cer_ci_low"],
            "cer_ci_upper": row["cer_ci_high"],
            "total_samples": row["num_samples"],
        }

    # Save standardized CI metrics CSV
    ci_csv_path = os.path.join(OUTPUT_DIR, "metrics_summary_with_ci.csv")
    with open(ci_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "language", "wer", "wer_ci_lower", "wer_ci_upper",
            "cer", "cer_ci_lower", "cer_ci_upper", "num_samples",
        ])
        ov = ci_metrics["overall"]
        w.writerow([
            "OVERALL",
            f"{ov['wer']:.4f}", f"{ov['wer_ci_lower']:.4f}", f"{ov['wer_ci_upper']:.4f}",
            f"{ov['cer']:.4f}", f"{ov['cer_ci_lower']:.4f}", f"{ov['cer_ci_upper']:.4f}",
            ov["total_samples"],
        ])
        for lang_name in sorted(ci_metrics["per_language"].keys()):
            lm = ci_metrics["per_language"][lang_name]
            w.writerow([
                lang_name,
                f"{lm['wer']:.4f}", f"{lm['wer_ci_lower']:.4f}", f"{lm['wer_ci_upper']:.4f}",
                f"{lm['cer']:.4f}", f"{lm['cer_ci_lower']:.4f}", f"{lm['cer_ci_upper']:.4f}",
                lm["total_samples"],
            ])

    # Save CI metrics as JSON
    ci_json_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(ci_json_path, "w", encoding="utf-8") as f:
        json.dump(ci_metrics, f, indent=2)

    # Generate visualization plots
    print(f"\n📈 Generating visualization plots...")
    generate_plots(results_rows, ci_metrics, OUTPUT_DIR)

    # =========================================================

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print(f"  📄 Detailed results: {results_path}")
    print(f"  📄 Per-language summary: {lang_path}")
    if error_analysis_rows:
        print(f"  📄 Error analysis: {os.path.join(OUTPUT_DIR, 'error_analysis.csv')}")
    print(f"  📄 Summary: {summary_path}")
    print(f"  📄 CI Metrics CSV: {ci_csv_path}")
    print(f"  📄 CI Metrics JSON: {ci_json_path}")
    print(f"  📈 Inference metrics: {os.path.join(OUTPUT_DIR, 'inference_metrics.png')}")
    print(f"  📈 WER vs CER scatter: {os.path.join(OUTPUT_DIR, 'wer_vs_cer_scatter.png')}")
    print(f"  📈 Confidence intervals: {os.path.join(OUTPUT_DIR, 'confidence_intervals.png')}")
    
    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()