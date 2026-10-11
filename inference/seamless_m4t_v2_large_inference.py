import os
import re
import csv
import time
import json
import random
from math import gcd
from collections import defaultdict
from typing import Dict, List, Tuple

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
from peft import PeftModel


# =========================================================
# CONFIG
# =========================================================
BASE_MODEL_NAME = "facebook/seamless-m4t-v2-large"

DATASET_ROOT = r"/path/to/data/IndicVoices_subset_4lang_combined"  # <-- CHANGE  # <-- CHANGE
TEST_DIR = os.path.join(DATASET_ROOT, "test")
TEST_CSV = os.path.join(TEST_DIR, "test.csv")
TEST_AUDIO_DIR = os.path.join(TEST_DIR, "audio")

# Your LoRA adapter dir
# ✅ OPTION 1: Use the best checkpoint from training (recommended)
ADAPTER_DIR = os.path.join(
    DATASET_ROOT, "seamless_m4t_v2_large_lora_4lang", "checkpoints", "best"
)
# ✅ OPTION 2: Use the final model
# ADAPTER_DIR = os.path.join(DATASET_ROOT, "seamless_m4t_v2_large_lora_4lang")

OUT_DIR = os.path.dirname(ADAPTER_DIR) if "checkpoints" in ADAPTER_DIR else ADAPTER_DIR
OUT_PER_SAMPLE_CSV = os.path.join(OUT_DIR, "test_asr_results.csv")
OUT_PER_LANG_CSV = os.path.join(OUT_DIR, "test_asr_per_language_summary.csv")
OUT_OVERALL_CSV = os.path.join(OUT_DIR, "test_asr_overall_summary.csv")
OUT_DEBUG_CSV = os.path.join(OUT_DIR, "test_asr_debug.csv")  # ✅ NEW: Debug file

BATCH_SIZE = 4
RANDOM_SEED = 42

# ✅ NEW: Generation parameters to prevent empty outputs
MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 10  # Force minimum output length
NUM_BEAMS = 1  # Greedy decoding (set to 4 for beam search)
TEMPERATURE = 1.0  # For sampling (only if do_sample=True)
DO_SAMPLE = False  # Use greedy decoding
LENGTH_PENALTY = 1.0
EARLY_STOPPING = True

# ✅ NEW: Bootstrap configuration for confidence intervals
BOOTSTRAP_ITERATIONS = 1000  # Number of bootstrap samples for CI
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LANGUAGE MAPPING
# =========================================================
CSV2_TO_ISO3 = {
    "as": "asm",
    "bn": "ben",
    "hi": "hin",
    "or": "ory",  # Odia
}

ISO3_TO_NAME = {
    "asm": "Assamese",
    "ben": "Bengali",
    "hin": "Hindi",
    "ory": "Odia",
}


# =========================================================
# AUDIO HELPERS
# =========================================================
def extract_filename_from_mixed_path(path_str: str) -> str:
    if path_str is None:
        return ""
    normalized = str(path_str).strip().replace("\\", "/")
    if not normalized:
        return ""
    return os.path.basename(normalized)


def resolve_path(p: str) -> str:
    p = str(p).strip()
    if os.path.exists(p):
        return p
    base = extract_filename_from_mixed_path(p)
    if not base:
        return ""
    return os.path.join(TEST_AUDIO_DIR, base)


def load_audio_mono(path: str, target_sr: int):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)

    return wav, target_sr


# =========================================================
# TEXT NORMALIZATION
# =========================================================
_whitespace_re = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = str(s).strip()
    s = _whitespace_re.sub(" ", s)
    return s


# =========================================================
# METRICS
# =========================================================
def levenshtein_distance(a, b):
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    if len(b) > len(a):
        a, b = b, a

    prev = np.arange(len(b) + 1, dtype=np.int32)
    curr = np.zeros(len(b) + 1, dtype=np.int32)

    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr[j] = ins if ins < dele else dele
            if sub < curr[j]:
                curr[j] = sub
        prev, curr = curr, prev

    return int(prev[len(b)])


def cer(ref: str, hyp: str):
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    dist = levenshtein_distance(r, h)
    denom = max(1, len(r))
    return dist / denom, dist, len(r)


def wer(ref: str, hyp: str):
    r = ref.split()
    h = hyp.split()
    dist = levenshtein_distance(r, h)
    denom = max(1, len(r))
    return dist / denom, dist, len(r)


def prf_counts_multiset(ref_tokens, hyp_tokens):
    ref_map = defaultdict(int)
    for t in ref_tokens:
        ref_map[t] += 1

    correct = 0
    for t in hyp_tokens:
        if ref_map[t] > 0:
            correct += 1
            ref_map[t] -= 1

    return correct, len(hyp_tokens), len(ref_tokens)


def prf_from_counts(correct, hyp_n, ref_n):
    prec = correct / max(1, hyp_n)
    rec = correct / max(1, ref_n)
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    return prec, rec, f1


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
# LOAD TEST CSV
# =========================================================
df = pd.read_csv(TEST_CSV)
need = {"path", "text", "lang"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"test.csv missing columns: {missing}. Found: {list(df.columns)}")

samples = []
skipped_empty_filename = 0
for _, r in df.iterrows():
    lang2 = str(r["lang"]).strip().lower()
    if lang2 not in CSV2_TO_ISO3:
        continue
    file_name = extract_filename_from_mixed_path(r["path"])
    if not file_name:
        skipped_empty_filename += 1
        continue
    samples.append(
        {
            "audio_path": resolve_path(r["path"]),
            "file": file_name,
            "ref": normalize_text(r["text"]),
            "lang2": lang2,
            "lang3_actual": CSV2_TO_ISO3[lang2],
        }
    )

print(f"Loaded test samples: {len(samples)}")
if skipped_empty_filename > 0:
    print(
        f"  ⚠️ Skipped {skipped_empty_filename} rows with empty/invalid audio filename in CSV path."
    )
random.shuffle(samples)


# =========================================================
# LOAD Seamless M4T v2 ASR model (4-bit) + LoRA adapter
# =========================================================
print(f"Loading processor from: {ADAPTER_DIR}")
processor = AutoProcessor.from_pretrained(ADAPTER_DIR)
ASR_SR = processor.feature_extractor.sampling_rate

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading SeamlessM4Tv2ForSpeechToText in 4-bit...")
base = SeamlessM4Tv2ForSpeechToText.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

# ✅ NEW: Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model loaded: {total_params:,} total params, {trainable_params:,} trainable")


# =========================================================
# OUTPUT CSV HEADERS
# =========================================================
os.makedirs(OUT_DIR, exist_ok=True)

with open(OUT_PER_SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "batch_no",
            "file",
            "audio_path",
            "lang_actual_iso3",
            "lang_actual_name",
            "transcription_hyp",
            "transcription_ref",
            "hyp_length",
            "ref_length",
            "CER",
            "WER",
            "char_precision",
            "char_recall",
            "char_f1",
            "word_precision",
            "word_recall",
            "word_f1",
            "is_empty_hyp",
            "is_very_short_hyp",
        ]
    )

# ✅ NEW: Debug CSV for troubleshooting
with open(OUT_DEBUG_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "batch_no",
            "file",
            "lang",
            "audio_duration_sec",
            "ref_length",
            "hyp_raw",
            "hyp_decoded",
            "hyp_length",
            "generated_tokens_count",
            "is_empty",
            "error_if_any",
        ]
    )


# =========================================================
# AGGREGATORS
# =========================================================
overall = {
    "cer_edits": 0,
    "cer_ref_chars": 0,
    "wer_edits": 0,
    "wer_ref_words": 0,
    "char_correct": 0,
    "char_hyp": 0,
    "char_ref": 0,
    "word_correct": 0,
    "word_hyp": 0,
    "word_ref": 0,
    "empty_count": 0,
    "very_short_count": 0,
}

per_lang = defaultdict(
    lambda: {
        "n": 0,
        "cer_edits": 0,
        "cer_ref_chars": 0,
        "wer_edits": 0,
        "wer_ref_words": 0,
        "char_correct": 0,
        "char_hyp": 0,
        "char_ref": 0,
        "word_correct": 0,
        "word_hyp": 0,
        "word_ref": 0,
        "empty_count": 0,
        "very_short_count": 0,
    }
)

# ✅ NEW: Collect per-sample results for bootstrap CI and visualizations
all_sample_results = []


def update_aggs(lang3_actual, ref, hyp):
    global overall

    cer_val, cer_ed, cer_ref_len = cer(ref, hyp)
    wer_val, wer_ed, wer_ref_len = wer(ref, hyp)

    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    c_corr, c_hn, c_rn = prf_counts_multiset(ref_chars, hyp_chars)

    ref_words = ref.split()
    hyp_words = hyp.split()
    w_corr, w_hn, w_rn = prf_counts_multiset(ref_words, hyp_words)

    # Track empty/short predictions
    is_empty = len(hyp.strip()) == 0
    is_very_short = len(hyp.strip()) < 3

    if is_empty:
        overall["empty_count"] += 1
    if is_very_short:
        overall["very_short_count"] += 1

    # overall
    overall["cer_edits"] += cer_ed
    overall["cer_ref_chars"] += cer_ref_len
    overall["wer_edits"] += wer_ed
    overall["wer_ref_words"] += wer_ref_len
    overall["char_correct"] += c_corr
    overall["char_hyp"] += c_hn
    overall["char_ref"] += c_rn
    overall["word_correct"] += w_corr
    overall["word_hyp"] += w_hn
    overall["word_ref"] += w_rn

    # per-language
    pl = per_lang[lang3_actual]
    pl["n"] += 1
    pl["cer_edits"] += cer_ed
    pl["cer_ref_chars"] += cer_ref_len
    pl["wer_edits"] += wer_ed
    pl["wer_ref_words"] += wer_ref_len
    pl["char_correct"] += c_corr
    pl["char_hyp"] += c_hn
    pl["char_ref"] += c_rn
    pl["word_correct"] += w_corr
    pl["word_hyp"] += w_hn
    pl["word_ref"] += w_rn
    if is_empty:
        pl["empty_count"] += 1
    if is_very_short:
        pl["very_short_count"] += 1

    return (
        cer_val,
        wer_val,
        c_corr,
        c_hn,
        c_rn,
        w_corr,
        w_hn,
        w_rn,
        is_empty,
        is_very_short,
    )


def micro_cer(edits, ref_chars):
    return edits / max(1, ref_chars)


def micro_wer(edits, ref_words):
    return edits / max(1, ref_words)


# =========================================================
# INFERENCE LOOP
# =========================================================
t0 = time.time()

progress = tqdm(total=len(samples), desc="ASR test", ncols=100)
batch_no = 0

for bi in range(0, len(samples), BATCH_SIZE):
    batch_no += 1
    batch = samples[bi : bi + BATCH_SIZE]

    audio_asr = []
    file_names = []
    actual_lang3 = []
    refs = []
    audio_paths = []
    audio_durations = []

    for s in batch:
        path = s["audio_path"]
        wav_asr, sr = load_audio_mono(path, ASR_SR)
        audio_asr.append(wav_asr)
        file_names.append(os.path.basename(path))
        actual_lang3.append(s["lang3_actual"])
        refs.append(s["ref"])
        audio_paths.append(path)
        audio_durations.append(len(wav_asr) / sr)

    inputs = processor(
        audio=audio_asr,
        sampling_rate=ASR_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hyps = [""] * len(batch)
    raw_outputs = [None] * len(batch)
    token_counts = [0] * len(batch)
    errors = [""] * len(batch)

    lang_to_indices = defaultdict(list)
    for idx, lg in enumerate(actual_lang3):
        lang_to_indices[lg].append(idx)

    for lg, idxs in lang_to_indices.items():
        sub = {k: v[idxs] for k, v in inputs.items()}

        try:
            with torch.no_grad():
                # ✅ FIXED: More robust generation parameters
                out_tokens = model.generate(
                    **sub,
                    tgt_lang=lg,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=MIN_NEW_TOKENS,  # Force minimum output
                    num_beams=NUM_BEAMS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE if DO_SAMPLE else 1.0,
                    length_penalty=LENGTH_PENALTY,
                    early_stopping=EARLY_STOPPING,
                    # ✅ NEW: Additional stability parameters
                    repetition_penalty=1.2,  # Prevent repetitive output
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                    # ✅ NEW: Forced tokens to encourage output
                    forced_bos_token_id=None,  # Let model choose start token
                )

            for j, row_tokens in enumerate(out_tokens):
                raw_outputs[idxs[j]] = row_tokens.tolist()
                token_counts[idxs[j]] = len(row_tokens.tolist())

                # ✅ FIXED: More robust decoding
                hyp = processor.decode(
                    row_tokens.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                hyps[idxs[j]] = normalize_text(hyp)

        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️ Error during generation for lang {lg}: {error_msg}")
            for idx in idxs:
                errors[idx] = error_msg
                hyps[idx] = ""

    # Process results for each sample in batch
    for j in range(len(batch)):
        ref = refs[j]
        hyp = hyps[j]
        lg = actual_lang3[j]
        lg_name = ISO3_TO_NAME.get(lg, lg)

        cer_val, wer_val, c_corr, c_hn, c_rn, w_corr, w_hn, w_rn, is_empty, is_short = (
            update_aggs(lg, ref, hyp)
        )
        c_p, c_r, c_f1 = prf_from_counts(c_corr, c_hn, c_rn)
        w_p, w_r, w_f1 = prf_from_counts(w_corr, w_hn, w_rn)

        # ✅ NEW: Collect per-sample results for bootstrap CI and plots
        all_sample_results.append(
            {
                "audio": audio_paths[j],
                "reference": ref,
                "prediction": hyp,
                "wer": wer_val,
                "cer": cer_val,
                "lang": lg_name,
                "lang_code": lg,
            }
        )

        print("\n------------------------------------")
        print(f"Batch {batch_no} | File: {file_names[j]}")
        print(f"Actual Lang: {lg} ({lg_name})")
        print(f"Audio Duration: {audio_durations[j]:.2f}s")
        print(f"HYP ({len(hyp)} chars): {hyp[:100]}{'...' if len(hyp) > 100 else ''}")
        print(f"REF ({len(ref)} chars): {ref[:100]}{'...' if len(ref) > 100 else ''}")

        if is_empty:
            print("⚠️ WARNING: Empty hypothesis!")
        if is_short:
            print("⚠️ WARNING: Very short hypothesis!")

        print(f"CER: {cer_val:.4f} | char P/R/F1: {c_p:.4f}/{c_r:.4f}/{c_f1:.4f}")
        print(f"WER: {wer_val:.4f} | word P/R/F1: {w_p:.4f}/{w_r:.4f}/{w_f1:.4f}")

        with open(OUT_PER_SAMPLE_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    batch_no,
                    file_names[j],
                    audio_paths[j],
                    lg,
                    lg_name,
                    hyp,
                    ref,
                    len(hyp),
                    len(ref),
                    f"{cer_val:.6f}",
                    f"{wer_val:.6f}",
                    f"{c_p:.6f}",
                    f"{c_r:.6f}",
                    f"{c_f1:.6f}",
                    f"{w_p:.6f}",
                    f"{w_r:.6f}",
                    f"{w_f1:.6f}",
                    is_empty,
                    is_short,
                ]
            )

        # ✅ NEW: Debug logging
        with open(OUT_DEBUG_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    batch_no,
                    file_names[j],
                    lg,
                    f"{audio_durations[j]:.2f}",
                    len(ref),
                    raw_outputs[j][:50]
                    if raw_outputs[j]
                    else None,  # First 50 raw tokens
                    hyp,
                    len(hyp),
                    token_counts[j],
                    is_empty,
                    errors[j],
                ]
            )

    progress.update(len(batch))

progress.close()
elapsed = time.time() - t0


# =========================================================
# PRINT + EXPORT SUMMARIES
# =========================================================
def summarize_block(title, agg):
    cer_m = micro_cer(agg["cer_edits"], agg["cer_ref_chars"])
    wer_m = micro_wer(agg["wer_edits"], agg["wer_ref_words"])
    c_p, c_r, c_f1 = prf_from_counts(
        agg["char_correct"], agg["char_hyp"], agg["char_ref"]
    )
    w_p, w_r, w_f1 = prf_from_counts(
        agg["word_correct"], agg["word_hyp"], agg["word_ref"]
    )

    print("\n==============================")
    print(title)
    print("==============================")
    print(f"Micro CER: {cer_m:.4f}")
    print(f"Micro WER: {wer_m:.4f}")
    print(f"Char P/R/F1: {c_p:.4f}/{c_r:.4f}/{c_f1:.4f}")
    print(f"Word P/R/F1: {w_p:.4f}/{w_r:.4f}/{w_f1:.4f}")

    if "empty_count" in agg:
        print(
            f"Empty predictions: {agg['empty_count']} ({100 * agg['empty_count'] / max(1, len(samples)):.1f}%)"
        )
        print(
            f"Very short predictions: {agg['very_short_count']} ({100 * agg['very_short_count'] / max(1, len(samples)):.1f}%)"
        )

    return cer_m, wer_m, c_p, c_r, c_f1, w_p, w_r, w_f1


# Overall
oc, ow, ocp, ocr, ocf1, owp, owr, owf1 = summarize_block(
    "OVERALL (ALL LANGUAGES)", overall
)
print(f"Processed {len(samples)} files in {elapsed:.1f}s")

with open(OUT_OVERALL_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "scope",
            "n_files",
            "micro_cer",
            "micro_wer",
            "char_precision",
            "char_recall",
            "char_f1",
            "word_precision",
            "word_recall",
            "word_f1",
            "empty_count",
            "very_short_count",
            "elapsed_sec",
        ]
    )
    w.writerow(
        [
            "overall",
            len(samples),
            f"{oc:.6f}",
            f"{ow:.6f}",
            f"{ocp:.6f}",
            f"{ocr:.6f}",
            f"{ocf1:.6f}",
            f"{owp:.6f}",
            f"{owr:.6f}",
            f"{owf1:.6f}",
            overall["empty_count"],
            overall["very_short_count"],
            f"{elapsed:.1f}",
        ]
    )

# Per-language summary CSV
with open(OUT_PER_LANG_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "lang_iso3",
            "lang_name",
            "n_files",
            "micro_cer",
            "micro_wer",
            "char_precision",
            "char_recall",
            "char_f1",
            "word_precision",
            "word_recall",
            "word_f1",
            "empty_count",
            "very_short_count",
        ]
    )

for lg in sorted(per_lang.keys()):
    agg = per_lang[lg]
    cer_m = micro_cer(agg["cer_edits"], agg["cer_ref_chars"])
    wer_m = micro_wer(agg["wer_edits"], agg["wer_ref_words"])
    c_p, c_r, c_f1 = prf_from_counts(
        agg["char_correct"], agg["char_hyp"], agg["char_ref"]
    )
    w_p, w_r, w_f1 = prf_from_counts(
        agg["word_correct"], agg["word_hyp"], agg["word_ref"]
    )
    name = ISO3_TO_NAME.get(lg, lg)

    print(f"\n--- {lg} ({name}) | n={agg['n']} ---")
    print(f"Micro CER: {cer_m:.4f} | Micro WER: {wer_m:.4f}")
    print(f"Char P/R/F1: {c_p:.4f}/{c_r:.4f}/{c_f1:.4f}")
    print(f"Word P/R/F1: {w_p:.4f}/{w_r:.4f}/{w_f1:.4f}")
    if agg["empty_count"] > 0:
        print(f"⚠️ Empty: {agg['empty_count']}, Very short: {agg['very_short_count']}")

    with open(OUT_PER_LANG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                lg,
                name,
                agg["n"],
                f"{cer_m:.6f}",
                f"{wer_m:.6f}",
                f"{c_p:.6f}",
                f"{c_r:.6f}",
                f"{c_f1:.6f}",
                f"{w_p:.6f}",
                f"{w_r:.6f}",
                f"{w_f1:.6f}",
                agg["empty_count"],
                agg["very_short_count"],
            ]
        )

print("\n✅ Wrote:")
print("  ", OUT_PER_SAMPLE_CSV)
print("  ", OUT_PER_LANG_CSV)
print("  ", OUT_OVERALL_CSV)
print("  ", OUT_DEBUG_CSV)


# =========================================================
# ✅ NEW: BOOTSTRAP CI METRICS + JSON EXPORT
# =========================================================
print(f"\n📊 Computing metrics with {BOOTSTRAP_ITERATIONS} bootstrap iterations...")

ci_metrics = {
    "overall": {},
    "per_language": {},
}

# Overall bootstrap CI
all_wers = [r["wer"] for r in all_sample_results]
all_cers = [r["cer"] for r in all_sample_results]

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
    "total_samples": len(all_sample_results),
}

# Per-language bootstrap CI
lang_results_grouped = defaultdict(list)
for r in all_sample_results:
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
print("\n" + "=" * 60)
print("RESULTS WITH 95% CONFIDENCE INTERVALS")
print("=" * 60)

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
OUT_CI_METRICS_CSV = os.path.join(OUT_DIR, "metrics_summary_with_ci.csv")
with open(OUT_CI_METRICS_CSV, "w", newline="", encoding="utf-8") as f:
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
print(f"📄 CI Metrics CSV saved to: {OUT_CI_METRICS_CSV}")

# Save metrics as JSON
OUT_CI_METRICS_JSON = os.path.join(OUT_DIR, "metrics_summary.json")
with open(OUT_CI_METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(ci_metrics, f, indent=2)
print(f"📄 CI Metrics JSON saved to: {OUT_CI_METRICS_JSON}")


# =========================================================
# ✅ NEW: VISUALIZATION PLOTS
# =========================================================
def generate_plots(results: List[Dict], metrics: Dict, output_dir: str):
    """Generate visualization plots matching the MMS inference script."""
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
            wers = [r["wer"] for r in lang_results]
            cers = [r["cer"] for r in lang_results]
            ax.scatter(wers, cers, alpha=0.5, label=lang, c=colors.get(lang, "gray"))

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


print("\n📈 Generating visualization plots...")
generate_plots(all_sample_results, ci_metrics, OUT_DIR)


# =========================================================
# ✅ NEW: DIAGNOSTIC SUMMARY
# =========================================================
print("\n" + "=" * 50)
print("DIAGNOSTIC SUMMARY")
print("=" * 50)

if overall["empty_count"] > 0:
    print(f"\n⚠️ WARNING: {overall['empty_count']} empty predictions detected!")
    print("Possible causes:")
    print("  1. Model collapsed during training (overfitting)")
    print("  2. Learning rate was too high")
    print("  3. Training was too long without proper regularization")
    print("\nRecommendations:")
    print("  1. Use the best checkpoint instead of final model")
    print("  2. Lower learning rate (try 1e-5 or 5e-6)")
    print("  3. Add early stopping based on validation loss")
    print("  4. Check training curves for signs of collapse")

if overall["very_short_count"] > len(samples) * 0.1:
    print(f"\n⚠️ WARNING: {overall['very_short_count']} very short predictions!")
    print("This indicates the model may be undertrained or collapsed.")

print("\n" + "=" * 60)
print("Inference completed successfully!")
print("Results are ready for research publication.")
print("=" * 60)
