"""
Phi-4-Multimodal-Instruct LoRA Inference for Indic Languages

This script performs ASR inference using a fine-tuned Phi-4-multimodal-instruct model
on 4 Indic languages: Assamese, Bengali, Hindi, and Odia.

Key Features:
- Loads fine-tuned model from local checkpoint
- Batch processing with batch size 2 (random shuffle)
- Complete audio clip usage (no chunking)
- CER, WER, char-level and word-level precision/recall/F1
- Per-language and overall metrics
- CSV export of all results
- tqdm progress bar

Dataset Structure:
    IndicVoices_subset_4lang_8hTrain_1hVal_2hTest_mini/
    |__test/
    |   |__audio/
    |   |__test.csv
"""

import os
import gc
import random
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)

# =========================
# COMPATIBILITY SHIMS for newer transformers versions
# =========================
# The Phi-4-multimodal model's custom modeling code (modeling_phi4mm.py) calls
# DynamicCache methods that were removed in transformers >= 4.49+:
#   - get_usable_length() removed in PR #39106 (replaced by get_seq_length())
#   - get_max_length() removed in transformers 4.49+
#   - SlidingWindowCache removed/restructured in transformers 5.x
# We monkey-patch these back so the saved model code works.

import transformers.cache_utils as _cache_utils
from transformers import DynamicCache as _DynCache

# Patch 1: Restore get_usable_length if missing
if not hasattr(_DynCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Compatibility shim: get_usable_length was removed, use get_seq_length."""
        if hasattr(self, "get_seq_length"):
            return self.get_seq_length(layer_idx)
        # Fallback: check key_cache directly
        if hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
            return self.key_cache[layer_idx].shape[-2]
        return 0

    _DynCache.get_usable_length = _get_usable_length
    print("[COMPAT] Patched DynamicCache.get_usable_length")

# Patch 2: Restore get_max_length if missing
if not hasattr(_DynCache, "get_max_length"):
    def _get_max_length(self) -> None:
        """Compatibility shim: DynamicCache has no max length."""
        return None

    _DynCache.get_max_length = _get_max_length
    print("[COMPAT] Patched DynamicCache.get_max_length")

# Patch 3: Restore SlidingWindowCache if missing
if not hasattr(_cache_utils, "SlidingWindowCache"):
    class _SlidingWindowCache(_DynCache):
        """Fallback stub for transformers versions that removed SlidingWindowCache."""
        def __init__(self, *args, **kwargs):
            kwargs.pop("sliding_window", None)
            super().__init__(*args, **kwargs)

    _cache_utils.SlidingWindowCache = _SlidingWindowCache
    import transformers as _tf
    _tf.SlidingWindowCache = _SlidingWindowCache
    print("[COMPAT] Patched SlidingWindowCache")

from jiwer import wer as compute_wer_jiwer, cer as compute_cer_jiwer

# =========================
# CONFIGURATION
# =========================

MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"

# Path to the fine-tuned model checkpoint saved by Trainer
# UPDATE THIS to your actual saved model path
FINETUNED_MODEL_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/phi4_multimodal_lora_indic"

# Test data path
TEST_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/test"

BATCH_SIZE = 2
RANDOM_SEED = 42
MAX_NEW_TOKENS = 512

# Output files (matching qwen2_audio_lora_inference.py style)
OUTPUT_TSV = "results.tsv"
PER_LANG_CSV = "per_language_results.csv"
OVERALL_CSV = "overall_results.csv"
PER_LANG_PNG = "per_language_summary.png"
OVERALL_PNG = "overall_summary.png"

MODEL_NAME_PRINT = f"{MODEL_NAME} + LoRA({FINETUNED_MODEL_DIR})"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disable CPU multithreading
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =========================
# LANGUAGE CODE MAPPING
# =========================

LANG_CODE_TO_NAME = {
    "as": "Assamese",
    "bn": "Bengali",
    "hi": "Hindi",
    "or": "Odia",
}

# Language-specific prompts (must match training)
LANG_PROMPTS = {
    "Assamese": "Transcribe the audio clip into Assamese text.",
    "Bengali": "Transcribe the audio clip into Bengali text.",
    "Hindi": "Transcribe the audio clip into Hindi text.",
    "Odia": "Transcribe the audio clip into Odia text.",
}


def get_language_name(lang_code):
    """Convert language code to full name."""
    code = str(lang_code).lower().strip()
    return LANG_CODE_TO_NAME.get(code, code.capitalize())


# =========================
# AUDIO LOADING
# =========================

def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio using soundfile and resample if needed. Uses full clip."""
    wav, sr = sf.read(path, dtype="float32")

    # Stereo to mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # Resample only if needed
    if sr != target_sr:
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)

    return wav, target_sr


# =========================
# METRIC FUNCTIONS
# =========================

def char_precision_recall_f1(pred: str, ref: str):
    """Compute character-level precision, recall, and F1."""
    pred_chars = list(pred)
    ref_chars = list(ref)

    if len(pred_chars) == 0 and len(ref_chars) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_chars) == 0:
        return 0.0, 0.0, 0.0
    if len(ref_chars) == 0:
        return 0.0, 0.0, 0.0

    pred_c = Counter(pred_chars)
    ref_c = Counter(ref_chars)

    tp = sum((pred_c & ref_c).values())
    fp = sum((pred_c - ref_c).values())
    fn = sum((ref_c - pred_c).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def word_precision_recall_f1(pred: str, ref: str):
    """Compute word-level precision, recall, and F1."""
    pred_words = pred.split()
    ref_words = ref.split()

    if len(pred_words) == 0 and len(ref_words) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_words) == 0:
        return 0.0, 0.0, 0.0
    if len(ref_words) == 0:
        return 0.0, 0.0, 0.0

    pred_c = Counter(pred_words)
    ref_c = Counter(ref_words)

    tp = sum((pred_c & ref_c).values())
    fp = sum((pred_c - ref_c).values())
    fn = sum((ref_c - pred_c).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def safe_cer(ref: str, pred: str) -> float:
    """Compute CER safely, handling empty references."""
    if len(ref.strip()) == 0 and len(pred.strip()) == 0:
        return 0.0
    if len(ref.strip()) == 0:
        return 1.0
    return compute_cer_jiwer(ref, pred)


def safe_wer(ref: str, pred: str) -> float:
    """Compute WER safely, handling empty references."""
    if len(ref.strip()) == 0 and len(pred.strip()) == 0:
        return 0.0
    if len(ref.strip()) == 0:
        return 1.0
    return compute_wer_jiwer(ref, pred)


# =========================
# LOAD SAMPLES
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
        raw_path = str(row["path"]).strip()
        normalized_path = raw_path.replace("\\", "/")

        path_candidates = []
        path_candidates.append(raw_path)
        path_candidates.append(normalized_path)

        filename = os.path.basename(normalized_path)
        if filename:
            path_candidates.append(os.path.join(audio_dir, filename))
            path_candidates.append(os.path.join(data_dir, filename))

        parts = normalized_path.split("/")
        if "audio" in parts:
            audio_idx = len(parts) - 1 - parts[::-1].index("audio")
            rel_after_audio = "/".join(parts[audio_idx + 1:]).strip()
            if rel_after_audio:
                path_candidates.append(os.path.join(audio_dir, rel_after_audio))

        # Deduplicate
        deduped = []
        seen = set()
        for c in path_candidates:
            key = c.strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(key)

        audio_path = None
        for c in deduped:
            if os.path.exists(c):
                audio_path = c
                break

        if audio_path is None:
            skipped_missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(raw_path)
            continue

        lang_code = str(row["lang"])
        lang_name = get_language_name(lang_code)

        if lang_name not in LANG_PROMPTS:
            continue

        samples.append({
            "audio": audio_path,
            "text": str(row["text"]),
            "lang": lang_name,
        })

    print(f"Loaded {len(samples)} samples from {csv_path}. Skipped missing: {skipped_missing}")
    if missing_examples:
        print("Missing path examples (first 5):")
        for p in missing_examples:
            print(f"  - {p}")

    return samples


# =========================
# EXTRACT TRANSCRIPTION
# =========================

def extract_transcription(raw_output: str, prompt: str) -> str:
    """
    Extract only the transcription from the model's raw output,
    stripping the prompt/chat template and any trailing special tokens.
    """
    # The model output includes the full prompt + generated text.
    # We need to remove the prompt portion.
    # Strategy 1: If the raw output starts with the prompt, strip it.
    if raw_output.startswith(prompt):
        transcription = raw_output[len(prompt):]
    else:
        # Strategy 2: Look for the assistant marker in chat template
        # Phi-4 chat template uses <|assistant|> before the response
        markers = ["<|assistant|>", "<|end|>"]
        for marker in ["<|assistant|>"]:
            if marker in raw_output:
                # Take text after the last occurrence of assistant marker
                parts = raw_output.split(marker)
                transcription = parts[-1]
                break
        else:
            # Fallback: use the raw output
            transcription = raw_output

    # Clean up special tokens and whitespace
    for token in ["<|end|>", "<|endoftext|>", "</s>", "<|user|>", "<|system|>"]:
        transcription = transcription.replace(token, "")

    return transcription.strip()


# =========================
# MAIN INFERENCE
# =========================

def main():
    print("=" * 60)
    print("Phi-4-Multimodal-Instruct LoRA Inference")
    print("Indic Languages: Assamese, Bengali, Hindi, Odia")
    print("=" * 60)

    # Load samples
    print("\nLoading test samples...")
    samples = load_samples(TEST_DIR)
    random.shuffle(samples)
    print(f"Total inference samples: {len(samples)}")

    if len(samples) == 0:
        print("ERROR: No samples found. Check TEST_DIR path and CSV.")
        return

    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Processor loaded.")

    # Patch GenerationMixin if needed (same as training script)
    import sys as _sys
    from transformers import GenerationMixin as _GenMixin
    for _key, _mod in _sys.modules.items():
        if "modeling_phi4mm" in _key and hasattr(_mod, "Phi4MMModel"):
            if not issubclass(_mod.Phi4MMModel, _GenMixin):
                _mod.Phi4MMModel.__bases__ = (_mod.Phi4MMModel.__bases__[0], _GenMixin)
                print("Applied GenerationMixin patch to Phi4MMModel (transformers>=4.50 compat)")
            break

    # Load fine-tuned model
    print(f"\nLoading fine-tuned model from: {FINETUNED_MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_MODEL_DIR,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    # Set speech LoRA adapter
    print("Setting speech LoRA adapter...")
    try:
        model.set_lora_adapter("speech")
        print("Speech LoRA adapter set successfully.")
    except Exception as e:
        print(f"Warning: Could not set LoRA adapter: {e}")
        print("Proceeding without explicit adapter setting (model may already have it active).")

    model.eval()

    # ---------------------------------------------------------------
    # Patch: modeling_phi4mm.py crashes when num_logits_to_keep=None
    # (newer transformers passes None; the custom forward does -None
    #  which is a TypeError). We wrap forward to default it to 0,
    #  meaning "keep all logits" (since -0 == 0 → full slice).
    # See: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/46
    # ---------------------------------------------------------------
    _original_forward = model.__class__.forward

    def _patched_forward(self, *args, **kwargs):
        if kwargs.get("num_logits_to_keep") is None:
            kwargs["num_logits_to_keep"] = 0  # -0 == 0 → keeps all positions
        return _original_forward(self, *args, **kwargs)

    model.__class__.forward = _patched_forward
    print("Applied num_logits_to_keep patch for modeling_phi4mm.py")

    print(f"Model loaded on {DEVICE}")

    # GPU memory info
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {alloc:.1f}GB allocated / {reserved:.1f}GB reserved")

    # =========================
    # STORAGE
    # =========================

    results_rows = []

    global_cer = []
    global_wer = []
    global_char_prec = []
    global_char_rec = []
    global_char_f1 = []
    global_word_prec = []
    global_word_rec = []
    global_word_f1 = []

    per_lang_stats = defaultdict(list)

    # =========================
    # INFERENCE LOOP
    # =========================

    print(f"\nStarting inference on {len(samples)} samples (batch size={BATCH_SIZE})...")
    progress = tqdm(total=len(samples), desc="Inference", unit="sample")

    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]

        for s in batch:
            fname = os.path.basename(s["audio"])
            lang = s["lang"]
            gt = s["text"].strip()

            # Load audio
            try:
                wav, sr = load_audio(s["audio"])
            except Exception as e:
                print(f"\nError loading {fname}: {e}")
                progress.update(1)
                continue

            # Build prompt using the same format as training
            instruction = LANG_PROMPTS.get(lang, f"Transcribe the audio clip into {lang} text.")

            user_message = {
                "role": "user",
                "content": f"<|audio_1|>\n{instruction}",
            }

            prompt = processor.tokenizer.apply_chat_template(
                [user_message], tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            inputs = processor(
                text=prompt,
                audios=[(wav, sr)],
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # Decode full output
            raw_output = processor.tokenizer.decode(
                generated_ids[0], skip_special_tokens=False
            )

            # Extract only the transcription (strip prompt and special tokens)
            pred = extract_transcription(raw_output, prompt)

            # Also try skip_special_tokens decode for cleaner output
            clean_output = processor.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            # If the clean output is shorter and doesn't contain prompt artifacts, prefer it
            # The clean output typically strips the prompt template tokens too
            # We need to extract just the generated part
            # generated_ids includes the input_ids, so we need to slice
            input_len = inputs["input_ids"].shape[1]
            generated_only = generated_ids[0][input_len:]
            pred = processor.tokenizer.decode(generated_only, skip_special_tokens=True).strip()

            # Clean up any remaining special tokens
            for token in ["<|end|>", "<|endoftext|>", "</s>"]:
                pred = pred.replace(token, "")
            pred = pred.strip()

            # =========================
            # COMPUTE METRICS
            # =========================

            sample_cer = safe_cer(gt, pred)
            sample_wer = safe_wer(gt, pred)
            char_p, char_r, char_f = char_precision_recall_f1(pred, gt)
            word_p, word_r, word_f = word_precision_recall_f1(pred, gt)

            # Accumulate global metrics
            global_cer.append(sample_cer)
            global_wer.append(sample_wer)
            global_char_prec.append(char_p)
            global_char_rec.append(char_r)
            global_char_f1.append(char_f)
            global_word_prec.append(word_p)
            global_word_rec.append(word_r)
            global_word_f1.append(word_f)

            per_lang_stats[lang].append({
                "cer": sample_cer,
                "wer": sample_wer,
                "char_p": char_p,
                "char_r": char_r,
                "char_f1": char_f,
                "word_p": word_p,
                "word_r": word_r,
                "word_f1": word_f,
            })

            # =========================
            # PRINT SAMPLE RESULTS
            # =========================

            print("\n==============================================")
            print(f"Model: {MODEL_NAME_PRINT}")
            print(f"Language: {lang}")
            print(f"File: {fname}")
            print("Prediction:")
            print(pred)
            print("Ground Truth:")
            print(gt)
            print(f"CER: {sample_cer:.4f}")
            print(f"Char Precision: {char_p:.4f}")
            print(f"Char Recall: {char_r:.4f}")
            print(f"Char F1: {char_f:.4f}")
            print(f"WER: {sample_wer:.4f}")
            print(f"Word Precision: {word_p:.4f}")
            print(f"Word Recall: {word_r:.4f}")
            print(f"Word F1: {word_f:.4f}")

            # Save row
            results_rows.append({
                "language": lang,
                "file": fname,
                "prediction": pred,
                "ground_truth": gt,
                "cer": sample_cer,
                "char_precision": char_p,
                "char_recall": char_r,
                "char_f1": char_f,
                "wer": sample_wer,
                "word_precision": word_p,
                "word_recall": word_r,
                "word_f1": word_f,
            })

            # Update progress bar
            progress.update(1)

            # Free GPU memory after each sample
            del inputs, generated_ids, generated_only
            torch.cuda.empty_cache()

    progress.close()

    # =========================
    # OVERALL METRICS
    # =========================

    if len(global_cer) == 0:
        print("\nNo samples were processed. Exiting.")
        return

    avg_cer = sum(global_cer) / len(global_cer)
    avg_wer = sum(global_wer) / len(global_wer)
    avg_char_p = sum(global_char_prec) / len(global_char_prec)
    avg_char_r = sum(global_char_rec) / len(global_char_rec)
    avg_char_f = sum(global_char_f1) / len(global_char_f1)
    avg_word_p = sum(global_word_prec) / len(global_word_prec)
    avg_word_r = sum(global_word_rec) / len(global_word_rec)
    avg_word_f = sum(global_word_f1) / len(global_word_f1)

    print("\n==============================================")
    print("OVERALL RESULTS")
    print("==============================================")
    print(f"Overall CER: {avg_cer:.4f}")
    print(f"Overall Char Precision: {avg_char_p:.4f}")
    print(f"Overall Char Recall: {avg_char_r:.4f}")
    print(f"Overall Char F1: {avg_char_f:.4f}")
    print(f"Overall WER: {avg_wer:.4f}")
    print(f"Overall Word Precision: {avg_word_p:.4f}")
    print(f"Overall Word Recall: {avg_word_r:.4f}")
    print(f"Overall Word F1: {avg_word_f:.4f}")

    # =========================
    # PER-LANGUAGE METRICS
    # =========================

    per_lang_rows = []

    print("\n==============================================")
    print("PER-LANGUAGE RESULTS")
    print("==============================================")

    for lang, stats in per_lang_stats.items():
        n = len(stats)
        lc = sum(s["cer"] for s in stats) / n
        lw = sum(s["wer"] for s in stats) / n
        lcp = sum(s["char_p"] for s in stats) / n
        lcr = sum(s["char_r"] for s in stats) / n
        lcf = sum(s["char_f1"] for s in stats) / n
        lwp = sum(s["word_p"] for s in stats) / n
        lwr = sum(s["word_r"] for s in stats) / n
        lwf = sum(s["word_f1"] for s in stats) / n

        print(f"\nLanguage: {lang}")
        print(f"  CER: {lc:.4f}")
        print(f"  Char Precision: {lcp:.4f}")
        print(f"  Char Recall: {lcr:.4f}")
        print(f"  Char F1: {lcf:.4f}")
        print(f"  WER: {lw:.4f}")
        print(f"  Word Precision: {lwp:.4f}")
        print(f"  Word Recall: {lwr:.4f}")
        print(f"  Word F1: {lwf:.4f}")

        per_lang_rows.append({
            "language": lang,
            "num_samples": n,
            "cer": lc,
            "char_precision": lcp,
            "char_recall": lcr,
            "char_f1": lcf,
            "wer": lw,
            "word_precision": lwp,
            "word_recall": lwr,
            "word_f1": lwf,
        })

    # =========================
    # SAVE FILES (CSV / TSV)
    # =========================

    # Detailed per-sample results as TSV (tab-separated, matching qwen2 style)
    df = pd.DataFrame(results_rows)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False, encoding="utf-8")

    # Per-language summary CSV
    df_lang = pd.DataFrame(per_lang_rows)
    df_lang.to_csv(PER_LANG_CSV, index=False, encoding="utf-8")

    # Overall summary CSV (single row)
    df_overall = pd.DataFrame([{
        "model": MODEL_NAME_PRINT,
        "num_samples": len(global_cer),
        "cer": avg_cer,
        "char_precision": avg_char_p,
        "char_recall": avg_char_r,
        "char_f1": avg_char_f,
        "wer": avg_wer,
        "word_precision": avg_word_p,
        "word_recall": avg_word_r,
        "word_f1": avg_word_f,
    }])
    df_overall.to_csv(OVERALL_CSV, index=False, encoding="utf-8")

    # =========================
    # SAVE SUMMARY PNGs
    # =========================

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # ---- Per-language grouped bar chart ----
        langs_order = [r["language"] for r in per_lang_rows]
        metric_names = ["CER", "WER", "Char F1", "Word F1",
                        "Char Prec", "Char Rec", "Word Prec", "Word Rec"]
        metric_keys = ["cer", "wer", "char_f1", "word_f1",
                       "char_precision", "char_recall", "word_precision", "word_recall"]

        x = np.arange(len(langs_order))
        width = 0.10

        fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(langs_order)), 6))
        for i, (name, key) in enumerate(zip(metric_names, metric_keys)):
            values = [r[key] for r in per_lang_rows]
            ax.bar(x + (i - len(metric_names) / 2) * width + width / 2,
                   values, width, label=name)

        ax.set_xticks(x)
        ax.set_xticklabels(langs_order)
        ax.set_ylabel("Score")
        ax.set_title(f"Per-Language Metrics\n{MODEL_NAME_PRINT}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", ncol=4, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(PER_LANG_PNG, dpi=150)
        plt.close(fig)

        # ---- Overall metrics bar chart ----
        overall_labels = ["CER", "WER", "Char Prec", "Char Rec", "Char F1",
                          "Word Prec", "Word Rec", "Word F1"]
        overall_values = [avg_cer, avg_wer, avg_char_p, avg_char_r, avg_char_f,
                          avg_word_p, avg_word_r, avg_word_f]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#d62728", "#d62728",        # error metrics (red)
                  "#1f77b4", "#1f77b4", "#1f77b4",  # char metrics (blue)
                  "#2ca02c", "#2ca02c", "#2ca02c"]  # word metrics (green)
        bars = ax.bar(overall_labels, overall_values, color=colors)
        for bar, v in zip(bars, overall_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Overall Metrics ({len(global_cer)} samples)\n{MODEL_NAME_PRINT}")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(OVERALL_PNG, dpi=150)
        plt.close(fig)

        png_saved = True
    except Exception as e:
        print(f"\n[WARN] Could not save PNG summaries: {e}")
        png_saved = False

    print("\n==============================================")
    print("FILES SAVED")
    print("==============================================")
    print(f"Detailed results:       {OUTPUT_TSV}")
    print(f"Per-language summary:   {PER_LANG_CSV}")
    print(f"Overall summary:        {OVERALL_CSV}")
    if png_saved:
        print(f"Per-language chart:     {PER_LANG_PNG}")
        print(f"Overall chart:          {OVERALL_PNG}")
    print("\nInference complete!")


if __name__ == "__main__":
    main()