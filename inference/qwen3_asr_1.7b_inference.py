"""
Qwen3-ASR-1.7B LoRA Inference for Indic Languages
(Assamese, Bengali, Hindi, Odia) - IndicVoices test split

Loads the LoRA adapter trained by `qwen3_asr_lora_indic_train.py` and runs
batched inference over the `test` split of the IndicVoices mini subset.

Metrics computed (per-sample, per-language, and overall):
    - CER (char-level edit distance via jiwer)
    - char-level precision / recall / F1  (Counter-based, matches the
      qwen2_audio_lora_inference.py strategy)
    - WER (word-level edit distance via jiwer)
    - word-level precision / recall / F1

Outputs (written next to this script):
    results.tsv              - per-sample rows
    per_language_results.csv - per-language summary
    overall_results.csv      - one-row global summary
    per_language_metrics.png - bar chart of per-lang CER/WER/F1

IMPORTANT design notes
----------------------
1. Model loading follows the SAME pattern as qwen3_asr_lora_indic_train.py:
   we go through the `qwen_asr` Python package's `Qwen3ASRModel.from_pretrained`
   wrapper and pull out the inner `.model` (nn.Module) and `.processor`. The
   stock HF `AutoProcessor` does NOT know about `Qwen3ASRProcessor`, so this
   wrapper is required.

2. The LoRA adapter is attached with `PeftModel.from_pretrained(...)` on the
   inner model. We do NOT call `model.merge_and_unload()` because the base
   is 4-bit quantized (merging LoRA into 4-bit layers is not supported).

3. We call `model.generate(**inputs)` directly. The outer
   `Qwen3ASRForConditionalGeneration.generate()` method internally routes to
   `self.thinker.generate(...)` and already handles `input_features` /
   `feature_attention_mask` / `attention_mask` correctly — so no `forward`
   patching is needed for inference. PeftModel.generate() forwards to this.

4. The model emits outputs in the format:
        "language {DetectedLang}<asr_text>{transcription}"
   We parse this to separate the detected language from the actual
   transcription, and we compare ONLY the transcription part against the
   ground truth — the prompt, chat template, and `language X<asr_text>`
   prefix are all stripped before metric computation.
"""

import os
import re
import csv
import random
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from tqdm import tqdm

from transformers import BitsAndBytesConfig
from peft import PeftModel
from jiwer import cer, wer

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"

# Path to the LoRA adapter produced by the training script.
# This should be the directory that contains adapter_config.json / adapter_model.safetensors.
LORA_PATH = r"/path/to/data/IndicVoices_subset_4lang_combined/qwen3_asr_lora_indic/final_adapter"

# Test split root (contains test.csv and an audio/ folder with the wav files)
TEST_DIR = r"/path/to/data/IndicVoices_subset_4lang_combined/test"
TEST_CSV_NAME = "test.csv"

# Output files (written to the working directory)
OUTPUT_TSV       = "results.tsv"
PER_LANG_CSV     = "per_language_results.csv"
OVERALL_CSV      = "overall_results.csv"
PER_LANG_PNG     = "per_language_metrics.png"

BATCH_SIZE       = 2
RANDOM_SEED      = 42
# Targets are "language X<asr_text>TRANSCRIPTION" — for IndicVoices clips
# averaging ~7s the target is rarely more than ~60-80 tokens. 128 is a
# generous upper bound and EOS (<|im_end|> / <|endoftext|>) handles the
# early stop. Lowering from 256 to 128 is pure speed win: the decoder
# never generates tokens that would have been thrown away anyway.
MAX_NEW_TOKENS   = 128

USE_FLASH_ATTENTION_2 = True

# Per spec: no CPU multithreading.
CPU_NUM_THREADS = 1

# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

torch.set_num_threads(CPU_NUM_THREADS)
torch.set_num_interop_threads(CPU_NUM_THREADS)

# Speed knobs that do NOT use CPU threads:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not found. This script requires a GPU.")

DEVICE = torch.device("cuda:0")


# ============================================================
# LANGUAGE HELPERS (identical to training script)
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

def normalize_language(lang_value) -> str:
    value = str(lang_value).strip().lower()
    return LANG_CODE_TO_NAME.get(value, str(lang_value).strip())


# ============================================================
# DATA LOADING
# ============================================================

def resolve_audio_path(path_value: str, split_dir: str) -> str:
    """Same resolution logic as the training script, so the inference code
    works against either WSL paths, Windows paths, or bare basenames in test.csv."""
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
    """Load the COMPLETE audio clip - no chunking. Mono float32 @ 16kHz."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.size == 0:
        raise RuntimeError(f"Audio file is empty: {path}")
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.asarray(wav, dtype=np.float32)


def get_audio_duration_fast(path: str) -> float:
    """
    Read ONLY the WAV/FLAC header to get duration, without decoding samples.
    soundfile.info() is O(1); it doesn't touch the audio data at all.
    Used for bucketed batching (see main()).
    """
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    except Exception:
        # If the header is unreadable we just return 0 so the sample sinks
        # to the short-bucket side; the actual load will error out later
        # and get skipped in the main loop.
        return 0.0


def bucketed_random_order(samples, bucket_size: int = 32):
    """
    Shuffle-then-bucket-sort: first randomize the whole list (so there is
    no CSV-order bias and so languages mix), then within contiguous buckets
    of `bucket_size` sort by duration. This keeps two adjacent samples in a
    batch of 2 close in length, which eliminates almost all padding waste
    from pairing a 21s clip with a 0.35s clip while still honoring the
    "random" sampling spec at the dataset level.

    Same pattern the training script uses for the same reason.
    """
    shuffled = list(samples)
    random.shuffle(shuffled)
    ordered = []
    for i in range(0, len(shuffled), bucket_size):
        bucket = shuffled[i:i + bucket_size]
        bucket.sort(key=lambda s: s.get("_duration", 0.0))
        ordered.extend(bucket)
    return ordered


def load_test_samples(test_dir: str, csv_name: str):
    csv_path = os.path.join(test_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["path", "text", "lang"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    samples = []
    missing_count = 0
    for _, row in df.iterrows():
        audio_path = resolve_audio_path(row["path"], test_dir)
        if not os.path.exists(audio_path):
            missing_count += 1
            continue

        samples.append({
            "audio_path": audio_path,
            "file_name":  os.path.basename(audio_path),
            "lang_name":  normalize_language(row["lang"]),
            "transcript": str(row["text"]).strip(),
            "_duration":  get_audio_duration_fast(audio_path),
        })

    if missing_count > 0:
        print(f"[Info] {missing_count} rows skipped because audio file was not found in {csv_path}")
    if not samples:
        raise RuntimeError(f"No usable test samples found in {csv_path}")

    return samples


# ============================================================
# MODEL LOADING (Qwen3-ASR + LoRA)
# ============================================================

def gpu_supports_bf16() -> bool:
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def load_model_and_processor():
    """Load Qwen3-ASR-1.7B through the qwen_asr wrapper and attach LoRA."""
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        raise ImportError(
            "The `qwen_asr` package is required. Install with: pip install -U qwen-asr"
        ) from e

    compute_dtype = torch.bfloat16 if gpu_supports_bf16() else torch.float16
    preferred_attn = "flash_attention_2" if USE_FLASH_ATTENTION_2 else "sdpa"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    def _load(attn_impl):
        wrapper = Qwen3ASRModel.from_pretrained(
            MODEL_NAME,
            dtype=compute_dtype,
            device_map="cuda:0",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
        return wrapper.model, wrapper.processor

    print("Loading Qwen3-ASR-1.7B base model in 4-bit...")
    try:
        model, processor = _load(preferred_attn)
        used_attn = preferred_attn
    except Exception as e:
        if preferred_attn == "flash_attention_2":
            print(f"[Info] flash_attention_2 load failed, retrying with sdpa.\n{e}")
            model, processor = _load("sdpa")
            used_attn = "sdpa"
        else:
            raise

    print(f"Attn implementation: {used_attn}")

    print(f"Loading LoRA adapter from: {LORA_PATH}")
    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(f"LORA_PATH does not exist: {LORA_PATH}")

    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()

    # ------------------------------------------------------------------
    # CRITICAL SPEED FIX: the training script enables gradient checkpointing,
    # which forces `config.use_cache=False`. This flag persists through the
    # saved PEFT adapter load, so by default `generate()` here runs WITHOUT
    # a KV cache — every new token recomputes attention over the entire
    # growing sequence (quadratic instead of linear). Re-enable use_cache
    # and disable gradient checkpointing explicitly for inference.
    # See: https://github.com/huggingface/peft/issues/742
    # ------------------------------------------------------------------
    def _reenable_cache(m):
        try:
            if hasattr(m, "gradient_checkpointing_disable"):
                m.gradient_checkpointing_disable()
        except Exception:
            pass
        if hasattr(m, "config") and hasattr(m.config, "use_cache"):
            m.config.use_cache = True
        # Some wrappers (PeftModel, Qwen3-ASR outer) hold an inner base model.
        for inner_attr in ("base_model", "model", "thinker"):
            inner = getattr(m, inner_attr, None)
            if inner is not None and inner is not m:
                _reenable_cache(inner)

    _reenable_cache(model)

    return model, processor, compute_dtype


# ============================================================
# PROMPT BUILDING
# ============================================================

def render_prefix_text(processor) -> str:
    """
    Exact same prefix used by the training script. We tokenize this on its own
    to get the prompt length, then feed it to the model with add_generation_prompt=True.
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
        messages, add_generation_prompt=True, tokenize=False,
    )
    if not isinstance(prefix_text, str):
        raise RuntimeError("processor.apply_chat_template(...) did not return a string.")
    return prefix_text


def move_inputs_to_device(inputs: dict, device: torch.device, model_dtype: torch.dtype):
    """Move tensors to device and cast floating tensors to model_dtype."""
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


# ============================================================
# OUTPUT PARSING
# ============================================================

_ASR_PATTERN = re.compile(
    r"^\s*language\s+(?P<lang>[^<\n]+?)\s*<asr_text>\s*(?P<text>.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

def parse_asr_output(raw: str):
    """
    Model emits:
        "language Hindi<asr_text>नमस्ते..."

    Returns (detected_language, transcription). If the prefix is missing we
    fall back to ("unknown", raw.strip()).
    """
    if raw is None:
        return "unknown", ""
    s = raw.strip()

    # Some decoders may leave leftover assistant markers or EOS text; strip them.
    for junk in ("<|im_end|>", "<|endoftext|>"):
        if junk in s:
            s = s.split(junk, 1)[0]

    m = _ASR_PATTERN.match(s)
    if m:
        return m.group("lang").strip(), m.group("text").strip()

    # Fallback: look for <asr_text> anywhere.
    if "<asr_text>" in s:
        left, right = s.split("<asr_text>", 1)
        lang = left.strip()
        if lang.lower().startswith("language"):
            lang = lang[len("language"):].strip()
        return (lang or "unknown"), right.strip()

    return "unknown", s


# ============================================================
# METRICS
# ============================================================

def counter_prf(pred_tokens, ref_tokens):
    """
    Multiset precision/recall/F1 on the given token lists. Matches the
    char_precision_recall_f1 strategy in qwen2_audio_lora_inference.py,
    generalized to work for both characters and words.
    """
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0:
        return 0.0, 0.0, 0.0

    pred_c = Counter(pred_tokens)
    ref_c  = Counter(ref_tokens)

    tp = sum((pred_c & ref_c).values())
    fp = sum((pred_c - ref_c).values())
    fn = sum((ref_c - pred_c).values())

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def safe_cer(ref: str, hyp: str) -> float:
    if len(ref) == 0 and len(hyp) == 0:
        return 0.0
    if len(ref) == 0:
        return 1.0
    try:
        return cer(ref, hyp)
    except Exception:
        return 1.0


def safe_wer(ref: str, hyp: str) -> float:
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) == 0 and len(hyp_w) == 0:
        return 0.0
    if len(ref_w) == 0:
        return 1.0
    try:
        return wer(ref, hyp)
    except Exception:
        return 1.0


def compute_all_metrics(pred: str, ref: str):
    c = safe_cer(ref, pred)
    w = safe_wer(ref, pred)

    char_p, char_r, char_f = counter_prf(list(pred), list(ref))
    word_p, word_r, word_f = counter_prf(pred.split(), ref.split())

    return {
        "cer": c,
        "wer": w,
        "char_precision": char_p,
        "char_recall":    char_r,
        "char_f1":        char_f,
        "word_precision": word_p,
        "word_recall":    word_r,
        "word_f1":        word_f,
    }


# ============================================================
# PLOTTING
# ============================================================

def save_per_language_plot(per_lang_rows, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Info] matplotlib not available, skipping PNG plot.")
        return

    if not per_lang_rows:
        return

    langs = [r["language"] for r in per_lang_rows]
    cers  = [r["cer"]      for r in per_lang_rows]
    wers  = [r["wer"]      for r in per_lang_rows]
    cf1s  = [r["char_f1"]  for r in per_lang_rows]
    wf1s  = [r["word_f1"]  for r in per_lang_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(langs))
    width = 0.35

    axes[0].bar(x - width/2, cers, width, label="CER", color="#e74c3c")
    axes[0].bar(x + width/2, wers, width, label="WER", color="#e67e22")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(langs, rotation=20)
    axes[0].set_ylabel("Error Rate")
    axes[0].set_title("Per-Language CER / WER (lower = better)")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x - width/2, cf1s, width, label="Char F1", color="#27ae60")
    axes[1].bar(x + width/2, wf1s, width, label="Word F1", color="#2980b9")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(langs, rotation=20)
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Per-Language F1 (higher = better)")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Qwen3-ASR-1.7B + LoRA — IndicVoices Test Split", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Qwen3-ASR-1.7B LoRA Inference — IndicVoices test split")
    print("=" * 70)
    print(f"Base model  : {MODEL_NAME}")
    print(f"LoRA adapter: {LORA_PATH}")
    print(f"Test dir    : {TEST_DIR}")
    print(f"Batch size  : {BATCH_SIZE}")
    print("=" * 70)

    # ---------- Load samples ----------
    samples = load_test_samples(TEST_DIR, TEST_CSV_NAME)
    print(f"Total test samples: {len(samples)}")
    lang_counts = defaultdict(int)
    for s in samples:
        lang_counts[s["lang_name"]] += 1
    print("Language distribution:")
    for lang, n in sorted(lang_counts.items()):
        print(f"  {lang}: {n}")

    # Bucketed-random order: shuffle dataset-wide, then sort by duration
    # within small buckets. Keeps "random" sampling spec at the dataset
    # level while pairing similar-length clips inside each batch of 2,
    # which eliminates padding-to-longest waste when a 21s clip would
    # otherwise get batched with a 0.35s clip.
    samples = bucketed_random_order(samples, bucket_size=32)

    # ---------- Load model + LoRA ----------
    model, processor, model_dtype = load_model_and_processor()

    # ---------- Build prompt prefix ----------
    prefix_text = render_prefix_text(processor)

    # Pre-build the padded prefix list for the nominal batch size. We only
    # slice it when the final batch is smaller (short tail). This avoids
    # allocating a fresh 2-element list on every iteration.
    _prefix_batch_full = [prefix_text] * BATCH_SIZE

    # Pre-tokenize the prefix ONCE (text-only, no audio) so we know how many
    # tokens to slice off each generated sequence. The actual prefix length
    # in the final batched input depends on how many audio placeholder tokens
    # get expanded, so we re-compute per-sample prompt length from input_ids.
    #
    # (We handle this dynamically below using inputs["input_ids"].shape[1].)

    # ---------- Inference ----------
    results_rows = []

    # Global accumulators
    global_cer, global_wer = [], []
    global_cp,  global_cr,  global_cf = [], [], []
    global_wp,  global_wr,  global_wf = [], [], []

    # Per-language accumulators
    per_lang = defaultdict(list)  # lang -> list of metric dicts

    progress = tqdm(total=len(samples), desc="Inference", unit="samples")

    MODEL_NAME_PRINT = f"{MODEL_NAME} + LoRA({os.path.basename(LORA_PATH.rstrip('/\\'))})"

    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]

        # Load audio for this batch
        audio_arrays = []
        file_names, actual_langs, gt_texts, audio_paths = [], [], [], []
        for s in batch:
            try:
                wav = load_audio(s["audio_path"], target_sr=16000)
            except Exception as e:
                print(f"  [skip] {s['audio_path']}: {e}")
                continue
            audio_arrays.append(wav)
            file_names.append(s["file_name"])
            actual_langs.append(s["lang_name"])
            gt_texts.append(s["transcript"])
            audio_paths.append(s["audio_path"])

        if not audio_arrays:
            progress.update(len(batch))
            continue

        # Build inputs: SAME prefix_text for every sample in the batch.
        n = len(audio_arrays)
        prefix_texts = _prefix_batch_full if n == BATCH_SIZE else [prefix_text] * n
        try:
            inputs = processor(
                text=prefix_texts,
                audio=audio_arrays,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
        except TypeError:
            inputs = processor(
                text=prefix_texts,
                audios=audio_arrays,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        inputs = move_inputs_to_device(inputs, DEVICE, model_dtype)

        # inference_mode is strictly faster than no_grad (skips version counter
        # bookkeeping). use_cache=True is also passed explicitly as a belt-and-
        # suspenders guarantee even if config.use_cache got reset somewhere.
        with torch.inference_mode():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )

        # `Qwen3ASRForConditionalGeneration.generate` returns a
        # GenerateDecoderOnlyOutput (return_dict_in_generate=True) OR a
        # bare tensor depending on kwargs. Handle both.
        if hasattr(gen_out, "sequences"):
            gen_ids = gen_out.sequences
        else:
            gen_ids = gen_out

        # Slice off the prompt (per-row). With left padding the prompt length
        # is uniform across the batch (== prompt_len), so a single slice works.
        new_ids = gen_ids[:, prompt_len:]

        decoded = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        # -------- Per-sample processing --------
        for j, raw in enumerate(decoded):
            detected_lang, pred_text = parse_asr_output(raw)
            gt_text = gt_texts[j].strip()
            actual_lang = actual_langs[j]
            fname = file_names[j]

            m = compute_all_metrics(pred_text, gt_text)

            # Accumulate globals
            global_cer.append(m["cer"])
            global_wer.append(m["wer"])
            global_cp.append(m["char_precision"])
            global_cr.append(m["char_recall"])
            global_cf.append(m["char_f1"])
            global_wp.append(m["word_precision"])
            global_wr.append(m["word_recall"])
            global_wf.append(m["word_f1"])

            per_lang[actual_lang].append(m)

            # Console print (per-sample, qwen2_audio_lora_inference.py style)
            print("\n" + "=" * 46)
            print(f"Model: {MODEL_NAME_PRINT}")
            print(f"Actual Language:   {actual_lang}")
            print(f"Detected Language: {detected_lang}")
            print(f"File: {fname}")
            print("Prediction:")
            print(pred_text)
            print("Ground Truth:")
            print(gt_text)
            print(f"CER: {m['cer']:.4f} | WER: {m['wer']:.4f}")
            print(f"Char  P/R/F1: {m['char_precision']:.4f} / {m['char_recall']:.4f} / {m['char_f1']:.4f}")
            print(f"Word  P/R/F1: {m['word_precision']:.4f} / {m['word_recall']:.4f} / {m['word_f1']:.4f}")

            results_rows.append({
                "file":              fname,
                "actual_language":   actual_lang,
                "detected_language": detected_lang,
                "prediction":        pred_text,
                "ground_truth":      gt_text,
                "cer":               m["cer"],
                "wer":               m["wer"],
                "char_precision":    m["char_precision"],
                "char_recall":       m["char_recall"],
                "char_f1":           m["char_f1"],
                "word_precision":    m["word_precision"],
                "word_recall":       m["word_recall"],
                "word_f1":           m["word_f1"],
            })

        progress.update(len(batch))

    progress.close()

    # ============================================================
    # OVERALL METRICS
    # ============================================================

    def _avg(xs): return (sum(xs) / len(xs)) if xs else 0.0

    overall = {
        "cer":            _avg(global_cer),
        "wer":            _avg(global_wer),
        "char_precision": _avg(global_cp),
        "char_recall":    _avg(global_cr),
        "char_f1":        _avg(global_cf),
        "word_precision": _avg(global_wp),
        "word_recall":    _avg(global_wr),
        "word_f1":        _avg(global_wf),
        "num_samples":    len(global_cer),
    }

    print("\n" + "=" * 46)
    print("OVERALL RESULTS")
    print("=" * 46)
    print(f"Samples evaluated : {overall['num_samples']}")
    print(f"Overall CER : {overall['cer']:.4f}")
    print(f"Overall WER : {overall['wer']:.4f}")
    print(f"Char  Precision / Recall / F1 : "
          f"{overall['char_precision']:.4f} / {overall['char_recall']:.4f} / {overall['char_f1']:.4f}")
    print(f"Word  Precision / Recall / F1 : "
          f"{overall['word_precision']:.4f} / {overall['word_recall']:.4f} / {overall['word_f1']:.4f}")

    # ============================================================
    # PER-LANGUAGE METRICS
    # ============================================================

    per_lang_rows = []
    print("\n" + "=" * 46)
    print("PER-LANGUAGE RESULTS")
    print("=" * 46)
    for lang in sorted(per_lang.keys()):
        ms = per_lang[lang]
        row = {
            "language":       lang,
            "num_samples":    len(ms),
            "cer":            _avg([m["cer"] for m in ms]),
            "wer":            _avg([m["wer"] for m in ms]),
            "char_precision": _avg([m["char_precision"] for m in ms]),
            "char_recall":    _avg([m["char_recall"]    for m in ms]),
            "char_f1":        _avg([m["char_f1"]        for m in ms]),
            "word_precision": _avg([m["word_precision"] for m in ms]),
            "word_recall":    _avg([m["word_recall"]    for m in ms]),
            "word_f1":        _avg([m["word_f1"]        for m in ms]),
        }
        per_lang_rows.append(row)

        print(f"\nLanguage: {lang}  (n={row['num_samples']})")
        print(f"  CER: {row['cer']:.4f}  |  WER: {row['wer']:.4f}")
        print(f"  Char P/R/F1: {row['char_precision']:.4f} / "
              f"{row['char_recall']:.4f} / {row['char_f1']:.4f}")
        print(f"  Word P/R/F1: {row['word_precision']:.4f} / "
              f"{row['word_recall']:.4f} / {row['word_f1']:.4f}")

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================

    # Per-sample TSV
    df_rows = pd.DataFrame(results_rows)
    df_rows.to_csv(OUTPUT_TSV, sep="\t", index=False, encoding="utf-8")

    # Per-language CSV
    df_lang = pd.DataFrame(per_lang_rows)
    df_lang.to_csv(PER_LANG_CSV, index=False, encoding="utf-8")

    # Overall CSV (single-row)
    df_overall = pd.DataFrame([overall])
    df_overall.to_csv(OVERALL_CSV, index=False, encoding="utf-8")

    # PNG plot
    save_per_language_plot(per_lang_rows, PER_LANG_PNG)

    print("\n" + "=" * 46)
    print("FILES SAVED")
    print("=" * 46)
    print(f"Per-sample results : {OUTPUT_TSV}")
    print(f"Per-language summary: {PER_LANG_CSV}")
    print(f"Overall summary     : {OVERALL_CSV}")
    print(f"Per-language plot   : {PER_LANG_PNG}")


if __name__ == "__main__":
    main()