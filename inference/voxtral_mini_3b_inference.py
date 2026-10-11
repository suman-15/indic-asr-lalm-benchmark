#!/usr/bin/env python3
"""
Voxtral-Mini-3B-2507 (+ LoRA) inference on IndicVoices test split.

Fixes vs previous version
-------------------------
1) Corrects ADAPTER_DIR so the saved LoRA adapter is actually loaded.
2) Raises an error if the adapter directory does not exist instead of silently
   falling back to the base model.
3) Removes the forced `language="hi"` hint for every sample and lets Voxtral
   detect language automatically.
4) Reports how many rows were skipped because audio files were missing or
   transcripts were empty.
"""

import os
import json
import csv
import random
import warnings
from collections import defaultdict, Counter
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import librosa
import jiwer
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import VoxtralForConditionalGeneration, AutoProcessor
from peft import PeftModel

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT    = r"/path/to/data/IndicVoices_subset_4lang_combined"
SPLIT        = "test"
BASE_MODEL   = "mistralai/Voxtral-Mini-3B-2507"
ADAPTER_DIR  = os.path.join(DATA_ROOT, "voxtral_lora_indic_v3", "final_model")
OUTPUT_DIR   = os.path.join(DATA_ROOT, "inference_test_fixed")

BATCH_SIZE       = 2
TARGET_SR        = 16000
MAX_NEW_TOKENS   = 384
N_BOOTSTRAP      = 2000
SEED             = 42
NORMALIZE_TEXT   = True

LANG_CODE_TO_NAME = {
    "as": "Assamese", "bn": "Bengali", "hi": "Hindi", "or": "Odia",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "kn": "Kannada",
    "gu": "Gujarati", "mr": "Marathi", "pa": "Punjabi",
}


# =============================================================================
# HELPERS
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_language_name(code):
    return LANG_CODE_TO_NAME.get(str(code).lower().strip(), str(code).capitalize())


def instruction_for(lang_code):
    """Per-sample English instruction. Must match the training-time prompt
    in mistralai-Voxtral-Mini-indic-finetune.py exactly."""
    name = get_language_name(lang_code)
    return f"Transcribe the following {name} audio word-for-word."


def normalize_text(text):
    if not NORMALIZE_TEXT:
        return str(text)
    text = str(text).strip()
    text = " ".join(text.split())
    text = text.strip('.,!?;:"\'-')
    return text


def resolve_audio_path(csv_path_value: str, audio_dir: str) -> str:
    raw = str(csv_path_value).strip()
    normalized = raw.replace("\\", "/")
    candidates = [
        raw,
        normalized,
        os.path.join(audio_dir, os.path.basename(normalized)),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return candidates[-1]


def load_audio(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return wav.astype(np.float32)


def load_test_samples():
    split_dir = os.path.join(DATA_ROOT, SPLIT)
    audio_dir = os.path.join(split_dir, "audio")
    csv_path  = os.path.join(split_dir, f"{SPLIT}.csv")
    df = pd.read_csv(csv_path)

    samples = []
    missing_audio = 0
    empty_text = 0
    per_lang = defaultdict(int)

    for _, row in df.iterrows():
        ap = resolve_audio_path(row["path"], audio_dir)
        if not os.path.isfile(ap):
            missing_audio += 1
            continue

        ref = row.get("normalized")
        if not isinstance(ref, str) or not ref.strip():
            ref = row.get("text")

        if not isinstance(ref, str) or not ref.strip():
            empty_text += 1
            continue

        lang = str(row["lang"]).strip().lower()
        samples.append({
            "audio_path": ap,
            "filename": os.path.basename(ap),
            "reference": ref.strip(),
            "lang": lang,
        })
        per_lang[lang] += 1

    print(f"Loaded {len(samples)} test samples from {csv_path}")
    print(f"Skipped missing_audio={missing_audio}, empty_text={empty_text}")
    print(f"Per-language counts: {dict(sorted(per_lang.items()))}")
    return samples


# =============================================================================
# METRICS
# =============================================================================
def compute_wer_measures(refs, hyps):
    if isinstance(refs, str):
        refs = [refs]
    if isinstance(hyps, str):
        hyps = [hyps]
    out = jiwer.process_words(refs, hyps)
    return {
        "wer": out.wer,
        "hits": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
    }


def compute_cer_measures(refs, hyps):
    if isinstance(refs, str):
        refs = [refs]
    if isinstance(hyps, str):
        hyps = [hyps]
    out = jiwer.process_characters(refs, hyps)
    return out.cer, out.substitutions, out.deletions, out.insertions, out.hits


def char_prf(pred, ref):
    pred, ref = list(pred), list(ref)
    if not pred and not ref:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    pc, rc = Counter(pred), Counter(ref)
    tp = sum((pc & rc).values())
    fp = sum((pc - rc).values())
    fn = sum((rc - pc).values())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def word_prf(pred, ref):
    pw, rw = pred.split(), ref.split()
    if not pw and not rw:
        return 1.0, 1.0, 1.0
    if not pw:
        return 0.0, 0.0, 0.0
    pc, rc = Counter(pw), Counter(rw)
    tp = sum((pc & rc).values())
    fp = sum((pc - rc).values())
    fn = sum((rc - pc).values())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def bootstrap_wer_ci(refs, hyps, n=N_BOOTSTRAP, alpha=0.05):
    if len(refs) < 2:
        m = compute_wer_measures(refs, hyps)
        return m["wer"], 0.0, m["wer"], m["wer"]
    N = len(refs)
    samples = []
    for _ in range(n):
        idx = np.random.randint(0, N, N)
        samples.append(compute_wer_measures([refs[i] for i in idx], [hyps[i] for i in idx])["wer"])
    a = np.array(samples)
    return float(a.mean()), float(a.std(ddof=1)), float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2)))


def bootstrap_cer_ci(refs, hyps, n=N_BOOTSTRAP, alpha=0.05):
    if len(refs) < 2:
        c, *_ = compute_cer_measures(refs, hyps)
        return c, 0.0, c, c
    N = len(refs)
    samples = []
    for _ in range(n):
        idx = np.random.randint(0, N, N)
        c, *_ = compute_cer_measures([refs[i] for i in idx], [hyps[i] for i in idx])
        samples.append(c)
    a = np.array(samples)
    return float(a.mean()), float(a.std(ddof=1)), float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2)))


def bootstrap_metric_ci(values, n=N_BOOTSTRAP, alpha=0.05):
    if len(values) < 2:
        v = float(values[0]) if values else 0.0
        return v, 0.0, v, v
    v = np.array(values)
    N = len(v)
    samples = []
    for _ in range(n):
        samples.append(v[np.random.randint(0, N, N)].mean())
    a = np.array(samples)
    return float(a.mean()), float(a.std(ddof=1)), float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2)))


# =============================================================================
# PLOTS
# =============================================================================
def generate_plots(results_rows, ci_metrics, output_dir):
    try:
        lang_wers = defaultdict(list)
        for r in results_rows:
            lang_wers[r["language"]].append(r["wer"])
        langs = sorted(lang_wers.keys())
        wer_data = [lang_wers[l] for l in langs]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].boxplot(wer_data, labels=langs)
        axes[0].set_title("WER distribution by language")
        axes[0].set_ylabel("WER")
        axes[0].grid(alpha=0.3)

        x = list(range(len(langs)))
        width = 0.35
        avg_wers   = [ci_metrics["per_language"][l]["wer"] for l in langs]
        wer_err_lo = [ci_metrics["per_language"][l]["wer"] - ci_metrics["per_language"][l]["wer_ci_lower"] for l in langs]
        wer_err_hi = [ci_metrics["per_language"][l]["wer_ci_upper"] - ci_metrics["per_language"][l]["wer"] for l in langs]
        avg_cers   = [ci_metrics["per_language"][l]["cer"] for l in langs]
        cer_err_lo = [ci_metrics["per_language"][l]["cer"] - ci_metrics["per_language"][l]["cer_ci_lower"] for l in langs]
        cer_err_hi = [ci_metrics["per_language"][l]["cer_ci_upper"] - ci_metrics["per_language"][l]["cer"] for l in langs]

        axes[1].bar([i - width / 2 for i in x], avg_wers, width, yerr=[wer_err_lo, wer_err_hi], capsize=4, label="WER")
        axes[1].bar([i + width / 2 for i in x], avg_cers, width, yerr=[cer_err_lo, cer_err_hi], capsize=4, label="CER")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(langs)
        axes[1].set_title("Mean WER/CER with 95% CI")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "inference_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for l in langs:
            lr = [r for r in results_rows if r["language"] == l]
            ax.scatter([r["wer"] for r in lr], [r["cer"] for r in lr], label=l, alpha=0.6, s=40)
        ax.set_xlabel("WER")
        ax.set_ylabel("CER")
        ax.set_title("Per-sample WER vs CER")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wer_vs_cer_scatter.png"), dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, max(6, 2 + len(langs) * 0.8)))
        ov = ci_metrics["overall"]
        ax.errorbar(
            ov["wer"], 0,
            xerr=[[ov["wer"] - ov["wer_ci_lower"]], [ov["wer_ci_upper"] - ov["wer"]]],
            fmt="o", capsize=5, label="Overall WER"
        )
        for i, l in enumerate(langs):
            lm = ci_metrics["per_language"][l]
            ax.errorbar(
                lm["wer"], i + 1,
                xerr=[[lm["wer"] - lm["wer_ci_lower"]], [lm["wer_ci_upper"] - lm["wer"]]],
                fmt="s", capsize=5, label=f"{l} WER"
            )
        ax.set_yticks(list(range(len(langs) + 1)))
        ax.set_yticklabels(["Overall"] + langs)
        ax.set_xlabel("WER")
        ax.set_title("WER with 95% bootstrap CIs")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confidence_intervals.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print("Saved: inference_metrics.png, wer_vs_cer_scatter.png, confidence_intervals.png")
    except Exception as e:
        print(f"[plot] skipped: {e}")


# =============================================================================
# INFERENCE
# =============================================================================
@torch.inference_mode()
def transcribe_batch(model, processor, chunk, device, compute_dtype):
    """Per-sample chat-template prompting -- matches v3 training exactly.
    Each sample carries its own language tag, so the model sees a
    distinct conditioning signal per audio."""
    conversations = [
        [{
            "role": "user",
            "content": [
                {"type": "audio", "path": s["audio_path"]},
                {"type": "text",  "text": instruction_for(s["lang"])},
            ],
        }]
        for s in chunk
    ]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device, dtype=compute_dtype)
    prompt_len = inputs.input_ids.shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    decoded = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return [d.strip() for d in decoded]


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {compute_dtype}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\nLoading processor from: {BASE_MODEL}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    print(f"Loading base model (bf16) from: {BASE_MODEL}")
    model = VoxtralForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=compute_dtype,
        device_map={"": 0} if device.type == "cuda" else None,
        attn_implementation="sdpa",
    )

    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(
            f"LoRA adapter directory not found: {ADAPTER_DIR}\n"
            f"Expected training output at: {os.path.join(DATA_ROOT, 'voxtral_lora_indic', 'final_model')}"
        )

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    try:
        model = model.merge_and_unload()
        print("LoRA adapter merged into base weights.")
    except Exception as e:
        print(f"[warn] could not merge adapter, running unmerged: {e}")

    model.eval()

    samples = load_test_samples()
    if not samples:
        raise RuntimeError("No test samples found.")
    random.shuffle(samples)

    results_rows: List[Dict] = []
    all_refs, all_hyps = [], []
    per_lang_refs = defaultdict(list)
    per_lang_hyps = defaultdict(list)
    per_lang_char_prf = defaultdict(list)
    per_lang_word_prf = defaultdict(list)

    num_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(total=len(samples), desc="Voxtral ASR", unit="audio")

    for bi in range(num_batches):
        chunk = samples[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        fnames = [s["filename"] for s in chunk]
        print(f"\n[Batch {bi + 1}/{num_batches}] files: {fnames}")

        try:
            hyps = transcribe_batch(model, processor, chunk, device, compute_dtype)
        except torch.cuda.OutOfMemoryError:
            print("  [OOM] skipping batch")
            torch.cuda.empty_cache()
            pbar.update(len(chunk))
            continue
        except Exception as e:
            print(f"  [skip] {e}")
            pbar.update(len(chunk))
            continue

        for s, hyp in zip(chunk, hyps):
            ref_n = normalize_text(s["reference"])
            hyp_n = normalize_text(hyp)

            wer_m = compute_wer_measures(ref_n, hyp_n)
            cer, *_ = compute_cer_measures(ref_n, hyp_n)
            cp, cr, cf = char_prf(hyp_n, ref_n)
            wp, wr, wf = word_prf(hyp_n, ref_n)

            lang = s["lang"]
            row = {
                "filename": s["filename"],
                "lang_code": lang,
                "language": get_language_name(lang),
                "reference": s["reference"],
                "hypothesis": hyp,
                "ref_norm": ref_n,
                "hyp_norm": hyp_n,
                "wer": wer_m["wer"],
                "cer": cer,
                "char_precision": cp, "char_recall": cr, "char_f1": cf,
                "word_precision": wp, "word_recall": wr, "word_f1": wf,
            }
            results_rows.append(row)
            all_refs.append(ref_n)
            all_hyps.append(hyp_n)
            per_lang_refs[lang].append(ref_n)
            per_lang_hyps[lang].append(hyp_n)
            per_lang_char_prf[lang].append((cp, cr, cf))
            per_lang_word_prf[lang].append((wp, wr, wf))

            print(f"  -> lang={get_language_name(lang)} ({lang})  file={s['filename']}")
            print(f"     REF: {s['reference'][:120]}")
            print(f"     HYP: {hyp[:120]}")
            print(
                f"     WER={wer_m['wer']:.3f}  CER={cer:.3f}  "
                f"char P/R/F1={cp:.3f}/{cr:.3f}/{cf:.3f}  "
                f"word P/R/F1={wp:.3f}/{wr:.3f}/{wf:.3f}"
            )

        pbar.update(len(chunk))
    pbar.close()

    if not results_rows:
        print("No results produced.")
        return

    print("\nComputing bootstrap CIs (this may take a moment)...")
    ov_wer_m, ov_wer_s, ov_wer_lo, ov_wer_hi = bootstrap_wer_ci(all_refs, all_hyps)
    ov_cer_m, ov_cer_s, ov_cer_lo, ov_cer_hi = bootstrap_cer_ci(all_refs, all_hyps)
    ov_cp_m, ov_cp_s, ov_cp_lo, ov_cp_hi = bootstrap_metric_ci([r["char_precision"] for r in results_rows])
    ov_cr_m, ov_cr_s, ov_cr_lo, ov_cr_hi = bootstrap_metric_ci([r["char_recall"] for r in results_rows])
    ov_cf_m, ov_cf_s, ov_cf_lo, ov_cf_hi = bootstrap_metric_ci([r["char_f1"] for r in results_rows])
    ov_wp_m, ov_wp_s, ov_wp_lo, ov_wp_hi = bootstrap_metric_ci([r["word_precision"] for r in results_rows])
    ov_wr_m, ov_wr_s, ov_wr_lo, ov_wr_hi = bootstrap_metric_ci([r["word_recall"] for r in results_rows])
    ov_wf_m, ov_wf_s, ov_wf_lo, ov_wf_hi = bootstrap_metric_ci([r["word_f1"] for r in results_rows])

    ci_metrics = {
        "overall": {
            "n": len(results_rows),
            "wer": ov_wer_m, "wer_std": ov_wer_s,
            "wer_ci_lower": ov_wer_lo, "wer_ci_upper": ov_wer_hi,
            "cer": ov_cer_m, "cer_std": ov_cer_s,
            "cer_ci_lower": ov_cer_lo, "cer_ci_upper": ov_cer_hi,
            "char_precision": ov_cp_m, "char_recall": ov_cr_m, "char_f1": ov_cf_m,
            "word_precision": ov_wp_m, "word_recall": ov_wr_m, "word_f1": ov_wf_m,
        },
        "per_language": {}
    }

    for lang in sorted(per_lang_refs.keys()):
        refs_l = per_lang_refs[lang]
        hyps_l = per_lang_hyps[lang]
        wl_m, wl_s, wl_lo, wl_hi = bootstrap_wer_ci(refs_l, hyps_l)
        cl_m, cl_s, cl_lo, cl_hi = bootstrap_cer_ci(refs_l, hyps_l)
        cps = [t[0] for t in per_lang_char_prf[lang]]
        crs = [t[1] for t in per_lang_char_prf[lang]]
        cfs = [t[2] for t in per_lang_char_prf[lang]]
        wps = [t[0] for t in per_lang_word_prf[lang]]
        wrs = [t[1] for t in per_lang_word_prf[lang]]
        wfs = [t[2] for t in per_lang_word_prf[lang]]

        ci_metrics["per_language"][lang] = {
            "n": len(refs_l),
            "wer": wl_m, "wer_std": wl_s, "wer_ci_lower": wl_lo, "wer_ci_upper": wl_hi,
            "cer": cl_m, "cer_std": cl_s, "cer_ci_lower": cl_lo, "cer_ci_upper": cl_hi,
            "char_precision": float(np.mean(cps)),
            "char_recall": float(np.mean(crs)),
            "char_f1": float(np.mean(cfs)),
            "word_precision": float(np.mean(wps)),
            "word_recall": float(np.mean(wrs)),
            "word_f1": float(np.mean(wfs)),
        }

    det_csv = os.path.join(OUTPUT_DIR, "results_detailed.csv")
    pd.DataFrame(results_rows).to_csv(det_csv, index=False, encoding="utf-8")
    print(f"\nSaved {det_csv}")

    per_lang_csv = os.path.join(OUTPUT_DIR, "per_language_results.csv")
    with open(per_lang_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "lang_code", "language", "n",
            "wer", "wer_ci_lo", "wer_ci_hi",
            "cer", "cer_ci_lo", "cer_ci_hi",
            "char_precision", "char_recall", "char_f1",
            "word_precision", "word_recall", "word_f1"
        ])
        for lang, m in ci_metrics["per_language"].items():
            w.writerow([
                lang, get_language_name(lang), m["n"],
                f"{m['wer']:.4f}", f"{m['wer_ci_lower']:.4f}", f"{m['wer_ci_upper']:.4f}",
                f"{m['cer']:.4f}", f"{m['cer_ci_lower']:.4f}", f"{m['cer_ci_upper']:.4f}",
                f"{m['char_precision']:.4f}", f"{m['char_recall']:.4f}", f"{m['char_f1']:.4f}",
                f"{m['word_precision']:.4f}", f"{m['word_recall']:.4f}", f"{m['word_f1']:.4f}",
            ])
    print(f"Saved {per_lang_csv}")

    summary_csv = os.path.join(OUTPUT_DIR, "metrics_summary_with_ci.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "metric", "value", "std", "ci_lower", "ci_upper"])
        ov = ci_metrics["overall"]
        w.writerow(["overall", "wer", f"{ov['wer']:.4f}", f"{ov['wer_std']:.4f}", f"{ov['wer_ci_lower']:.4f}", f"{ov['wer_ci_upper']:.4f}"])
        w.writerow(["overall", "cer", f"{ov['cer']:.4f}", f"{ov['cer_std']:.4f}", f"{ov['cer_ci_lower']:.4f}", f"{ov['cer_ci_upper']:.4f}"])
        for lang, m in ci_metrics["per_language"].items():
            w.writerow([lang, "wer", f"{m['wer']:.4f}", f"{m['wer_std']:.4f}", f"{m['wer_ci_lower']:.4f}", f"{m['wer_ci_upper']:.4f}"])
            w.writerow([lang, "cer", f"{m['cer']:.4f}", f"{m['cer_std']:.4f}", f"{m['cer_ci_lower']:.4f}", f"{m['cer_ci_upper']:.4f}"])
    print(f"Saved {summary_csv}")

    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(ci_metrics, f, indent=2, ensure_ascii=False)

    generate_plots(results_rows, ci_metrics, OUTPUT_DIR)

    report_path = os.path.join(OUTPUT_DIR, "inference_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Voxtral-Mini-3B-2507 (+ LoRA)  |  IndicVoices TEST inference\n")
        f.write("=" * 70 + "\n")
        f.write(f"Base model:   {BASE_MODEL}\n")
        f.write(f"Adapter dir:  {ADAPTER_DIR}\n")
        f.write(f"Total samples: {len(results_rows)}\n\n")

        ov = ci_metrics["overall"]
        f.write("OVERALL (95% bootstrap CI)\n")
        f.write(f"  WER: {ov['wer']:.4f}  [{ov['wer_ci_lower']:.4f}, {ov['wer_ci_upper']:.4f}]\n")
        f.write(f"  CER: {ov['cer']:.4f}  [{ov['cer_ci_lower']:.4f}, {ov['cer_ci_upper']:.4f}]\n")
        f.write(f"  char P/R/F1: {ov['char_precision']:.4f} / {ov['char_recall']:.4f} / {ov['char_f1']:.4f}\n")
        f.write(f"  word P/R/F1: {ov['word_precision']:.4f} / {ov['word_recall']:.4f} / {ov['word_f1']:.4f}\n\n")

        f.write("PER LANGUAGE\n")
        for lang, m in ci_metrics["per_language"].items():
            f.write(f"  [{lang}] {get_language_name(lang)}  (n={m['n']})\n")
            f.write(f"     WER: {m['wer']:.4f}  [{m['wer_ci_lower']:.4f}, {m['wer_ci_upper']:.4f}]\n")
            f.write(f"     CER: {m['cer']:.4f}  [{m['cer_ci_lower']:.4f}, {m['cer_ci_upper']:.4f}]\n")
            f.write(f"     char P/R/F1: {m['char_precision']:.4f} / {m['char_recall']:.4f} / {m['char_f1']:.4f}\n")
            f.write(f"     word P/R/F1: {m['word_precision']:.4f} / {m['word_recall']:.4f} / {m['word_f1']:.4f}\n")
    print(f"Saved {report_path}")

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    ov = ci_metrics["overall"]
    print(f"Samples: {ov['n']}")
    print(f"WER: {ov['wer']:.4f}  [95% CI {ov['wer_ci_lower']:.4f}, {ov['wer_ci_upper']:.4f}]")
    print(f"CER: {ov['cer']:.4f}  [95% CI {ov['cer_ci_lower']:.4f}, {ov['cer_ci_upper']:.4f}]")
    print(f"Char  P/R/F1: {ov['char_precision']:.4f} / {ov['char_recall']:.4f} / {ov['char_f1']:.4f}")
    print(f"Word  P/R/F1: {ov['word_precision']:.4f} / {ov['word_recall']:.4f} / {ov['word_f1']:.4f}")

    print("\nPER-LANGUAGE")
    for lang, m in ci_metrics["per_language"].items():
        print(f"  [{lang}] {get_language_name(lang)}  n={m['n']}")
        print(f"     WER {m['wer']:.4f}  CER {m['cer']:.4f}")
        print(f"     char P/R/F1 {m['char_precision']:.3f}/{m['char_recall']:.3f}/{m['char_f1']:.3f}")
        print(f"     word P/R/F1 {m['word_precision']:.3f}/{m['word_recall']:.3f}/{m['word_f1']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()