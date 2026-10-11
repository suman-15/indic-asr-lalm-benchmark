"""Load the fine-tuned per-utterance predictions and compute corpus-level WER/CER.

Corpus WER = sum_i(WER_i * N_i) / sum_i(N_i), with N_i the number of reference
words of utterance i (CER: reference characters). The per-utterance WER/CER
columns are the ones written by each model's own inference script.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "finetuned")

# name: (file, separator, key column, language column, reference column, wer, cer)
SPEC = {
    "Whisper-Large-v3": ("whisper_large_v3/inference_results.csv", ",", "filename", "actual_language", "reference_text", "WER", "CER"),
    "SeamlessM4T-v2-Large": ("seamless_m4t_v2_large/test_asr_results.csv", ",", "file", "lang_actual_name", "transcription_ref", "WER", "CER"),
    "MMS-1B-All": ("mms_1b_all/inference_results.csv", ",", "filename", "language", "reference_text", "WER", "CER"),
    "IndicConformer-600M": ("indicconformer_600m/inference_results.csv", ",", "filename", "actual_language", "reference_text", "WER", "CER"),
    "Qwen2-Audio-7B-Instruct": ("qwen2_audio_7b/results_detailed.csv", ",", "file", "language", "ground_truth", "wer", "cer"),
    "Qwen3-ASR-1.7B": ("qwen3_asr_1.7b/results.tsv", "\t", "file", "actual_language", "ground_truth", "wer", "cer"),
    "Phi-4-Multimodal-Instruct": ("phi4_multimodal/phi4_inference_results.csv", ",", "file", "language", "ground_truth", "wer", "cer"),
    "Voxtral-Mini-3B-2507": ("voxtral_mini_3b/results_detailed.csv", ",", "filename", "language", "ref_norm", "wer", "cer"),
}
LANGS = ["Assamese", "Bengali", "Hindi", "Odia"]


def load(name):
    path, sep, key, lang, ref, wer, cer = SPEC[name]
    d = pd.read_csv(os.path.join(ROOT, path), sep=sep)
    out = pd.DataFrame({
        "key": d[key].astype(str).map(os.path.basename),
        "lang": d[lang].astype(str),
        "ref": d[ref].fillna("").astype(str),
        "wer": d[wer].astype(float),
        "cer": d[cer].astype(float),
    })
    out["N"] = out.ref.str.split().str.len()
    # SeamlessM4T-v2's own scoring script excludes spaces from CER.
    out["C"] = out.ref.str.replace(" ", "").str.len() if name.startswith("Seamless") else out.ref.str.len()
    out["we"] = out.wer * out.N
    out["ce"] = out.cer * out.C
    return out.set_index("key")


def corpus(df):
    return df.we.sum() / df.N.sum(), df.ce.sum() / df.C.sum()
