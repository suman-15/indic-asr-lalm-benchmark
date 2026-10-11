"""Paired utterance-level bootstrap (2,000 resamples, seed 42), as in the paper's appendix.

For each resample, the same utterance indices are drawn for both systems and
corpus WER/CER are recomputed. Per-language tests resample within that language.
The 95% interval is the 2.5th-97.5th percentile of the resampled differences.
"""
import numpy as np
from common import load

N_BOOT, SEED = 2000, 42
COMPARISONS = [
    ("Phi-4-Multimodal-Instruct", "MMS-1B-All", "Overall"),
    ("Qwen2-Audio-7B-Instruct", "MMS-1B-All", "Overall"),
    ("Voxtral-Mini-3B-2507", "MMS-1B-All", "Overall"),
    ("IndicConformer-600M", "Phi-4-Multimodal-Instruct", "Overall"),
] + [("Phi-4-Multimodal-Instruct", "MMS-1B-All", l) for l in ["Assamese", "Bengali", "Hindi", "Odia"]]

cache = {}
def get(name):
    if name not in cache:
        cache[name] = load(name)
    return cache[name]

for a, b, scope in COMPARISONS:
    A, B = get(a), get(b)
    keys = sorted(set(A.index) & set(B.index))
    A, B = A.loc[keys], B.loc[keys]
    idx = np.arange(len(keys)) if scope == "Overall" else np.where(A.lang.values == scope)[0]
    rng = np.random.default_rng(SEED)
    dw, dc = [], []
    for _ in range(N_BOOT):
        k = idx[rng.integers(0, len(idx), len(idx))]
        dw.append(A.we.values[k].sum() / A.N.values[k].sum() - B.we.values[k].sum() / B.N.values[k].sum())
        dc.append(A.ce.values[k].sum() / A.C.values[k].sum() - B.ce.values[k].sum() / B.C.values[k].sum())
    pw = A.we.values[idx].sum() / A.N.values[idx].sum() - B.we.values[idx].sum() / B.N.values[idx].sum()
    pc = A.ce.values[idx].sum() / A.C.values[idx].sum() - B.ce.values[idx].sum() / B.C.values[idx].sum()
    lw, hw = np.percentile(dw, [2.5, 97.5]); lc, hc = np.percentile(dc, [2.5, 97.5])
    print(f"{a} - {b} [{scope}]: WER {pw:+.4f} [{lw:+.4f}, {hw:+.4f}]  CER {pc:+.4f} [{lc:+.4f}, {hc:+.4f}]")
