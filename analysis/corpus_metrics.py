"""Per-language corpus-level WER/CER of the fine-tuned systems (Table 2, FT values)."""
from common import SPEC, LANGS, load, corpus

print(f"{'Model':28s}" + "".join(f"{l:>16s}" for l in LANGS + ["Overall"]))
for name in SPEC:
    d = load(name)
    cells = [corpus(d[d.lang == l]) for l in LANGS] + [corpus(d)]
    print(f"{name:28s}" + "".join(f"{w:8.3f}/{c:.3f}" for w, c in cells))
