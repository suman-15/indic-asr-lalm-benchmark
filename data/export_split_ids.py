"""Export the file IDs of each split from the local IndicVoices subset.

Usage:
    python data/export_split_ids.py /path/to/IndicVoices_subset_4lang_combined

Reads <root>/<split>/<split>.csv (columns: path, lang, text) and writes
data/<split>_ids.csv with only the file name and language code.
Audio and transcripts are not exported; obtain them from IndicVoices.
"""
import os
import sys
import pandas as pd

root = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
for split in ["train", "validation", "test"]:
    src = os.path.join(root, split, f"{split}.csv")
    if not os.path.exists(src):
        print(f"skip {split}: {src} not found")
        continue
    d = pd.read_csv(src)
    out = pd.DataFrame({"file": d["path"].astype(str).map(os.path.basename), "lang": d["lang"]})
    out.to_csv(os.path.join(here, f"{split}_ids.csv"), index=False)
    print(f"{split}: {len(out)} utterances -> data/{split}_ids.csv")
