"""Best validation epoch, wall-clock time, active step time and peak memory from the training logs.

Active step time sums the gaps between consecutive logged training steps that are
at most 60 s (longer gaps are validation/checkpointing). Memory is the largest
logged torch.cuda.memory_reserved() value (Voxtral logged memory_allocated() only).
"""
import os
import pandas as pd

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
for run in sorted(os.listdir(LOGS)):
    t = pd.read_csv(os.path.join(LOGS, run, "training_log.csv"))
    v = pd.read_csv(os.path.join(LOGS, run, "validation_log.csv"))
    v = v[v.epoch >= 1]
    best = v.loc[v.val_loss.idxmin()]
    ep = int(-(-float(best.epoch) // 1))
    msg = f"{run:24s} best val loss {best.val_loss:.4f} (epoch {ep}), final {v.val_loss.iloc[-1]:.4f}"
    if "elapsed_sec" in t:
        gaps = t.elapsed_sec.diff().dropna()
        msg += f" | wall {t.elapsed_sec.iloc[-1]/3600:.2f} h, active {gaps[gaps <= 60].sum()/3600:.2f} h"
    mem = "gpu_mem_reserved_gb" if "gpu_mem_reserved_gb" in t else "gpu_mem_gb"
    msg += f" | peak {mem} {t[mem].max():.2f}"
    print(msg)
