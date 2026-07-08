# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Windows/Anaconda can load duplicate OpenMP runtimes when matplotlib imports
# numerical backends. This plot-only process does not run training kernels.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.utils import ensure_dir, iter_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot train/eval loss curves from metrics.jsonl.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [row for row in iter_jsonl(args.metrics) if row.get("train_loss") is not None]
    if not rows:
        raise ValueError("metrics file is empty")
    steps = [row["step"] for row in rows]
    train_loss = [row.get("train_loss") for row in rows]
    eval_steps = [row["step"] for row in rows if row.get("eval_loss") is not None]
    eval_loss = [row.get("eval_loss") for row in rows if row.get("eval_loss") is not None]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, label="train loss", linewidth=1.5)
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval loss", marker="o", linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Tiny Pretrain Smoke Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    ensure_dir(str(Path(args.out).parent))
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("wrote:", args.out)
    print("points:", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
