# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.trainer import run_pretrain


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tiny Causal LM pretrain smoke.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max_steps override.")
    args = parser.parse_args()
    summary = run_pretrain(args.config, resume_path=args.resume, max_steps_override=args.max_steps)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
