# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.dpo_trainer import run_dpo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full DPO or DPO-LoRA smoke training.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = run_dpo(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
