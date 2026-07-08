# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.utils import ensure_dir, save_json


def _probe_dataset_worker(
    result_queue: mp.Queue,
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    field: str,
    max_samples: int,
) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    started = time.perf_counter()
    try:
        from datasets import load_dataset  # type: ignore

        dataset = (
            load_dataset(dataset_name, config_name, split=split, streaming=True, trust_remote_code=False)
            if config_name
            else load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=False)
        )
        sample_count = 0
        nonempty_count = 0
        first_sample: Dict[str, Any] = {}
        fields = []
        chars = 0
        for item in dataset:
            if sample_count == 0:
                first_sample = {str(k): str(v)[:300] for k, v in dict(item).items()}
                fields = sorted(str(k) for k in dict(item).keys())
            sample_count += 1
            text = str(dict(item).get(field, "") or "").strip()
            if text:
                nonempty_count += 1
                chars += len(text)
            if sample_count >= max_samples:
                break
        result_queue.put(
            {
                "dataset": dataset_name,
                "config": config_name,
                "split": split,
                "field": field,
                "ok": sample_count > 0,
                "sample_count": sample_count,
                "nonempty_count": nonempty_count,
                "chars": chars,
                "fields": fields,
                "first_sample": first_sample,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "error": None,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "dataset": dataset_name,
                "config": config_name,
                "split": split,
                "field": field,
                "ok": False,
                "sample_count": 0,
                "nonempty_count": 0,
                "chars": 0,
                "fields": [],
                "first_sample": {},
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "error": repr(exc),
            }
        )


def run_probe(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    field: str,
    max_samples: int,
    timeout_sec: float,
) -> Dict[str, Any]:
    result_queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_probe_dataset_worker,
        args=(result_queue, dataset_name, config_name, split, field, max_samples),
    )
    process.start()
    process.join(timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return {
            "dataset": dataset_name,
            "config": config_name,
            "split": split,
            "field": field,
            "ok": False,
            "timed_out": True,
            "timeout_sec": timeout_sec,
            "error": f"probe exceeded {timeout_sec} seconds and was terminated",
        }
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = {
            "dataset": dataset_name,
            "config": config_name,
            "split": split,
            "field": field,
            "ok": False,
            "timed_out": False,
            "error": f"probe process exited with code {process.exitcode} without result",
        }
    result["timed_out"] = False
    result["timeout_sec"] = timeout_sec
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative Stage 8 public data recovery probe.")
    parser.add_argument("--timeout-sec", type=float, default=90.0)
    parser.add_argument("--wikitext-samples", type=int, default=50)
    parser.add_argument("--alpaca-samples", type=int, default=1000)
    args = parser.parse_args()

    ensure_dir("audit_stage8")
    probes = [
        run_probe(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            "train",
            "text",
            int(args.wikitext_samples),
            float(args.timeout_sec),
        ),
        run_probe(
            "tatsu-lab/alpaca",
            None,
            "train",
            "output",
            int(args.alpaca_samples),
            float(args.timeout_sec),
        ),
    ]
    success = [item for item in probes if item.get("ok")]
    data = {
        "python_executable": sys.executable,
        "probe_timeout_sec": args.timeout_sec,
        "probes": probes,
        "any_public_success": bool(success),
        "successful_datasets": [item["dataset"] for item in success],
        "manual_download_guidance": {
            "pretrain": {
                "preferred": "Salesforce/wikitext, config=wikitext-2-raw-v1, split=train, field=text",
                "output": "data/stage8_public/raw/pretrain_public.txt",
            },
            "sft": {
                "preferred": "tatsu-lab/alpaca, split=train, fields=instruction/input/output",
                "output": "data/stage8_public/sft/sft_train.jsonl and sft_val.jsonl",
            },
            "dpo": {
                "preferred": "tatsu-lab/alpaca_farm preference config, fields instruction/input plus chosen/rejected or output_1/output_2/preference",
                "output": "data/stage8_public/dpo/dpo_train.jsonl and dpo_val.jsonl",
            },
        },
    }
    save_json(data, "audit_stage8/public_recovery_probe.json")
    lines = [
        "# Stage 8 Public Recovery Probe",
        "",
        f"- Python executable: `{sys.executable}`",
        f"- Per-dataset timeout: `{args.timeout_sec}` seconds",
        f"- Any public dataset success: `{data['any_public_success']}`",
        f"- Successful datasets: `{data['successful_datasets']}`",
        "",
        "## Probe Results",
        "",
    ]
    for item in probes:
        lines.extend(
            [
                f"### {item.get('dataset')}",
                "",
                f"- config: `{item.get('config')}`",
                f"- split: `{item.get('split')}`",
                f"- field: `{item.get('field')}`",
                f"- ok: `{item.get('ok')}`",
                f"- timed_out: `{item.get('timed_out')}`",
                f"- sample_count: `{item.get('sample_count')}`",
                f"- nonempty_count: `{item.get('nonempty_count')}`",
                f"- fields: `{item.get('fields')}`",
                f"- error: `{item.get('error')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Manual Recovery Notes",
            "",
            "- Pretrain text should become `data/stage8_public/raw/pretrain_public.txt`, one cleaned text segment per line or paragraph.",
            "- Alpaca SFT should be converted to JSONL rows with `instruction`, `input`, `output`, `category`.",
            "- Preference data should be converted to JSONL rows with `instruction`, `input`, `chosen`, `rejected`, `category`, and `rejected_type` or `reason`.",
            "- After manually placing public data, rerun tokenizer/tokenization and then start Stage 8 public long-run.",
            "",
        ]
    )
    Path("audit_stage8/public_recovery_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
