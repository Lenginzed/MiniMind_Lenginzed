# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.utils import ensure_dir, save_json


TARGET_DATASETS = [
    ("roneneldan/TinyStories", None),
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("tatsu-lab/alpaca", None),
    ("tatsu-lab/alpaca_farm", None),
    ("OpenAssistant/oasst1", None),
]


def check_url(url: str, timeout: float = 8.0) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "MiniLLM-stage8-preflight"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "ok": True,
                "status": getattr(response, "status", None),
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }


def dataset_license_from_info(info: Dict[str, Any]) -> Any:
    card_data = info.get("cardData") or info.get("card_data")
    if isinstance(card_data, dict):
        return card_data.get("license") or card_data.get("licenses")
    tags = info.get("tags") or []
    if isinstance(tags, list):
        license_tags = [tag for tag in tags if isinstance(tag, str) and tag.startswith("license:")]
        return license_tags
    return None


def check_dataset(name: str, config: str | None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": name,
        "config": config,
        "accessible": False,
        "license": None,
        "configs": None,
        "error": None,
    }
    try:
        quoted = urllib.parse.quote(name, safe="/")
        request = urllib.request.Request(
            f"https://huggingface.co/api/datasets/{quoted}",
            headers={"User-Agent": "MiniLLM-stage8-preflight"},
        )
        with urllib.request.urlopen(request, timeout=12.0) as response:
            info = json.loads(response.read().decode("utf-8", errors="replace"))
        result["accessible"] = True
        result["description_head"] = str(info.get("description") or "")[:400]
        result["license"] = dataset_license_from_info(info)
        siblings = info.get("siblings") or []
        result["siblings_count"] = len(siblings) if isinstance(siblings, list) else None
        result["tags"] = (info.get("tags") or [])[:30] if isinstance(info.get("tags"), list) else None
        result["last_modified"] = info.get("lastModified")
        result["downloads"] = info.get("downloads")
        result["likes"] = info.get("likes")
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 8 public dataset preflight.")
    parser.add_argument("--out", default="audit_stage8/public_data_preflight.md")
    args = parser.parse_args()

    ensure_dir("audit_stage8")
    datasets_installed = importlib.util.find_spec("datasets") is not None
    hub_installed = importlib.util.find_spec("huggingface_hub") is not None
    network = check_url("https://huggingface.co")
    disk = shutil.disk_usage(str(ROOT))
    hostname = socket.gethostname()
    cache_path = None
    if datasets_installed:
        try:
            import datasets  # type: ignore

            cache_path = str(datasets.config.HF_DATASETS_CACHE)
        except Exception as exc:
            cache_path = f"unavailable: {exc!r}"
    else:
        cache_path = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME")

    dataset_results: List[Dict[str, Any]] = []
    if datasets_installed and network.get("ok"):
        for name, config in TARGET_DATASETS:
            dataset_results.append(check_dataset(name, config))
    else:
        for name, config in TARGET_DATASETS:
            dataset_results.append(
                {
                    "name": name,
                    "config": config,
                    "accessible": False,
                    "error": "datasets missing or Hugging Face network check failed",
                }
            )

    env = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "hostname": hostname,
        "datasets_installed": datasets_installed,
        "huggingface_hub_installed": hub_installed,
        "huggingface_network": network,
        "hf_datasets_cache": cache_path,
        "hf_home": os.environ.get("HF_HOME"),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "targets": dataset_results,
    }
    save_json(env, "audit_stage8/public_data_preflight.json")

    lines = [
        "# Stage 8 Public Data Preflight",
        "",
        f"- Python executable: `{sys.executable}`",
        f"- datasets importable: `{datasets_installed}`",
        f"- huggingface_hub importable: `{hub_installed}`",
        f"- Hugging Face network: `{network}`",
        f"- HF datasets cache: `{cache_path}`",
        f"- Disk free: `{env['disk_free_gb']}` GB / `{env['disk_total_gb']}` GB",
        "",
        "## Target Dataset Access",
        "",
    ]
    for item in dataset_results:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- config: `{item.get('config')}`",
                f"- accessible: `{item.get('accessible')}`",
                f"- license: `{item.get('license')}`",
                f"- splits: `{item.get('splits')}`",
                f"- configs: `{item.get('configs')}`",
                f"- error: `{item.get('error')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendation",
            "",
            "- If TinyStories or WikiText access succeeds, use it for public pretraining.",
            "- If Alpaca access succeeds, use it for public SFT.",
            "- If Alpaca Farm preference conversion fails, fallback to Stage 7 synthetic DPO and mark fallback explicitly.",
            "- Do not download large models or unbounded dataset splits in Stage 8.",
            "",
        ]
    )
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(env, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
