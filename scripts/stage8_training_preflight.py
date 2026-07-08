# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.config import MiniLLMConfig
from minillm.model import MiniLLMForCausalLM, count_parameters
from minillm.tokenizer import MiniTokenizer
from minillm.utils import ensure_dir, iter_jsonl, load_json, load_yaml, save_json


def nvidia_smi() -> Dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
        gpus = []
        for line in output.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    {
                        "name": parts[0],
                        "memory_total_mib": int(parts[1]),
                        "memory_used_mib": int(parts[2]),
                        "memory_free_mib": int(parts[3]),
                    }
                )
        return {"ok": True, "gpus": gpus, "raw": output}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def read_jsonl_count(path: str, limit: int = 3) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    rows: List[Dict[str, Any]] = []
    total = 0
    for row in iter_jsonl(path):
        total += 1
        if len(rows) < limit:
            rows.append(row)
    return {"exists": True, "count": total, "sample": rows}


def npy_info(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    arr = np.load(path, mmap_mode="r")
    return {"exists": True, "shape": list(arr.shape), "dtype": str(arr.dtype), "tokens": int(arr.size)}


def output_status(path: str) -> Dict[str, Any]:
    p = Path(path)
    metrics = p / "metrics.jsonl"
    return {
        "path": str(p),
        "exists": p.exists(),
        "metrics_exists": metrics.exists(),
        "metrics_size": metrics.stat().st_size if metrics.exists() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 8 public long-run training preflight.")
    parser.add_argument("--out", default="audit_stage8/stage8_training_preflight.md")
    args = parser.parse_args()

    ensure_dir("audit_stage8")
    model_cfg_path = "configs/stage8_public/model_50m_public.yaml"
    pretrain_cfg_path = "configs/stage8_public/pretrain_public_long.yaml"
    model_cfg = load_yaml(model_cfg_path)
    model = MiniLLMForCausalLM(MiniLLMConfig(**model_cfg))
    param_count = count_parameters(model)
    del model

    tokenizer = MiniTokenizer.load("data/stage8_public/tokenizers/public_tokenizer.json")
    disk = shutil.disk_usage(str(ROOT))
    smi = nvidia_smi()
    pretrain_cfg = load_yaml(pretrain_cfg_path)
    outputs = {
        "pretrain": output_status(pretrain_cfg["output_dir"]),
        "sft_full": output_status("outputs/stage8_public/sft_public_full"),
        "sft_lora": output_status("outputs/stage8_public/sft_public_lora"),
        "dpo_full": output_status("outputs/stage8_public/dpo_public_full"),
        "dpo_lora": output_status("outputs/stage8_public/dpo_public_lora"),
        "grpo_full": output_status("outputs/stage8_public/grpo_public_full"),
        "grpo_lora": output_status("outputs/stage8_public/grpo_public_lora"),
    }
    token_meta = load_json("data/stage8_public/processed_pretrain/metadata.json")
    dataset_meta = load_json("data/stage8_public/dataset_metadata.json")

    data = {
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "gpu": smi,
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "tokenizer": tokenizer.summary(),
        "train_npy": npy_info("data/stage8_public/processed_pretrain/train.npy"),
        "val_npy": npy_info("data/stage8_public/processed_pretrain/val.npy"),
        "sft_train": read_jsonl_count("data/stage8_public/sft/sft_train.jsonl"),
        "sft_val": read_jsonl_count("data/stage8_public/sft/sft_val.jsonl"),
        "dpo_train": read_jsonl_count("data/stage8_public/dpo/dpo_train.jsonl"),
        "dpo_val": read_jsonl_count("data/stage8_public/dpo/dpo_val.jsonl"),
        "model_config_path": model_cfg_path,
        "parameter_count": int(param_count),
        "pretrain_config_path": pretrain_cfg_path,
        "tokenization_metadata": token_meta,
        "dataset_metadata_summary": {
            "pretrain": dataset_meta.get("pretrain"),
            "sft": dataset_meta.get("sft"),
            "dpo": dataset_meta.get("dpo"),
            "grpo": dataset_meta.get("grpo"),
        },
        "output_status": outputs,
        "recommendation": {
            "pretrain_batch_size": 4,
            "pretrain_gradient_accumulation_steps": 8,
            "context_length": 256,
            "pretrain_max_steps": 1500,
            "sft_max_steps": 1500,
            "dpo_max_steps": 1000,
            "grpo_max_steps": 150,
            "dtype": "bf16",
        },
    }
    save_json(data, "audit_stage8/stage8_training_preflight.json")

    lines = [
        "# Stage 8 Training Preflight",
        "",
        f"- Python executable: `{sys.executable}`",
        f"- Torch: `{torch.__version__}`, CUDA: `{torch.version.cuda}`",
        f"- CUDA available: `{data['cuda_available']}`",
        f"- bf16 supported: `{data['bf16_supported']}`",
        f"- GPU: `{smi}`",
        f"- Disk free: `{data['disk_free_gb']}` GB / `{data['disk_total_gb']}` GB",
        f"- Public tokenizer vocab_size: `{tokenizer.vocab_size}`",
        f"- Train npy: `{data['train_npy']}`",
        f"- Val npy: `{data['val_npy']}`",
        f"- SFT train/val rows: `{data['sft_train'].get('count')}` / `{data['sft_val'].get('count')}`",
        f"- DPO train/val rows: `{data['dpo_train'].get('count')}` / `{data['dpo_val'].get('count')}`",
        f"- Model parameter count: `{param_count}`",
        "",
        "## Output Status",
        "",
    ]
    for name, status in outputs.items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Use the 50M-tier public config with gradient checkpointing.",
            "- Start with pretrain batch_size=4 and grad_accum=8 in bf16.",
            "- If OOM occurs, reduce batch_size to 2 and increase grad_accum to 16, then resume/restart with an explicit audit note.",
            "- Do not overwrite existing Stage 7 outputs.",
            "",
        ]
    )
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
