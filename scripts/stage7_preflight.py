# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.config import MiniLLMConfig
from minillm.model import MiniLLMForCausalLM, count_parameters
from minillm.tokenizer import MiniTokenizer
from minillm.utils import ensure_dir, save_json


def run_cmd(args: List[str]) -> Dict[str, Any]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=15, encoding="utf-8", errors="replace")
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def nvidia_smi_query() -> Dict[str, Any]:
    query = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = run_cmd(["nvidia-smi"])
    data: Dict[str, Any] = {"raw_query": query, "raw_processes": processes}
    if query.get("ok") and query.get("stdout"):
        parts = [part.strip() for part in query["stdout"].splitlines()[0].split(",")]
        if len(parts) >= 4:
            data.update(
                {
                    "gpu_name": parts[0],
                    "memory_total_mb": int(float(parts[1])),
                    "memory_free_mb": int(float(parts[2])),
                    "driver_version": parts[3],
                }
            )
    return data


def bf16_matmul_test() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"ok": False, "reason": "cuda unavailable"}
    try:
        device = torch.device("cuda")
        a = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        b = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        return {"ok": bool(torch.isfinite(c.float()).all().item()), "shape": list(c.shape), "dtype": str(c.dtype)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def load_artifact_checks() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    tokenizer_path = Path("data/tokenizers/mixed_tokenizer.json")
    checkpoint_path = Path("outputs/sft_full/checkpoints/best.pt")
    checks["tokenizer_path"] = str(tokenizer_path)
    checks["checkpoint_path"] = str(checkpoint_path)
    try:
        tokenizer = MiniTokenizer.load(str(tokenizer_path))
        checks["tokenizer_ok"] = True
        checks["tokenizer_vocab_size"] = tokenizer.vocab_size
    except Exception as exc:
        checks["tokenizer_ok"] = False
        checks["tokenizer_error"] = repr(exc)
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        cfg = MiniLLMConfig(**ckpt["model_config"])
        model = MiniLLMForCausalLM(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        checks["checkpoint_ok"] = True
        checks["checkpoint_model_config"] = ckpt["model_config"]
        checks["checkpoint_params"] = count_parameters(model)
    except Exception as exc:
        checks["checkpoint_ok"] = False
        checks["checkpoint_error"] = repr(exc)
    return checks


def recommend_tier(memory_free_mb: int) -> Dict[str, Any]:
    if memory_free_mb >= 13000:
        return {
            "recommended_tier": "50M",
            "model_config": "configs/stage7/model_50m.yaml",
            "context_length": 256,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "notes": "Enough free memory for a 40M-50M educational run with bf16 and gradient checkpointing.",
        }
    if memory_free_mb >= 9000:
        return {
            "recommended_tier": "30M",
            "model_config": "configs/stage7/model_20m.yaml",
            "context_length": 256,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "notes": "Moderate free memory; prefer the 20M-30M fallback tier.",
        }
    return {
        "recommended_tier": "20M",
        "model_config": "configs/stage7/model_20m.yaml",
        "context_length": 256,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "notes": "Low free memory; use conservative batch size and resume-friendly shorter segments.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7 hardware/env preflight.")
    parser.add_argument("--out", default="audit_stage7/preflight_report.md")
    args = parser.parse_args()

    ensure_dir(str(Path(args.out).parent))
    nvidia = nvidia_smi_query()
    disk = shutil.disk_usage(Path.cwd())
    cuda_available = torch.cuda.is_available()
    torch_gpu = {}
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        torch_gpu = {
            "name": props.name,
            "total_memory_mb": int(props.total_memory / 1024 / 1024),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "allocated_mb": int(torch.cuda.memory_allocated() / 1024 / 1024),
            "reserved_mb": int(torch.cuda.memory_reserved() / 1024 / 1024),
        }
    free_mb = int(nvidia.get("memory_free_mb") or torch_gpu.get("total_memory_mb") or 0)
    recommendation = recommend_tier(free_mb)
    artifact_checks = load_artifact_checks()
    bf16_test = bf16_matmul_test()
    env = {
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "torch_gpu": torch_gpu,
        "nvidia_smi": nvidia,
        "bf16_matmul_test": bf16_test,
        "disk": {
            "total_gb": disk.total / 1024 / 1024 / 1024,
            "free_gb": disk.free / 1024 / 1024 / 1024,
            "used_gb": disk.used / 1024 / 1024 / 1024,
        },
        "artifact_checks": artifact_checks,
        "recommendation": recommendation,
        "risks": [
            "OOM risk increases with 50M model, DPO reference model, GRPO online generation, and larger context length.",
            "Long runs may be interrupted; use checkpoint last.pt and --resume for pretrain.",
            "Public dataset download may fail; Stage 7 dataset script falls back to local synthetic corpus.",
        ],
    }
    save_json(env, "audit_stage7/preflight_env.json")
    report = [
        "# Stage 7 Preflight Report",
        "",
        f"- Python executable: `{sys.executable}`",
        f"- Torch: `{torch.__version__}` CUDA `{torch.version.cuda}`",
        f"- CUDA available: `{cuda_available}`",
        f"- GPU: `{torch_gpu.get('name', nvidia.get('gpu_name', 'unknown'))}`",
        f"- GPU total/free memory: `{torch_gpu.get('total_memory_mb', nvidia.get('memory_total_mb', 'unknown'))}` / `{nvidia.get('memory_free_mb', 'unknown')}` MB",
        f"- bf16 supported: `{torch_gpu.get('bf16_supported', False)}`",
        f"- bf16 matmul test: `{bf16_test}`",
        f"- Disk free: `{env['disk']['free_gb']:.2f}` GB",
        f"- Tokenizer check: `{artifact_checks.get('tokenizer_ok')}` vocab `{artifact_checks.get('tokenizer_vocab_size')}`",
        f"- Checkpoint check: `{artifact_checks.get('checkpoint_ok')}` params `{artifact_checks.get('checkpoint_params')}`",
        "",
        "## Recommendation",
        "",
        f"- Model tier: `{recommendation['recommended_tier']}`",
        f"- Model config: `{recommendation['model_config']}`",
        f"- Context length: `{recommendation['context_length']}`",
        f"- Batch size / grad accum: `{recommendation['batch_size']}` / `{recommendation['gradient_accumulation_steps']}`",
        f"- Notes: {recommendation['notes']}",
        "",
        "## GPU Process Snapshot",
        "",
        "```text",
        str(nvidia.get("raw_processes", {}).get("stdout", ""))[:6000],
        "```",
        "",
        "## Risks",
        "",
        "\n".join("- " + item for item in env["risks"]),
        "",
    ]
    Path(args.out).write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(env, indent=2, ensure_ascii=False))
    print("wrote:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
