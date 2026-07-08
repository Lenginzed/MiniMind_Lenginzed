# -*- coding: utf-8 -*-
"""
Verify the YSJAirCombat Python environment for the MiniMInd mini-LLM project.

Recommended command:
    D:\anaconda3\envs\YSJAirCombat\python.exe scripts\verify_ysj_env.py

Optional audit output:
    D:\anaconda3\envs\YSJAirCombat\python.exe scripts\verify_ysj_env.py --write-audit
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit_ysj"
RAW_OUTPUTS: List[Dict[str, Any]] = []


MINIMAL_PACKAGES = {
    "numpy": ("numpy", "numpy"),
    "tqdm": ("tqdm", "tqdm"),
    "matplotlib": ("matplotlib", "matplotlib"),
    "tensorboard": ("tensorboard", "tensorboard"),
    "pyyaml": ("yaml", "PyYAML"),
    "pytest": ("pytest", "pytest"),
    "safetensors": ("safetensors", "safetensors"),
}

DEFERRED_PACKAGES = {
    "transformers": ("transformers", "transformers"),
    "datasets": ("datasets", "datasets"),
    "accelerate": ("accelerate", "accelerate"),
    "peft": ("peft", "peft"),
    "trl": ("trl", "trl"),
    "bitsandbytes": ("bitsandbytes", "bitsandbytes"),
}


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def gib(num_bytes: Optional[int]) -> Optional[float]:
    if num_bytes is None:
        return None
    return round(float(num_bytes) / (1024 ** 3), 2)


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def package_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def run_command(label: str, command: List[str], timeout: int = 30) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        result = {
            "label": label,
            "command": " ".join(command),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        result = {
            "label": label,
            "command": " ".join(command),
            "exit_code": None,
            "stdout": "",
            "stderr": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }
    RAW_OUTPUTS.append(result)
    return result


def collect_raw_commands() -> None:
    run_command("python --version", [sys.executable, "--version"])
    run_command("pip --version", [sys.executable, "-m", "pip", "--version"])
    run_command(
        "python torch summary",
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print(torch.__version__); "
                "print(torch.version.cuda); "
                "print(torch.cuda.is_available()); "
                "print(torch.cuda.device_count()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
            ),
        ],
        timeout=60,
    )
    run_command(
        "nvidia-smi query gpu",
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )


def collect_packages() -> Dict[str, Any]:
    packages: Dict[str, Any] = {"minimal": {}, "deferred": {}}
    for name, (module_name, dist_name) in MINIMAL_PACKAGES.items():
        packages["minimal"][name] = {
            "installed": module_available(module_name),
            "version": package_version(dist_name),
            "module": module_name,
            "distribution": dist_name,
        }
    for name, (module_name, dist_name) in DEFERRED_PACKAGES.items():
        packages["deferred"][name] = {
            "installed": module_available(module_name),
            "version": package_version(dist_name),
            "module": module_name,
            "distribution": dist_name,
        }
    return packages


def matmul_test(torch_module: Any, dtype: Any, label: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"attempted": True, "label": label, "dtype": str(dtype)}
    try:
        device = torch_module.device("cuda")
        a = torch_module.randn((128, 128), device=device, dtype=dtype)
        b = torch_module.randn((128, 128), device=device, dtype=dtype)
        c = a @ b
        torch_module.cuda.synchronize()
        result.update(
            {
                "ok": True,
                "shape": list(c.shape),
                "mean": float(c.float().mean().item()),
                "isfinite": bool(torch_module.isfinite(c.float()).all().item()),
            }
        )
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def collect_torch() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "import_ok": False,
        "cuda_tests": {},
        "sdpa": {},
        "devices": [],
    }
    try:
        import torch
        import torch.nn.functional as F

        info.update(
            {
                "import_ok": True,
                "version": torch.__version__,
                "cuda_compiled_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cudnn_available": bool(torch.backends.cudnn.is_available()),
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
        info["sdpa"] = {
            "scaled_dot_product_attention_exists": hasattr(F, "scaled_dot_product_attention"),
        }
        if hasattr(torch.backends, "cuda"):
            for attr in ["flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"]:
                fn = getattr(torch.backends.cuda, attr, None)
                if callable(fn):
                    try:
                        info["sdpa"][attr] = bool(fn())
                    except Exception as exc:
                        info["sdpa"][attr] = "%s: %s" % (type(exc).__name__, exc)

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                free_bytes = None
                total_bytes = None
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                except Exception:
                    pass
                info["devices"].append(
                    {
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "capability": list(torch.cuda.get_device_capability(idx)),
                        "total_memory_bytes": int(props.total_memory),
                        "total_memory_gib": gib(int(props.total_memory)),
                        "free_memory_bytes": int(free_bytes) if free_bytes is not None else None,
                        "free_memory_gib": gib(int(free_bytes)) if free_bytes is not None else None,
                    }
                )
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            info["cuda_tests"]["fp32"] = matmul_test(torch, torch.float32, "fp32")
            info["cuda_tests"]["fp16"] = matmul_test(torch, torch.float16, "fp16")
            if torch.cuda.is_bf16_supported():
                info["cuda_tests"]["bf16"] = matmul_test(torch, torch.bfloat16, "bf16")
            else:
                info["cuda_tests"]["bf16"] = {
                    "attempted": False,
                    "ok": False,
                    "reason": "torch.cuda.is_bf16_supported() returned False",
                }
        else:
            info["bf16_supported"] = False
            for label in ["fp32", "fp16", "bf16"]:
                info["cuda_tests"][label] = {
                    "attempted": False,
                    "ok": False,
                    "reason": "torch.cuda.is_available() returned False",
                }
    except Exception as exc:
        info.update(
            {
                "import_ok": False,
                "import_error": "%s: %s" % (type(exc).__name__, exc),
                "import_traceback": traceback.format_exc(),
            }
        )
    return info


def collect_audit() -> Dict[str, Any]:
    collect_raw_commands()
    pip_result = run_command("pip --version for parse", [sys.executable, "-m", "pip", "--version"])
    packages = collect_packages()
    torch_info = collect_torch()
    minimal_missing = [
        name for name, data in packages["minimal"].items() if not data.get("installed")
    ]
    deferred_missing = [
        name for name, data in packages["deferred"].items() if not data.get("installed")
    ]
    stage1_ok = bool(torch_info.get("import_ok"))
    gpu_pretrain_ok = bool(
        torch_info.get("import_ok")
        and torch_info.get("cuda_available")
        and torch_info.get("cuda_tests", {}).get("fp32", {}).get("ok")
        and torch_info.get("cuda_tests", {}).get("fp16", {}).get("ok")
    )
    return {
        "generated_at": now_text(),
        "repo_root": str(ROOT),
        "target_python": sys.executable,
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "pip_version": pip_result["stdout"].strip() if pip_result["exit_code"] == 0 else None,
        },
        "torch": torch_info,
        "packages": packages,
        "suitability": {
            "stage1_cpu_shape_unit_tests": stage1_ok,
            "stage1_note": (
                "Suitable for Stage 1 smoke/shape/unit tests."
                if stage1_ok
                else "Not suitable until torch imports successfully."
            ),
            "gpu_pretrain": gpu_pretrain_ok,
            "gpu_pretrain_note": (
                "Suitable for later small GPU pretrain after minimal dependencies are present."
                if gpu_pretrain_ok
                else "Not suitable for GPU pretrain until torch/cuda tests pass."
            ),
            "needs_minimal_packages": minimal_missing,
            "missing_deferred_packages": deferred_missing,
        },
        "model_scale_recommendation": {
            "tiny_smoke_config": {
                "vocab_size": 128,
                "context_length": 32,
                "n_layer": 2,
                "n_embd": 64,
                "n_head": 4,
                "n_kv_head": 2,
                "intermediate_size": 128,
            },
            "main_target_config": {
                "vocab_size": 20000,
                "context_length": 512,
                "n_layer": 10,
                "n_embd": 576,
                "n_head": 9,
                "n_kv_head": 3,
                "intermediate_size": 1728,
                "rough_parameters": "about 50.2M with tied embeddings in the current skeleton",
                "micro_batch_size": "4 for pretrain/SFT at ctx=512; use 1-2 for DPO",
                "gradient_accumulation_steps": "8-16",
                "vram_pressure": "medium for pretrain/SFT; high for DPO/GRPO if completions or sequence length grow",
            },
            "stretch_target": {
                "size": "60M",
                "note": "Current suggested stretch shape is about 59.6M; use only after Stage 1/2/3 are stable and GRPO/DPO memory behavior is understood.",
            },
        },
    }


def write_raw_outputs(path: Path) -> None:
    parts: List[str] = ["Generated at: %s\n" % now_text()]
    for item in RAW_OUTPUTS:
        parts.append("=" * 100)
        parts.append("LABEL: %s" % item.get("label"))
        parts.append("COMMAND: %s" % item.get("command"))
        parts.append("EXIT_CODE: %s" % item.get("exit_code"))
        parts.append("--- STDOUT ---")
        parts.append(item.get("stdout") or "")
        parts.append("--- STDERR ---")
        parts.append(item.get("stderr") or "")
        if item.get("traceback"):
            parts.append("--- TRACEBACK ---")
            parts.append(item["traceback"])
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def build_hardware_md(audit: Dict[str, Any]) -> str:
    torch_info = audit["torch"]
    packages = audit["packages"]
    lines = [
        "# YSJAirCombat Environment Audit",
        "",
        "Generated at: `%s`" % audit["generated_at"],
        "",
        "## Target Environment",
        "- Python executable: `%s`" % audit["python"]["executable"],
        "- Python version: %s" % audit["python"]["version"],
        "- pip: %s" % audit["python"].get("pip_version"),
        "",
        "## PyTorch / CUDA",
        "- torch import ok: %s" % markdown_bool(torch_info.get("import_ok")),
        "- torch version: %s" % torch_info.get("version", "unknown"),
        "- torch.version.cuda: %s" % torch_info.get("cuda_compiled_version", "unknown"),
        "- torch.cuda.is_available(): %s" % torch_info.get("cuda_available", False),
        "- cuDNN: %s / %s" % (torch_info.get("cudnn_available"), torch_info.get("cudnn_version")),
        "",
        "## GPU",
    ]
    if torch_info.get("devices"):
        for device in torch_info["devices"]:
            lines.append(
                "- GPU %s: %s, VRAM %.2f GiB total, %.2f GiB free, capability %s"
                % (
                    device["index"],
                    device["name"],
                    device.get("total_memory_gib") or 0.0,
                    device.get("free_memory_gib") or 0.0,
                    device.get("capability"),
                )
            )
    else:
        lines.append("- No CUDA GPU visible to PyTorch.")
    lines.extend(["", "## CUDA Matmul Tests"])
    for name in ["fp32", "fp16", "bf16"]:
        item = torch_info.get("cuda_tests", {}).get(name, {})
        lines.append("- %s: %s; details: `%s`" % (name, markdown_bool(item.get("ok")), item))
    lines.extend(["", "## PyTorch SDPA"])
    for key, value in torch_info.get("sdpa", {}).items():
        lines.append("- %s: %s" % (key, value))
    lines.extend(["", "## Minimal Packages"])
    for name, data in packages["minimal"].items():
        lines.append(
            "- %s: %s, version %s"
            % (name, "installed" if data.get("installed") else "missing", data.get("version"))
        )
    lines.extend(["", "## Deferred Packages"])
    for name, data in packages["deferred"].items():
        lines.append(
            "- %s: %s, version %s"
            % (name, "installed" if data.get("installed") else "missing", data.get("version"))
        )
    lines.extend(
        [
            "",
            "## Suitability",
            "- Stage 1 CPU/shape/unit tests: %s. %s"
            % (
                markdown_bool(audit["suitability"]["stage1_cpu_shape_unit_tests"]),
                audit["suitability"]["stage1_note"],
            ),
            "- Later GPU pretrain: %s. %s"
            % (
                markdown_bool(audit["suitability"]["gpu_pretrain"]),
                audit["suitability"]["gpu_pretrain_note"],
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_recommendation_md(audit: Dict[str, Any]) -> str:
    missing_minimal = audit["suitability"]["needs_minimal_packages"]
    missing_deferred = audit["suitability"]["missing_deferred_packages"]
    tiny = audit["model_scale_recommendation"]["tiny_smoke_config"]
    main = audit["model_scale_recommendation"]["main_target_config"]
    lines = [
        "# YSJAirCombat Recommendation",
        "",
        "## Summary",
        "- Use `D:\\anaconda3\\envs\\YSJAirCombat\\python.exe` for this project stage.",
        "- Do not use the Anaconda base environment because its PyTorch import is broken.",
        "- Stage 1 should use the tiny smoke config only; do not test the 40M-50M config yet.",
        "- First full from-scratch target is reduced to about 40M-50M for a complete, stable training/post-training loop.",
        "",
        "## Environment Readiness",
        "- Stage 1 CPU/shape/unit tests: %s" % markdown_bool(audit["suitability"]["stage1_cpu_shape_unit_tests"]),
        "- Later GPU pretrain: %s" % markdown_bool(audit["suitability"]["gpu_pretrain"]),
        "- Missing minimal packages: %s" % (", ".join(missing_minimal) if missing_minimal else "none"),
        "- Deferred missing packages: %s" % (", ".join(missing_deferred) if missing_deferred else "none"),
        "",
        "## Minimal Dependency Status",
    ]
    for name, data in audit["packages"]["minimal"].items():
        lines.append(
            "- %s: %s%s"
            % (
                name,
                "已满足" if data.get("installed") else "需补充",
                " (%s)" % data.get("version") if data.get("version") else "",
            )
        )
    lines.extend(
        [
            "",
            "Install only missing minimal packages when needed, using the target interpreter:",
            "",
            "```powershell",
            "& 'D:\\anaconda3\\envs\\YSJAirCombat\\python.exe' -m pip install -r requirements-minimal.txt",
            "```",
            "",
            "Do not install transformers/datasets/peft/trl/bitsandbytes/vLLM/DeepSpeed/flash-attn for Stage 1.",
            "bitsandbytes should wait until Stage 3/QLoRA evaluation.",
            "",
            "## Tiny Smoke Config",
        ]
    )
    for key, value in tiny.items():
        lines.append("- %s: %s" % (key, value))
    lines.extend(["", "## Main Target Config: about 40M-50M"])
    for key, value in main.items():
        lines.append("- %s: %s" % (key, value))
    lines.extend(
        [
            "",
            "Rationale: 16GB VRAM is enough for a larger model, but this project must also cover SFT, LoRA, DPO, GRPO, inference, visualization, and quantization. DPO and especially GRPO are more expensive than ordinary pretrain/SFT because they require paired/reference passes or online generation.",
            "",
            "## 60M Stretch Target",
            "- Keep the previous 60M idea as a stretch target after Stage 1/2/3 are stable.",
            "- Suggested stretch shape: vocab_size=24000, context_length=1024, n_layer=10, n_embd=640, n_head=10, n_kv_head=2, intermediate_size=1792.",
            "- Use gradient checkpointing and smaller micro batches for DPO/GRPO.",
            "",
            "## Current Blockers",
        ]
    )
    if missing_minimal:
        lines.append("- Missing minimal packages: %s." % ", ".join(missing_minimal))
    else:
        lines.append("- No minimal package blocker for Stage 1.")
    if "pytest" in missing_minimal:
        lines.append("- `python -m pytest -q` will not run until pytest is installed; smoke scripts can still run.")
    return "\n".join(lines) + "\n"


def write_audit_files(audit: Dict[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "env_audit_ysj.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (AUDIT_DIR / "hardware_audit_ysj.md").write_text(build_hardware_md(audit), encoding="utf-8")
    (AUDIT_DIR / "recommendation_ysj.md").write_text(build_recommendation_md(audit), encoding="utf-8")
    write_raw_outputs(AUDIT_DIR / "raw_command_outputs_ysj.txt")


def print_summary(audit: Dict[str, Any]) -> None:
    torch_info = audit["torch"]
    print("Python executable:", audit["python"]["executable"])
    print("Python version:", audit["python"]["version"])
    print("torch import ok:", torch_info.get("import_ok"))
    if torch_info.get("import_ok"):
        print("torch version:", torch_info.get("version"))
        print("torch.version.cuda:", torch_info.get("cuda_compiled_version"))
        print("torch.cuda.is_available():", torch_info.get("cuda_available"))
        for device in torch_info.get("devices", []):
            print(
                "GPU %s: %s, total=%.2f GiB, free=%.2f GiB"
                % (
                    device["index"],
                    device["name"],
                    device.get("total_memory_gib") or 0.0,
                    device.get("free_memory_gib") or 0.0,
                )
            )
        print("fp32 matmul:", torch_info.get("cuda_tests", {}).get("fp32"))
        print("fp16 matmul:", torch_info.get("cuda_tests", {}).get("fp16"))
        print("bf16 supported:", torch_info.get("bf16_supported"))
        print("bf16 matmul:", torch_info.get("cuda_tests", {}).get("bf16"))
        print("PyTorch SDPA:", torch_info.get("sdpa"))
    else:
        print("torch import error:", torch_info.get("import_error"))
    print("Stage 1 suitable:", audit["suitability"]["stage1_cpu_shape_unit_tests"])
    print("GPU pretrain suitable:", audit["suitability"]["gpu_pretrain"])
    print("Missing minimal packages:", audit["suitability"]["needs_minimal_packages"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify YSJAirCombat environment.")
    parser.add_argument("--write-audit", action="store_true", help="Write audit_ysj report files.")
    args = parser.parse_args()
    try:
        audit = collect_audit()
        print_summary(audit)
        if args.write_audit:
            write_audit_files(audit)
            print("Audit files written to:", AUDIT_DIR)
        return 0
    except Exception as exc:
        print("Environment verification failed with an unexpected error:")
        print("%s: %s" % (type(exc).__name__, exc))
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
