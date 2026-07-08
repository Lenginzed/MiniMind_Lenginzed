# -*- coding: utf-8 -*-
"""
Environment audit script for the MiniMInd mini-LLM project.

This script is intentionally read-only:
- it does not install packages
- it does not download models or datasets
- it does not start training
- it only runs local detection commands and a tiny CUDA tensor test when possible
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import json
import locale
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
RAW_OUTPUTS: list[dict[str, Any]] = []


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 2)


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.2f} {units[idx]}"


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def decode_bytes(data: bytes) -> str:
    encodings = ["utf-8", locale.getpreferredencoding(False), "gbk", "cp936", "latin-1"]
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def run_command(label: str, command: list[str] | str, timeout: int = 20) -> dict[str, Any]:
    started = now_iso()
    try:
        completed = subprocess.run(
            command,
            shell=isinstance(command, str),
            capture_output=True,
            timeout=timeout,
        )
        result = {
            "label": label,
            "command": command if isinstance(command, str) else " ".join(command),
            "started_at": started,
            "exit_code": completed.returncode,
            "stdout": decode_bytes(completed.stdout or b""),
            "stderr": decode_bytes(completed.stderr or b""),
        }
    except Exception as exc:
        result = {
            "label": label,
            "command": command if isinstance(command, str) else " ".join(command),
            "started_at": started,
            "exit_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "exception": traceback.format_exc(),
        }
    RAW_OUTPUTS.append(result)
    return result


def run_powershell_json(label: str, script: str, timeout: int = 20) -> Any:
    utf8_prefix = (
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
        "$OutputEncoding=[System.Text.UTF8Encoding]::new(); "
    )
    ps = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        utf8_prefix + script,
    ]
    result = run_command(label, ps, timeout=timeout)
    if result["exit_code"] != 0 or not result["stdout"].strip():
        return None
    return safe_json_loads(result["stdout"])


def metadata_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def collect_basic_commands() -> None:
    commands: list[tuple[str, list[str] | str, int]] = [
        ("where python", "where python", 10),
        ("where pip", "where pip", 10),
        ("python --version", [sys.executable, "--version"], 10),
        ("pip --version", [sys.executable, "-m", "pip", "--version"], 20),
        ("conda info --json", "conda info --json", 30),
        ("conda list", "conda list", 45),
        ("pip list --format=columns", [sys.executable, "-m", "pip", "list", "--format=columns"], 45),
        ("nvidia-smi", "nvidia-smi", 20),
        (
            "nvidia-smi query gpu",
            "nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version --format=csv,noheader,nounits",
            20,
        ),
        ("nvcc --version", "nvcc --version", 20),
        ("systeminfo", "systeminfo", 45),
    ]
    for label, command, timeout in commands:
        run_command(label, command, timeout=timeout)


def collect_os_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if platform.system().lower() == "windows":
        os_info = run_powershell_json(
            "powershell os",
            "Get-CimInstance Win32_OperatingSystem | "
            "Select-Object Caption,Version,BuildNumber,OSArchitecture,InstallDate,LastBootUpTime | "
            "ConvertTo-Json -Depth 4",
        )
        if os_info:
            info["windows_cim"] = os_info
    return info


def collect_cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_processor": platform.processor(),
        "logical_cores": os.cpu_count(),
    }
    if platform.system().lower() == "windows":
        cpu = run_powershell_json(
            "powershell cpu",
            "Get-CimInstance Win32_Processor | "
            "Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize | "
            "ConvertTo-Json -Depth 4",
        )
        if isinstance(cpu, list):
            info["processors"] = cpu
            info["physical_cores"] = sum(int(x.get("NumberOfCores") or 0) for x in cpu)
            info["logical_cores"] = sum(int(x.get("NumberOfLogicalProcessors") or 0) for x in cpu) or info["logical_cores"]
            info["model"] = "; ".join(str(x.get("Name", "")).strip() for x in cpu if x.get("Name"))
        elif isinstance(cpu, dict):
            info["processors"] = [cpu]
            info["physical_cores"] = int(cpu.get("NumberOfCores") or 0)
            info["logical_cores"] = int(cpu.get("NumberOfLogicalProcessors") or info["logical_cores"] or 0)
            info["model"] = str(cpu.get("Name", "")).strip()
    return info


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def collect_memory_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    if platform.system().lower() == "windows":
        try:
            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            info.update(
                {
                    "total_bytes": int(status.ullTotalPhys),
                    "available_bytes": int(status.ullAvailPhys),
                    "total_gib": bytes_to_gib(status.ullTotalPhys),
                    "available_gib": bytes_to_gib(status.ullAvailPhys),
                    "memory_load_percent": int(status.dwMemoryLoad),
                }
            )
        except Exception as exc:
            info["ctypes_error"] = f"{type(exc).__name__}: {exc}"
        ps_mem = run_powershell_json(
            "powershell memory",
            "Get-CimInstance Win32_ComputerSystem | "
            "Select-Object TotalPhysicalMemory,Manufacturer,Model | ConvertTo-Json -Depth 4",
        )
        if ps_mem:
            info["windows_cim"] = ps_mem
    return info


def collect_disk_info() -> list[dict[str, Any]]:
    disks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in [ROOT, Path.cwd(), Path.home(), Path(ROOT.anchor)]:
        try:
            resolved = str(path.resolve())
            usage = shutil.disk_usage(resolved)
            key = str(Path(resolved).anchor or resolved)
            if key in seen:
                continue
            seen.add(key)
            disks.append(
                {
                    "path": resolved,
                    "anchor": key,
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                    "total_gib": bytes_to_gib(usage.total),
                    "free_gib": bytes_to_gib(usage.free),
                }
            )
        except Exception:
            continue
    if platform.system().lower() == "windows":
        logical = run_powershell_json(
            "powershell logical disks",
            "Get-CimInstance Win32_LogicalDisk -Filter \"DriveType=3\" | "
            "Select-Object DeviceID,VolumeName,Size,FreeSpace | ConvertTo-Json -Depth 4",
        )
        if logical:
            if isinstance(logical, dict):
                logical = [logical]
            for item in logical:
                device = str(item.get("DeviceID") or "")
                if not device:
                    continue
                if any(d.get("anchor", "").upper().startswith(device.upper()) for d in disks):
                    continue
                disks.append(
                    {
                        "path": device + "\\",
                        "anchor": device + "\\",
                        "volume_name": item.get("VolumeName"),
                        "total_bytes": item.get("Size"),
                        "free_bytes": item.get("FreeSpace"),
                        "total_gib": bytes_to_gib(item.get("Size")),
                        "free_gib": bytes_to_gib(item.get("FreeSpace")),
                    }
                )
    return disks


def parse_nvidia_smi_query(stdout: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        index, name, mem_total, mem_free, compute_cap, driver_version = parts[:6]
        def to_float(text: str) -> float | None:
            try:
                return float(text)
            except Exception:
                return None
        total_mib = to_float(mem_total)
        free_mib = to_float(mem_free)
        gpus.append(
            {
                "index": index,
                "name": name,
                "memory_total_mib": total_mib,
                "memory_free_mib": free_mib,
                "memory_total_gib": round(total_mib / 1024, 2) if total_mib is not None else None,
                "memory_free_gib": round(free_mib / 1024, 2) if free_mib is not None else None,
                "compute_capability": compute_cap,
                "driver_version": driver_version,
            }
        )
    return gpus


def collect_nvidia_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "nvcc_available": shutil.which("nvcc") is not None,
    }
    smi = run_command("nvidia-smi for parse", "nvidia-smi", timeout=20)
    if smi["exit_code"] == 0:
        text = smi["stdout"]
        info["nvidia_smi_summary"] = text.splitlines()[:20]
        for line in text.splitlines():
            if "Driver Version:" in line and "CUDA Version:" in line:
                try:
                    driver_part = line.split("Driver Version:", 1)[1].split("CUDA Version:", 1)[0].strip().strip("|").strip()
                    cuda_part = line.split("CUDA Version:", 1)[1].strip().strip("|").strip()
                    info["driver_version"] = driver_part
                    info["cuda_version_from_driver"] = cuda_part
                except Exception:
                    pass
    query = run_command(
        "nvidia-smi query gpu for parse",
        "nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version --format=csv,noheader,nounits",
        timeout=20,
    )
    if query["exit_code"] == 0:
        info["gpus"] = parse_nvidia_smi_query(query["stdout"])
    else:
        info["gpus"] = []
    nvcc = run_command("nvcc --version for parse", "nvcc --version", timeout=20)
    if nvcc["exit_code"] == 0:
        info["nvcc_version_output"] = nvcc["stdout"].strip()
    if platform.system().lower() == "windows":
        video = run_powershell_json(
            "powershell video controllers",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion,VideoProcessor | ConvertTo-Json -Depth 4",
        )
        if video:
            info["windows_video_controllers"] = video if isinstance(video, list) else [video]
    return info


def collect_package_info() -> dict[str, Any]:
    targets = {
        "torch": ("torch", "torch"),
        "transformers": ("transformers", "transformers"),
        "datasets": ("datasets", "datasets"),
        "accelerate": ("accelerate", "accelerate"),
        "peft": ("peft", "peft"),
        "trl": ("trl", "trl"),
        "bitsandbytes": ("bitsandbytes", "bitsandbytes"),
        "sentencepiece": ("sentencepiece", "sentencepiece"),
        "tokenizers": ("tokenizers", "tokenizers"),
        "safetensors": ("safetensors", "safetensors"),
        "matplotlib": ("matplotlib", "matplotlib"),
        "wandb": ("wandb", "wandb"),
        "tensorboard": ("tensorboard", "tensorboard"),
        "flash-attn": ("flash_attn", "flash-attn"),
        "vllm": ("vllm", "vllm"),
        "deepspeed": ("deepspeed", "deepspeed"),
    }
    packages: dict[str, Any] = {}
    for display, (module, dist) in targets.items():
        packages[display] = {
            "installed": module_available(module) or metadata_version(dist) is not None,
            "module_available": module_available(module),
            "version": metadata_version(dist),
        }
    return packages


def collect_conda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
        "CONDA_EXE": os.environ.get("CONDA_EXE"),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
    }
    result = run_command("conda info --json for parse", "conda info --json", timeout=30)
    if result["exit_code"] == 0:
        parsed = safe_json_loads(result["stdout"])
        if isinstance(parsed, dict):
            keep = [
                "active_prefix",
                "active_prefix_name",
                "conda_version",
                "default_prefix",
                "envs",
                "env_vars",
                "platform",
                "python_version",
                "root_prefix",
            ]
            info["conda_info"] = {key: parsed.get(key) for key in keep}
    return info


def collect_torch_info() -> dict[str, Any]:
    torch_info: dict[str, Any] = {
        "installed": False,
        "import_error": None,
        "cuda_test": {"attempted": False},
    }
    try:
        import torch  # type: ignore

        torch_info.update(
            {
                "installed": True,
                "version": getattr(torch, "__version__", None),
                "cuda_compiled_version": getattr(torch.version, "cuda", None),
                "hip_compiled_version": getattr(torch.version, "hip", None),
                "debug": getattr(torch.version, "debug", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if hasattr(torch, "cuda") else 0,
                "cudnn_available": bool(torch.backends.cudnn.is_available()) if hasattr(torch.backends, "cudnn") else None,
                "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            }
        )
        devices = []
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                free_bytes = None
                total_bytes = None
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                except Exception:
                    pass
                devices.append(
                    {
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "capability": list(torch.cuda.get_device_capability(idx)),
                        "total_memory_bytes": int(props.total_memory),
                        "total_memory_gib": bytes_to_gib(props.total_memory),
                        "free_memory_bytes": int(free_bytes) if free_bytes is not None else None,
                        "free_memory_gib": bytes_to_gib(free_bytes),
                        "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
                    }
                )
            try:
                torch_info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception as exc:
                torch_info["bf16_supported_error"] = f"{type(exc).__name__}: {exc}"
            torch_info["fp16_supported"] = True
        else:
            torch_info["bf16_supported"] = False
            torch_info["fp16_supported"] = False
        torch_info["cuda_devices"] = devices

        cuda_test: dict[str, Any] = {"attempted": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda:0")
                x = torch.randn((128, 128), device=device)
                y = x @ x.T
                torch.cuda.synchronize()
                cuda_test.update(
                    {
                        "ok": True,
                        "device": torch.cuda.get_device_name(0),
                        "dtype": str(x.dtype),
                        "shape": list(y.shape),
                        "mean": float(y.mean().item()),
                        "allocated_bytes_after_test": int(torch.cuda.memory_allocated(0)),
                        "reserved_bytes_after_test": int(torch.cuda.memory_reserved(0)),
                    }
                )
                try:
                    a = torch.randn((64, 64), device=device, dtype=torch.float16)
                    b = a @ a.T
                    torch.cuda.synchronize()
                    cuda_test["fp16_matmul_ok"] = True
                    cuda_test["fp16_mean"] = float(b.float().mean().item())
                except Exception as exc:
                    cuda_test["fp16_matmul_ok"] = False
                    cuda_test["fp16_error"] = f"{type(exc).__name__}: {exc}"
                try:
                    if torch.cuda.is_bf16_supported():
                        c = torch.randn((64, 64), device=device, dtype=torch.bfloat16)
                        d = c @ c.T
                        torch.cuda.synchronize()
                        cuda_test["bf16_matmul_ok"] = True
                        cuda_test["bf16_mean"] = float(d.float().mean().item())
                    else:
                        cuda_test["bf16_matmul_ok"] = False
                        cuda_test["bf16_error"] = "torch.cuda.is_bf16_supported() returned False"
                except Exception as exc:
                    cuda_test["bf16_matmul_ok"] = False
                    cuda_test["bf16_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                cuda_test.update(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
        torch_info["cuda_test"] = cuda_test
    except Exception as exc:
        torch_info["import_error"] = f"{type(exc).__name__}: {exc}"
        torch_info["import_traceback"] = traceback.format_exc()
    return torch_info


def compute_capability_tuple(value: str | list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return None
    if isinstance(value, str):
        try:
            first, second = value.split(".", 1)
            return int(first), int(second)
        except Exception:
            return None
    return None


def max_gpu_vram_gib(audit: dict[str, Any]) -> float:
    values: list[float] = []
    for gpu in audit.get("torch", {}).get("cuda_devices", []) or []:
        if gpu.get("total_memory_gib") is not None:
            values.append(float(gpu["total_memory_gib"]))
    for gpu in audit.get("nvidia", {}).get("gpus", []) or []:
        if gpu.get("memory_total_gib") is not None:
            values.append(float(gpu["memory_total_gib"]))
    return max(values) if values else 0.0


def best_compute_capability(audit: dict[str, Any]) -> tuple[int, int] | None:
    caps: list[tuple[int, int]] = []
    for gpu in audit.get("torch", {}).get("cuda_devices", []) or []:
        cap = compute_capability_tuple(gpu.get("capability"))
        if cap:
            caps.append(cap)
    for gpu in audit.get("nvidia", {}).get("gpus", []) or []:
        cap = compute_capability_tuple(gpu.get("compute_capability"))
        if cap:
            caps.append(cap)
    return max(caps) if caps else None


def choose_model_preset(vram_gib: float) -> dict[str, Any]:
    presets = [
        {
            "name": "5M",
            "reason": "CPU/very small GPU safe baseline",
            "vocab_size": 8192,
            "context_length": 256,
            "n_layer": 4,
            "n_embd": 192,
            "n_head": 6,
            "n_kv_head": 2,
            "intermediate_size": 512,
            "micro_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "target_tokens": "10M-50M tokens",
            "vram_pressure": "low",
        },
        {
            "name": "15M",
            "reason": "comfortable first real mini-LLM size",
            "vocab_size": 12000,
            "context_length": 512,
            "n_layer": 6,
            "n_embd": 384,
            "n_head": 6,
            "n_kv_head": 2,
            "intermediate_size": 1024,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "target_tokens": "50M-150M tokens",
            "vram_pressure": "low-medium",
        },
        {
            "name": "30M",
            "reason": "good portfolio-scale from-scratch target on a modest CUDA GPU",
            "vocab_size": 16000,
            "context_length": 512,
            "n_layer": 8,
            "n_embd": 512,
            "n_head": 8,
            "n_kv_head": 2,
            "intermediate_size": 1536,
            "micro_batch_size": 6,
            "gradient_accumulation_steps": 8,
            "target_tokens": "100M-300M tokens",
            "vram_pressure": "medium",
        },
        {
            "name": "60M",
            "reason": "upper practical learning target on 8-12 GiB VRAM if using mixed precision and checkpointing",
            "vocab_size": 24000,
            "context_length": 1024,
            "n_layer": 10,
            "n_embd": 640,
            "n_head": 10,
            "n_kv_head": 2,
            "intermediate_size": 1792,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "target_tokens": "200M-600M tokens",
            "vram_pressure": "medium-high",
        },
        {
            "name": "100M",
            "reason": "stretch target; useful for experiments but slower and more memory hungry",
            "vocab_size": 32000,
            "context_length": 1024,
            "n_layer": 12,
            "n_embd": 768,
            "n_head": 12,
            "n_kv_head": 4,
            "intermediate_size": 2304,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "target_tokens": "300M-1B tokens if time permits",
            "vram_pressure": "high",
        },
    ]
    if vram_gib <= 0:
        preset = presets[0].copy()
        preset["selected_reason"] = "No CUDA GPU was confirmed, so keep the first run tiny."
        return preset
    if vram_gib < 4:
        preset = presets[0].copy()
    elif vram_gib < 6:
        preset = presets[1].copy()
    elif vram_gib < 8:
        preset = presets[2].copy()
    elif vram_gib < 12:
        preset = presets[3].copy()
    elif vram_gib < 20:
        preset = presets[3].copy()
        preset["stretch_target"] = presets[4]
        preset["reason"] = "recommended first serious portfolio target; 100M is a later stretch target on 16 GiB VRAM"
    else:
        preset = presets[4].copy()
    preset["selected_reason"] = f"Largest detected GPU memory is about {vram_gib:.2f} GiB."
    return preset


def infer_compatibility(audit: dict[str, Any]) -> dict[str, Any]:
    os_system = str(audit.get("os", {}).get("system") or "")
    is_windows = os_system.lower() == "windows"
    hardware_cuda = bool(audit.get("nvidia", {}).get("gpus"))
    torch_cuda = bool(audit.get("torch", {}).get("cuda_available"))
    vram = max_gpu_vram_gib(audit)
    cap = best_compute_capability(audit)
    cap_major = cap[0] if cap else 0
    bf16_hw = cap_major >= 8
    fp16_hw = cap_major >= 6
    bf16_torch = bool(torch_cuda and audit.get("torch", {}).get("bf16_supported"))
    fp16_torch = bool(torch_cuda and audit.get("torch", {}).get("fp16_supported"))
    return {
        "fp16": {
            "supported": fp16_hw,
            "currently_usable_in_torch": fp16_torch,
            "note": (
                "Hardware supports fp16 and PyTorch CUDA confirmed it."
                if fp16_torch
                else "Hardware appears to support fp16, but the current PyTorch CUDA stack did not confirm it."
                if fp16_hw
                else "No CUDA fp16 path was confirmed."
            ),
        },
        "bf16": {
            "supported": bf16_hw,
            "currently_usable_in_torch": bf16_torch,
            "note": (
                "Hardware supports bf16 and PyTorch CUDA confirmed it."
                if bf16_torch
                else "Hardware should support bf16 on compute capability >= 8.0, but the current PyTorch CUDA stack did not confirm it."
                if bf16_hw
                else "Prefer fp16 or fp32; bf16 hardware support was not confirmed."
            ),
        },
        "bitsandbytes": {
            "fit": "conditional" if hardware_cuda else "not recommended",
            "note": (
                "CUDA GPU exists; fix PyTorch CUDA first. Recent bitsandbytes docs list Windows CUDA builds; use WSL2/Linux if native install fails."
                if hardware_cuda
                else "No CUDA GPU was confirmed, so bitsandbytes has little value."
            ),
        },
        "flash_attn": {
            "fit": "conditional" if (hardware_cuda and cap_major >= 8 and not is_windows) else "not recommended",
            "note": (
                "Best on Linux with Ampere/Ada/Hopper GPUs; use PyTorch SDPA as fallback."
                if hardware_cuda
                else "CUDA is not available."
            ),
        },
        "vllm": {
            "fit": "conditional" if (hardware_cuda and not is_windows and vram >= 8) else "not recommended",
            "note": (
                "Native Windows is not the ideal target; use WSL2/Linux for vLLM experiments."
                if is_windows
                else "Useful mainly for serving larger HF models; overkill for tiny from-scratch models."
            ),
        },
        "deepspeed_fsdp": {
            "fit": "conditional",
            "note": (
                "FSDP/DeepSpeed are educationally useful, but single-GPU mini models do not need them. "
                "DeepSpeed on Windows can be installation-heavy; PyTorch DDP/FSDP is a better later module."
            ),
        },
    }


def recommended_install_versions(audit: dict[str, Any]) -> dict[str, str]:
    torch_cuda = audit.get("torch", {}).get("cuda_compiled_version")
    cuda_hint = torch_cuda or audit.get("nvidia", {}).get("cuda_version_from_driver") or "matching your driver"
    return {
        "torch": (
            "Use the official PyTorch install selector for Windows + CUDA; choose a stable CUDA wheel "
            f"compatible with the driver/runtime ({cuda_hint}). Re-test `import torch` and `torch.cuda.is_available()` afterward."
        ),
        "transformers": "transformers>=4.44, preferably current stable.",
        "datasets": "datasets>=2.20 or current stable.",
        "accelerate": "accelerate>=0.33 or current stable.",
        "peft": "peft>=0.12 or current stable.",
        "trl": "trl>=0.11 or current stable for DPO/GRPO-style APIs; pin once code is written.",
        "bitsandbytes": "Use a recent bitsandbytes release after PyTorch CUDA works; official docs list Windows CUDA builds, but WSL2/Linux remains a good fallback if native wheels fail.",
        "sentencepiece": "sentencepiece>=0.2.0.",
        "tokenizers": "tokenizers version compatible with transformers.",
        "safetensors": "safetensors>=0.4.",
        "matplotlib": "matplotlib>=3.8.",
        "wandb": "wandb is optional; install only if you want cloud experiment tracking.",
        "tensorboard": "tensorboard is enough for the first local training curves.",
        "flash-attn": "Skip on native Windows at first; use PyTorch SDPA or move this experiment to WSL2/Linux.",
        "vllm": "Skip on native Windows at first; use transformers.generate locally, or use WSL2/Linux for vLLM serving experiments.",
        "deepspeed": "Skip until multi-GPU/distributed experiments; FSDP/DDP concepts can be studied later.",
        "wandb/tensorboard": "Install tensorboard for local logging first; add wandb only if you want cloud experiment tracking.",
    }


def create_hardware_markdown(audit: dict[str, Any]) -> str:
    os_info = audit["os"]
    cpu = audit["cpu"]
    mem = audit["memory"]
    nvidia = audit["nvidia"]
    torch_info = audit["torch"]
    packages = audit["packages"]
    compat = audit["compatibility"]
    lines: list[str] = []
    lines.append("# Hardware and Environment Audit")
    lines.append("")
    lines.append(f"Generated at: `{audit['generated_at']}`")
    lines.append(f"Repository root: `{audit['repo_root']}`")
    lines.append("")
    lines.append("## System")
    lines.append(f"- OS: {os_info.get('windows_cim', {}).get('Caption') or os_info.get('platform')}")
    lines.append(f"- OS version/build: {os_info.get('windows_cim', {}).get('Version') or os_info.get('version')} / {os_info.get('windows_cim', {}).get('BuildNumber', 'unknown')}")
    lines.append(f"- Architecture: {os_info.get('windows_cim', {}).get('OSArchitecture') or os_info.get('machine')}")
    lines.append("")
    lines.append("## CPU and Memory")
    lines.append(f"- CPU: {cpu.get('model') or cpu.get('python_processor') or 'unknown'}")
    lines.append(f"- Physical cores: {cpu.get('physical_cores', 'unknown')}")
    lines.append(f"- Logical cores: {cpu.get('logical_cores', 'unknown')}")
    lines.append(f"- RAM total: {mem.get('total_gib', 'unknown')} GiB")
    lines.append(f"- RAM available during audit: {mem.get('available_gib', 'unknown')} GiB")
    lines.append("")
    lines.append("## Disk")
    for disk in audit["disks"]:
        lines.append(f"- {disk.get('anchor')}: total {disk.get('total_gib')} GiB, free {disk.get('free_gib')} GiB")
    lines.append("")
    lines.append("## GPU / CUDA")
    gpus = torch_info.get("cuda_devices") or nvidia.get("gpus") or []
    if gpus:
        for gpu in gpus:
            name = gpu.get("name", "unknown")
            total = gpu.get("total_memory_gib") or gpu.get("memory_total_gib")
            free = gpu.get("free_memory_gib") or gpu.get("memory_free_gib")
            cap = gpu.get("capability") or gpu.get("compute_capability")
            lines.append(f"- GPU {gpu.get('index', 0)}: {name}, VRAM total {total} GiB, free {free} GiB, compute capability {cap}")
    else:
        lines.append("- No NVIDIA CUDA GPU was confirmed.")
    lines.append(f"- NVIDIA Driver: {nvidia.get('driver_version', 'unknown')}")
    lines.append(f"- CUDA reported by NVIDIA driver: {nvidia.get('cuda_version_from_driver', 'unknown')}")
    lines.append(f"- CUDA used by PyTorch build: {torch_info.get('cuda_compiled_version', 'unknown')}")
    lines.append(f"- cuDNN available/version: {torch_info.get('cudnn_available', 'unknown')} / {torch_info.get('cudnn_version', 'unknown')}")
    if torch_info.get("import_error"):
        lines.append(f"- PyTorch import status: failed - {torch_info.get('import_error')}")
    lines.append(f"- torch.cuda available: {torch_info.get('cuda_available', False)}")
    lines.append(f"- CUDA tensor test: {torch_info.get('cuda_test')}")
    lines.append("")
    lines.append("## Precision Support")
    lines.append(
        f"- fp16 hardware support: {compat['fp16']['supported']}; "
        f"currently usable in PyTorch: {compat['fp16'].get('currently_usable_in_torch')} - {compat['fp16']['note']}"
    )
    lines.append(
        f"- bf16 hardware support: {compat['bf16']['supported']}; "
        f"currently usable in PyTorch: {compat['bf16'].get('currently_usable_in_torch')} - {compat['bf16']['note']}"
    )
    lines.append("")
    lines.append("## Python and Packages")
    lines.append(f"- Python executable: `{audit['python']['executable']}`")
    lines.append(f"- Python version: {audit['python']['version']}")
    lines.append(f"- pip version: {audit['python'].get('pip_version', 'unknown')}")
    lines.append(f"- Conda env: {audit['conda'].get('CONDA_DEFAULT_ENV') or audit['conda'].get('conda_info', {}).get('active_prefix_name') or 'unknown'}")
    lines.append("")
    for pkg, data in packages.items():
        if pkg == "torch" and data.get("installed") and data.get("import_ok") is False:
            status = "installed metadata, import failed"
        else:
            status = "installed" if data.get("installed") else "missing"
        version = data.get("version") or "unknown"
        lines.append(f"- {pkg}: {status}, version {version}")
    lines.append("")
    lines.append("## Compatibility Notes")
    for name, item in compat.items():
        if name in {"fp16", "bf16"}:
            continue
        lines.append(f"- {name}: {item['fit']} - {item['note']}")
    lines.append("")
    lines.append("Detailed raw outputs are saved in `audit/raw_command_outputs.txt`; structured data is saved in `audit/env_audit.json`.")
    lines.append("")
    return "\n".join(lines)


def create_recommendation_markdown(audit: dict[str, Any]) -> str:
    preset = audit["model_recommendation"]
    torch_info = audit["torch"]
    compat = audit["compatibility"]
    packages = audit["packages"]
    missing = [name for name, data in packages.items() if not data.get("installed")]
    torch_broken = bool(packages.get("torch", {}).get("installed") and not torch_info.get("installed"))
    if compat["bf16"]["supported"]:
        precision = "bf16 after PyTorch CUDA is fixed; fp32 only for CPU smoke tests until then" if torch_broken else "bf16"
    elif compat["fp16"]["supported"]:
        precision = "fp16 after PyTorch CUDA is fixed; fp32 only for CPU smoke tests until then" if torch_broken else "fp16"
    else:
        precision = "fp32"
    use_checkpointing = preset["name"] in {"60M", "100M"} or max_gpu_vram_gib(audit) < 8
    lines: list[str] = []
    lines.append("# Mini-LLM Project Recommendation")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Recommended from-scratch model size: **{preset['name']}**.")
    if preset.get("stretch_target"):
        lines.append(f"- Later stretch target after the stack is stable: **{preset['stretch_target']['name']}**.")
    lines.append(f"- Recommended precision: **{precision}**.")
    lines.append(f"- Recommended context length for the first full run: **{preset['context_length']}**.")
    lines.append(f"- Gradient checkpointing: **{'yes' if use_checkpointing else 'optional'}**.")
    lines.append(f"- Start Stage 1 now: **yes**, after keeping the first implementation small and testable.")
    if torch_broken:
        lines.append("- Blocking environment issue before GPU training: **PyTorch is installed but fails to import**, so CUDA tensor tests could not run.")
    lines.append("")
    lines.append("## Suggested From-Scratch Config")
    lines.append(f"- vocab_size: {preset['vocab_size']}")
    lines.append(f"- context_length: {preset['context_length']}")
    lines.append(f"- n_layer: {preset['n_layer']}")
    lines.append(f"- n_embd: {preset['n_embd']}")
    lines.append(f"- n_head: {preset['n_head']}")
    lines.append(f"- n_kv_head: {preset['n_kv_head']}")
    lines.append(f"- intermediate_size: {preset['intermediate_size']}")
    lines.append(f"- per-device micro batch size: {preset['micro_batch_size']}")
    lines.append(f"- gradient_accumulation_steps: {preset['gradient_accumulation_steps']}")
    lines.append(f"- expected pretrain token budget: {preset['target_tokens']}")
    lines.append(f"- rough VRAM pressure: {preset['vram_pressure']}")
    lines.append(f"- selection reason: {preset.get('selected_reason', preset.get('reason'))}")
    lines.append("")
    lines.append("## Training Feasibility")
    if torch_broken:
        lines.append("- Before any GPU training, repair/reinstall the PyTorch environment so `import torch` and `torch.cuda.is_available()` work.")
    lines.append("- PreTrain: feasible for learning if dataset size and sequence length are controlled. Start with a smoke test, then 10M tokens, then scale upward.")
    lines.append("- Full-parameter SFT on the mini model: feasible; use the same model and shorter sequence lengths first.")
    lines.append("- LoRA-SFT on the mini model: feasible but less necessary for a tiny model; still valuable as an implementation exercise.")
    lines.append("- DPO: feasible on a mini model with small pairwise batches; memory pressure is higher because policy/reference evaluations are involved.")
    lines.append("- GRPO: feasible only as a tiny toy run at first. Online generation of multiple completions increases VRAM and wall-clock cost substantially.")
    lines.append("- HF small-model engineering baseline: recommended. Use a small instruction model such as Qwen2.5-0.5B-Instruct or Qwen3-0.6B only as an engineering comparison, not as the main from-scratch target.")
    lines.append("")
    lines.append("## Inference and Quantization")
    generate_ok = bool(torch_info.get("installed") and packages.get("transformers", {}).get("installed"))
    lines.append(f"- transformers.generate local inference: {'yes' if generate_ok else 'fix PyTorch and install transformers first'}. This is the recommended baseline.")
    lines.append("- Custom Top-k / Top-p / Temperature decoding: strongly recommended; implement this yourself for the mini model.")
    lines.append(f"- vLLM: {compat['vllm']['fit']} - {compat['vllm']['note']}")
    lines.append("- GPTQ educational demo: feasible on tiny models as an algorithm walkthrough. Do not expect meaningful speedups unless kernels and model size justify it.")
    lines.append("- SmoothQuant educational demo: feasible as a calibration/statistics and weight/activation rescaling exercise on a mini transformer or small HF model.")
    lines.append("- If acceleration libraries are painful on Windows, implement the quantization math and validation on small tensors/models first.")
    lines.append("")
    lines.append("## Library Installation Guidance")
    if torch_broken:
        lines.append(f"- torch: installed according to package metadata, but import failed: {torch_info.get('import_error')}")
    if missing:
        lines.append("Missing or unconfirmed packages:")
        for pkg in missing:
            suggestion = audit["recommended_install_versions"].get(pkg) or audit["recommended_install_versions"].get("wandb/tensorboard")
            lines.append(f"- {pkg}: {suggestion or 'install a current stable version when needed'}")
    else:
        lines.append("- Core packages appear to be installed. Pin versions in `requirements.txt` after the first working stage.")
    lines.append("")
    lines.append("## Suggested Project Stages")
    lines.append("1. Stage 1: tokenizer stub/data loader, config objects, RMSNorm, RoPE, GQA attention, SwiGLU MLP, decoder block, Causal LM forward/loss, generation smoke test.")
    lines.append("2. Stage 2: pretraining loop with checkpointing, TensorBoard/matplotlib curves, resume support, gradient clipping, mixed precision.")
    lines.append("3. Stage 3: SFT and LoRA-SFT with small curated data and evaluation prompts.")
    lines.append("4. Stage 4: DPO on tiny preference data, then GRPO toy run with very small completion counts.")
    lines.append("5. Stage 5: quantization demos for GPTQ/SmoothQuant and comparison notebooks/scripts.")
    lines.append("")
    lines.append("## Main Risks")
    lines.append("- Windows native support for flash-attn/vLLM/DeepSpeed-style stacks may be the biggest friction point; WSL2/Linux is the preferred fallback for those modules.")
    lines.append("- GRPO can become slow quickly because generation is inside the training loop; keep toy settings tiny.")
    lines.append("- Tokenizer vocabulary size dominates parameters at small scales; use 8k-24k vocab first rather than jumping to 32k by default.")
    lines.append("- Keep the first model intentionally small so correctness, curves, checkpointing, and evaluation are easy to debug.")
    lines.append("")
    lines.append("## External Compatibility References")
    lines.append("- PyTorch official install selector: https://pytorch.org/get-started/locally/")
    lines.append("- vLLM GPU install docs: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/")
    lines.append("- bitsandbytes installation docs: https://huggingface.co/docs/bitsandbytes/en/installation")
    lines.append("- flash-attention official repository: https://github.com/Dao-AILab/flash-attention")
    lines.append("- DeepSpeed advanced install docs: https://www.deepspeed.ai/tutorials/advanced-install/")
    lines.append("- DeepSpeed Windows notes: https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/windows/08-2024/README.md")
    lines.append("")
    return "\n".join(lines)


def write_raw_outputs() -> None:
    path = AUDIT_DIR / "raw_command_outputs.txt"
    chunks: list[str] = []
    chunks.append(f"Generated at: {now_iso()}\n")
    for item in RAW_OUTPUTS:
        chunks.append("=" * 100)
        chunks.append(f"LABEL: {item.get('label')}")
        chunks.append(f"COMMAND: {item.get('command')}")
        chunks.append(f"STARTED_AT: {item.get('started_at')}")
        chunks.append(f"EXIT_CODE: {item.get('exit_code')}")
        chunks.append("--- STDOUT ---")
        chunks.append(item.get("stdout") or "")
        chunks.append("--- STDERR ---")
        chunks.append(item.get("stderr") or "")
        if item.get("exception"):
            chunks.append("--- EXCEPTION ---")
            chunks.append(item["exception"])
        chunks.append("")
    path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    collect_basic_commands()

    pip_result = run_command("pip --version for parse", [sys.executable, "-m", "pip", "--version"], timeout=20)
    audit: dict[str, Any] = {
        "generated_at": now_iso(),
        "repo_root": str(ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "architecture": platform.architecture(),
            "pip_version": pip_result["stdout"].strip() if pip_result["exit_code"] == 0 else None,
        },
        "os": collect_os_info(),
        "cpu": collect_cpu_info(),
        "memory": collect_memory_info(),
        "disks": collect_disk_info(),
        "nvidia": collect_nvidia_info(),
        "conda": collect_conda_info(),
        "packages": collect_package_info(),
    }
    audit["torch"] = collect_torch_info()
    if "torch" in audit["packages"]:
        audit["packages"]["torch"]["import_ok"] = bool(audit["torch"].get("installed"))
        audit["packages"]["torch"]["import_error"] = audit["torch"].get("import_error")
    audit["compatibility"] = infer_compatibility(audit)
    audit["recommended_install_versions"] = recommended_install_versions(audit)
    audit["model_recommendation"] = choose_model_preset(max_gpu_vram_gib(audit))

    (AUDIT_DIR / "env_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (AUDIT_DIR / "hardware_audit.md").write_text(create_hardware_markdown(audit), encoding="utf-8")
    (AUDIT_DIR / "recommendation.md").write_text(create_recommendation_markdown(audit), encoding="utf-8")
    write_raw_outputs()

    print(f"Audit complete. Files written to: {AUDIT_DIR}")
    print(f"Recommended model size: {audit['model_recommendation']['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
