# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.metadata
import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.utils import ensure_dir, save_json


URLS = [
    "https://huggingface.co",
    "https://huggingface.co/api/datasets/Salesforce/wikitext",
    "https://huggingface.co/api/datasets/tatsu-lab/alpaca",
]

ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "HF_ENDPOINT",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
]


def version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def redact(value: str | None) -> str | None:
    if not value:
        return value
    redacted = value
    if "@" in redacted and "://" in redacted:
        scheme, rest = redacted.split("://", 1)
        host = rest.split("@", 1)[-1]
        redacted = f"{scheme}://***:***@{host}"
    for key in ["token", "password", "passwd", "secret", "apikey", "api_key"]:
        lower = redacted.lower()
        idx = lower.find(key)
        if idx >= 0:
            redacted = redacted[:idx] + key + "=***"
    return redacted


def urllib_check(url: str, timeout: float = 12.0) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MiniLLM-stage8-network-diagnose"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read(300)
        return {
            "ok": True,
            "status": getattr(response, "status", None),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "body_head": body.decode("utf-8", errors="replace")[:200],
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": repr(exc),
        }


def requests_check(url: str, timeout: float = 12.0) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        import requests  # type: ignore

        response = requests.get(url, timeout=timeout, headers={"User-Agent": "MiniLLM-stage8-network-diagnose"})
        return {
            "ok": response.ok,
            "status": response.status_code,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "body_head": response.text[:200],
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": repr(exc),
        }


def hf_api_check(dataset_id: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        from huggingface_hub import HfApi  # type: ignore

        info = HfApi().dataset_info(dataset_id, timeout=12)
        return {
            "ok": True,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "id": getattr(info, "id", dataset_id),
            "sha": getattr(info, "sha", None),
            "tags": list(getattr(info, "tags", []) or [])[:20],
            "siblings_count": len(getattr(info, "siblings", []) or []),
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": repr(exc),
        }


def cache_info() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        import datasets  # type: ignore

        cache = Path(str(datasets.config.HF_DATASETS_CACHE))
        result["hf_datasets_cache"] = str(cache)
    except Exception as exc:
        cache = Path(os.environ.get("HF_DATASETS_CACHE") or Path.home() / ".cache" / "huggingface" / "datasets")
        result["hf_datasets_cache_error"] = repr(exc)
        result["hf_datasets_cache"] = str(cache)
    partials = []
    if cache.exists():
        for pattern in ["*.lock", "*.incomplete", "*.tmp", "*partial*", "*.arrow.incomplete"]:
            partials.extend(str(p) for p in cache.rglob(pattern))
            if len(partials) >= 50:
                break
    result["cache_exists"] = cache.exists()
    result["partial_or_lock_files_sample"] = partials[:50]
    return result


def main() -> int:
    ensure_dir("audit_stage8")
    env = {key: redact(os.environ.get(key)) for key in ENV_KEYS}
    data: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "versions": {
            "datasets": version("datasets"),
            "huggingface_hub": version("huggingface_hub"),
            "requests": version("requests"),
            "certifi": version("certifi"),
        },
        "environment": env,
        "ssl": {
            "default_verify_paths": ssl.get_default_verify_paths()._asdict(),
            "openssl_version": ssl.OPENSSL_VERSION,
        },
        "urllib": {url: urllib_check(url) for url in URLS},
        "requests": {url: requests_check(url) for url in URLS},
        "huggingface_hub_api": {
            "Salesforce/wikitext": hf_api_check("Salesforce/wikitext"),
            "tatsu-lab/alpaca": hf_api_check("tatsu-lab/alpaca"),
            "roneneldan/TinyStories": hf_api_check("roneneldan/TinyStories"),
        },
        "cache": cache_info(),
    }
    save_json(data, "audit_stage8/network_diagnose.json")

    lines = [
        "# Stage 8 Network Diagnose",
        "",
        f"- Python executable: `{data['python_executable']}`",
        f"- datasets version: `{data['versions']['datasets']}`",
        f"- huggingface_hub version: `{data['versions']['huggingface_hub']}`",
        f"- requests version: `{data['versions']['requests']}`",
        f"- OpenSSL: `{data['ssl']['openssl_version']}`",
        f"- HF datasets cache: `{data['cache'].get('hf_datasets_cache')}`",
        f"- Cache exists: `{data['cache'].get('cache_exists')}`",
        f"- Partial/cache lock sample count: `{len(data['cache'].get('partial_or_lock_files_sample') or [])}`",
        "",
        "## Environment Variables",
        "",
    ]
    for key, value in env.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## URL Checks", ""])
    for url in URLS:
        lines.append(f"- urllib `{url}`: `{data['urllib'][url]}`")
        lines.append(f"- requests `{url}`: `{data['requests'][url]}`")
    lines.extend(["", "## Hugging Face Hub API", ""])
    for name, result in data["huggingface_hub_api"].items():
        lines.append(f"- `{name}`: `{result}`")
    lines.extend(["", "## SSL", "", f"```json\n{json.dumps(data['ssl'], indent=2, ensure_ascii=False)}\n```", ""])
    Path("audit_stage8/network_diagnose.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
