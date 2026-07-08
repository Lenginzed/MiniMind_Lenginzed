# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.utils import ensure_dir, save_json


USER_AGENT = "MiniLLM-stage8-public-downloader"
HF = "https://huggingface.co"


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def clean_segment(text: str) -> Optional[str]:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = []
    for line in text.splitlines():
        line = " ".join(line.strip().split())
        if len(line) < 20:
            continue
        if "\ufffd" in line:
            continue
        lines.append(line)
    if not lines:
        return None
    out = "\n".join(lines).strip()
    if len(out) < 40:
        return None
    return out


def hf_api_dataset(repo_id: str, timeout_sec: float) -> Dict[str, Any]:
    url = f"{HF}/api/datasets/{urllib.parse.quote(repo_id, safe='/')}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def siblings(repo_id: str, timeout_sec: float) -> List[str]:
    data = hf_api_dataset(repo_id, timeout_sec)
    return [str(item.get("rfilename")) for item in data.get("siblings", []) if item.get("rfilename")]


def resolve_url(repo_id: str, filename: str) -> str:
    return f"{HF}/datasets/{repo_id}/resolve/main/{urllib.parse.quote(filename, safe='/')}"


def download_url(url: str, path: Path, timeout_sec: float, max_bytes: Optional[int] = None) -> Dict[str, Any]:
    ensure_dir(str(path.parent))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    total = 0
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_sec) as response, path.open("wb") as out:
        while True:
            chunk_size = 1024 * 1024
            if max_bytes is not None:
                remaining = max_bytes - total
                if remaining <= 0:
                    break
                chunk_size = min(chunk_size, remaining)
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    return {"path": str(path), "bytes": total, "elapsed_sec": round(time.perf_counter() - started, 3), "url": url}


def run_with_retries(
    label: str,
    fn: Callable[[], Dict[str, Any]],
    max_retries: int,
    timeout_sec: float,
    sleep_sec: float,
    log: List[Dict[str, Any]],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    started = time.perf_counter()
    max_dataset_sec = 300.0
    for attempt in range(1, max_retries + 1):
        if time.perf_counter() - started > max_dataset_sec:
            log.append({"label": label, "attempt": attempt, "ok": False, "error": "dataset time budget exceeded"})
            break
        try:
            result = fn()
            log.append({"label": label, "attempt": attempt, "ok": True, "result": result})
            return True, result
        except Exception as exc:
            log.append({"label": label, "attempt": attempt, "ok": False, "error": repr(exc)})
            if attempt < max_retries:
                time.sleep(sleep_sec)
    return False, None


def read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore

    table = pq.read_table(path)
    return table.to_pylist()


def try_datasets_pretrain(dataset: str, config: Optional[str], split: str, field: str, target_bytes: int, timeout_sec: float) -> Dict[str, Any]:
    started = time.perf_counter()
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset, config, split=split, streaming=True, trust_remote_code=False) if config else load_dataset(dataset, split=split, streaming=True, trust_remote_code=False)
    rows = []
    total = 0
    for item in ds:
        text = clean_segment(str(item.get(field, "")))
        if not text:
            continue
        rows.append(text)
        total += len((text + "\n").encode("utf-8"))
        if total >= target_bytes or time.perf_counter() - started > timeout_sec:
            break
    if not rows:
        raise RuntimeError("datasets streaming returned no usable rows")
    return {"rows": rows, "bytes": total, "samples": len(rows), "method": "datasets_streaming"}


def direct_wikitext_pretrain(target_bytes: int, cache_dir: Path, timeout_sec: float) -> Dict[str, Any]:
    repo = "Salesforce/wikitext"
    filenames = siblings(repo, timeout_sec)
    candidates = [f for f in filenames if f.startswith("wikitext-2-raw-v1/train") and f.endswith(".parquet")]
    if not candidates:
        raise RuntimeError(f"no wikitext-2 raw train parquet found; siblings={filenames[:20]}")
    local = cache_dir / candidates[0].replace("/", "__")
    info = download_url(resolve_url(repo, candidates[0]), local, timeout_sec)
    rows = read_parquet_rows(local)
    text_rows = []
    total = 0
    for item in rows:
        text = clean_segment(str(item.get("text", "")))
        if not text:
            continue
        text_rows.append(text)
        total += len((text + "\n").encode("utf-8"))
        if total >= target_bytes:
            break
    if not text_rows:
        raise RuntimeError("downloaded wikitext parquet but found no usable text")
    return {"rows": text_rows, "bytes": total, "samples": len(text_rows), "method": "urllib_parquet", "download": info, "source_file": candidates[0]}


def direct_tinystories_pretrain(target_bytes: int, cache_dir: Path, timeout_sec: float) -> Dict[str, Any]:
    repo = "roneneldan/TinyStories"
    filenames = siblings(repo, timeout_sec)
    filename = "TinyStories-train.txt" if "TinyStories-train.txt" in filenames else "TinyStories-valid.txt"
    max_download = min(max(target_bytes * 2, 5 * 1024 * 1024), 120 * 1024 * 1024)
    local = cache_dir / filename
    info = download_url(resolve_url(repo, filename), local, timeout_sec, max_bytes=max_download)
    text_rows = []
    total = 0
    with local.open("r", encoding="utf-8", errors="replace") as f:
        buffer: List[str] = []
        for line in f:
            if line.strip():
                buffer.append(line.strip())
            elif buffer:
                text = clean_segment(" ".join(buffer))
                buffer = []
                if text:
                    text_rows.append(text)
                    total += len((text + "\n").encode("utf-8"))
                if total >= target_bytes:
                    break
        if buffer and total < target_bytes:
            text = clean_segment(" ".join(buffer))
            if text:
                text_rows.append(text)
                total += len((text + "\n").encode("utf-8"))
    if not text_rows:
        raise RuntimeError("downloaded TinyStories text but found no usable text")
    return {"rows": text_rows, "bytes": total, "samples": len(text_rows), "method": "urllib_text", "download": info, "source_file": filename}


def save_pretrain(rows: List[str], out_dir: Path, source: str, config: Optional[str], split: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    path = out_dir / "raw" / "pretrain_public.txt"
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.strip() + "\n")
    meta = {
        "success": True,
        "fallback": False,
        "source_dataset": source,
        "config": config,
        "split": split,
        "path": str(path),
        "samples": len(rows),
        "chars": sum(len(row) for row in rows),
        "bytes": path.stat().st_size,
        "lines": sum(1 for _ in path.open("r", encoding="utf-8")),
        **extra,
    }
    save_json(meta, str(out_dir / "raw" / "pretrain_public_metadata.json"))
    return meta


def download_pretrain(args, cache_dir: Path, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_bytes = int(float(args.pretrain_target_mb) * 1024 * 1024)
    order = ["wikitext", "tinystories"] if args.prefer == "wikitext" else ["tinystories", "wikitext"]
    failures = []
    for source in order:
        if source == "wikitext":
            dataset, config, split, field = "Salesforce/wikitext", "wikitext-2-raw-v1", "train", "text"
            def fn():
                try:
                    return try_datasets_pretrain(dataset, config, split, field, target_bytes, args.timeout_sec)
                except Exception as exc:
                    attempts.append({"label": "pretrain_wikitext_datasets", "attempt": "fallback_direct", "ok": False, "error": repr(exc)})
                    return direct_wikitext_pretrain(target_bytes, cache_dir, args.timeout_sec)
        else:
            dataset, config, split, field = "roneneldan/TinyStories", None, "train", "text"
            def fn():
                try:
                    return try_datasets_pretrain(dataset, config, split, field, target_bytes, args.timeout_sec)
                except Exception as exc:
                    attempts.append({"label": "pretrain_tinystories_datasets", "attempt": "fallback_direct", "ok": False, "error": repr(exc)})
                    return direct_tinystories_pretrain(target_bytes, cache_dir, args.timeout_sec)
        ok, result = run_with_retries(f"pretrain_{source}", fn, args.max_retries, args.timeout_sec, args.retry_sleep_sec, attempts)
        if ok and result:
            return save_pretrain(result["rows"], Path(args.out_dir), dataset, config, split, {k: v for k, v in result.items() if k != "rows"})
        failures.append({"source": source, "dataset": dataset, "config": config, "success": False})
    return {"success": False, "fallback": False, "failures": failures}


def direct_alpaca_sft(max_samples: int, cache_dir: Path, timeout_sec: float) -> Dict[str, Any]:
    repo = "tatsu-lab/alpaca"
    filenames = siblings(repo, timeout_sec)
    candidates = [f for f in filenames if f.endswith(".parquet")]
    if not candidates:
        raise RuntimeError(f"no Alpaca parquet found; siblings={filenames[:20]}")
    local = cache_dir / candidates[0].replace("/", "__")
    info = download_url(resolve_url(repo, candidates[0]), local, timeout_sec)
    raw_rows = read_parquet_rows(local)
    rows = []
    for item in raw_rows:
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        if instruction and output:
            rows.append(
                {
                    "instruction": instruction,
                    "input": str(item.get("input", "") or "").strip(),
                    "output": output,
                    "category": "alpaca",
                }
            )
        if len(rows) >= max_samples:
            break
    if not rows:
        raise RuntimeError("downloaded Alpaca parquet but no usable instruction/output rows")
    return {"rows": rows, "method": "urllib_parquet", "download": info, "source_file": candidates[0]}


def split_rows(rows: List[Dict[str, Any]], val_count: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    val_count = min(val_count, max(1, len(rows) // 10))
    return rows[:-val_count], rows[-val_count:]


def save_sft(rows: List[Dict[str, Any]], out_dir: Path, extra: Dict[str, Any], seed: int) -> Dict[str, Any]:
    train, val = split_rows(rows, val_count=max(1, min(2000, len(rows) // 10)), seed=seed)
    write_jsonl(train, out_dir / "sft" / "sft_train.jsonl")
    write_jsonl(val, out_dir / "sft" / "sft_val.jsonl")
    meta = {
        "success": True,
        "fallback": False,
        "source_dataset": "tatsu-lab/alpaca",
        "split": "train",
        "train": len(train),
        "val": len(val),
        "category_counts": dict(Counter(row.get("category", "unknown") for row in train)),
        "field_mapping": {"instruction": "instruction", "input": "input", "output": "output", "category": "alpaca"},
        **extra,
    }
    save_json(meta, str(out_dir / "sft" / "sft_metadata.json"))
    return meta


def download_sft(args, cache_dir: Path, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    def fn():
        return direct_alpaca_sft(int(args.sft_max_samples), cache_dir, float(args.timeout_sec))
    ok, result = run_with_retries("sft_alpaca", fn, args.max_retries, args.timeout_sec, args.retry_sleep_sec, attempts)
    if ok and result:
        return save_sft(result["rows"], Path(args.out_dir), {k: v for k, v in result.items() if k != "rows"}, args.seed + 1)
    return {"success": False, "fallback": False, "source_dataset": "tatsu-lab/alpaca"}


def load_json_from_url(repo: str, filename: str, cache_dir: Path, timeout_sec: float) -> Tuple[Any, Dict[str, Any]]:
    local = cache_dir / filename.replace("/", "__")
    info = download_url(resolve_url(repo, filename), local, timeout_sec)
    return json.loads(local.read_text(encoding="utf-8", errors="replace")), info


def map_preference_item(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    instruction = str(item.get("instruction") or item.get("prompt") or "").strip()
    input_text = str(item.get("input") or "").strip()
    category = "alpaca_farm"
    if item.get("chosen") and item.get("rejected"):
        chosen = str(item.get("chosen")).strip()
        rejected = str(item.get("rejected")).strip()
    elif item.get("output_1") and item.get("output_2"):
        output_1 = str(item.get("output_1")).strip()
        output_2 = str(item.get("output_2")).strip()
        pref = str(item.get("preference") or item.get("winner") or item.get("preferred") or "").lower()
        if pref in {"1", "output_1", "a", "model_1"}:
            chosen, rejected = output_1, output_2
        elif pref in {"2", "output_2", "b", "model_2"}:
            chosen, rejected = output_2, output_1
        else:
            return None
    elif item.get("preference") and isinstance(item.get("preference"), int) and item.get("output") and isinstance(item.get("output"), list):
        outputs = [str(x).strip() for x in item.get("output") if str(x).strip()]
        pref = int(item["preference"])
        if len(outputs) < 2 or pref < 1 or pref > len(outputs):
            return None
        chosen = outputs[pref - 1]
        rejected = outputs[1 - (pref - 1)] if len(outputs) == 2 else next((x for i, x in enumerate(outputs) if i != pref - 1), "")
    else:
        return None
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    return {
        "instruction": instruction,
        "input": input_text,
        "chosen": chosen,
        "rejected": rejected,
        "category": category,
        "rejected_type": "public_preference",
        "reason": "mapped from AlpacaFarm public preference fields",
    }


def direct_alpaca_farm_dpo(max_samples: int, cache_dir: Path, timeout_sec: float, out_dir: Path) -> Dict[str, Any]:
    repo = "tatsu-lab/alpaca_farm"
    preference_files = [
        "alpaca_gpt4_preference.json",
        "alpaca_human_preference.json",
        "alpaca_noisy_multi_preference.json",
        "alpaca_instructions/preference.json",
    ]
    try:
        filenames = siblings(repo, timeout_sec)
        files_to_try = [filename for filename in preference_files if filename in filenames]
    except Exception:
        files_to_try = preference_files
    field_probe: List[Dict[str, Any]] = []
    for filename in files_to_try:
        data, info = load_json_from_url(repo, filename, cache_dir, timeout_sec)
        raw_rows = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        if not raw_rows:
            field_probe.append({"file": filename, "error": "no list rows"})
            continue
        first = raw_rows[0]
        field_probe.append({"file": filename, "fields": sorted(first.keys()) if isinstance(first, dict) else str(type(first)), "first": str(first)[:1000]})
        rows = []
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            mapped = map_preference_item(item)
            if mapped:
                rows.append(mapped)
            if len(rows) >= max_samples:
                break
        if rows:
            return {"rows": rows, "method": "urllib_json", "download": info, "source_file": filename, "field_probe": field_probe}
    ensure_dir(str(Path(out_dir) / "dpo"))
    save_json({"field_probe": field_probe, "error": "no reliable chosen/rejected mapping found"}, str(Path(out_dir) / "dpo" / "alpaca_farm_field_probe.json"))
    raise RuntimeError("AlpacaFarm files were accessible but no reliable DPO mapping was found")


def save_dpo(rows: List[Dict[str, Any]], out_dir: Path, extra: Dict[str, Any], seed: int) -> Dict[str, Any]:
    train, val = split_rows(rows, val_count=max(1, min(1000, len(rows) // 10)), seed=seed)
    write_jsonl(train, out_dir / "dpo" / "dpo_train.jsonl")
    write_jsonl(val, out_dir / "dpo" / "dpo_val.jsonl")
    meta = {
        "success": True,
        "fallback": False,
        "source_dataset": "tatsu-lab/alpaca_farm",
        "train": len(train),
        "val": len(val),
        "category_counts": dict(Counter(row.get("category", "unknown") for row in train)),
        "rejected_type_counts": dict(Counter(row.get("rejected_type", "unknown") for row in train)),
        "field_mapping": "instruction/input plus chosen/rejected or output_1/output_2/preference",
        **extra,
    }
    save_json(meta, str(out_dir / "dpo" / "dpo_metadata.json"))
    return meta


def download_dpo(args, cache_dir: Path, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    def fn():
        return direct_alpaca_farm_dpo(int(args.dpo_max_samples), cache_dir, float(args.timeout_sec), Path(args.out_dir))
    ok, result = run_with_retries("dpo_alpaca_farm", fn, args.max_retries, args.timeout_sec, args.retry_sleep_sec, attempts)
    if ok and result:
        return save_dpo(result["rows"], Path(args.out_dir), {k: v for k, v in result.items() if k != "rows"}, args.seed + 2)
    failure = {
        "success": False,
        "fallback": False,
        "source_dataset": "tatsu-lab/alpaca_farm",
        "error": "all retry attempts failed or no reliable preference mapping found",
        "stale_fallback_files_may_exist": Path(args.out_dir, "dpo", "dpo_train.jsonl").exists(),
    }
    save_json(failure, str(Path(args.out_dir) / "dpo" / "dpo_metadata.json"))
    return failure


def write_reports(out_dir: Path, metadata: Dict[str, Any], attempts: List[Dict[str, Any]]) -> None:
    ensure_dir("audit_stage8")
    save_json(metadata, str(out_dir / "dataset_metadata.json"))
    save_json({"attempts": attempts, "metadata": metadata}, "audit_stage8/stage8_public_download_attempts.json")
    lines = [
        "# Stage 8 Public Dataset Report",
        "",
        "This report is generated by the retry downloader. It does not label synthetic fallback as public data.",
        "",
    ]
    for key in ["pretrain", "sft", "dpo"]:
        item = metadata.get(key, {})
        lines.extend(
            [
                f"## {key}",
                "",
                f"- success: `{item.get('success')}`",
                f"- fallback: `{item.get('fallback')}`",
                f"- source: `{item.get('source_dataset')}`",
                f"- train/val: `{item.get('train')}` / `{item.get('val')}`",
                f"- samples: `{item.get('samples')}`",
                f"- bytes: `{item.get('bytes')}`",
                f"- error/failures: `{item.get('error') or item.get('failures')}`",
                "",
            ]
        )
    if metadata.get("grpo"):
        lines.extend(["## grpo", "", f"`{metadata['grpo']}`", ""])
    lines.extend(["## Attempts", "", "```json", json.dumps(attempts, indent=2, ensure_ascii=False)[:12000], "```", ""])
    Path("audit_stage8/stage8_public_dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry downloading Stage 8 public dataset subsets without synthetic fallback.")
    parser.add_argument("--out-dir", default="data/stage8_public")
    parser.add_argument("--pretrain-target-mb", type=float, default=50.0)
    parser.add_argument("--sft-max-samples", type=int, default=22000)
    parser.add_argument("--dpo-max-samples", type=int, default=11000)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-sec", type=float, default=6.0)
    parser.add_argument("--prefer", choices=["wikitext", "tinystories"], default="wikitext")
    parser.add_argument("--only", choices=["all", "pretrain", "sft", "dpo"], default="all")
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "download_cache"
    for directory in [out_dir / "raw", out_dir / "sft", out_dir / "dpo", cache_dir, Path("audit_stage8")]:
        ensure_dir(str(directory))

    attempts: List[Dict[str, Any]] = []
    existing_metadata: Dict[str, Any] = {}
    existing_path = out_dir / "dataset_metadata.json"
    if existing_path.exists():
        try:
            existing_metadata = json.loads(existing_path.read_text(encoding="utf-8"))
        except Exception:
            existing_metadata = {}

    metadata: Dict[str, Any] = {
        "note": "Stage 8 redownload attempt. No Stage 7 synthetic fallback is written by this script.",
        "pretrain": existing_metadata.get("pretrain") if args.only not in {"all", "pretrain"} else download_pretrain(args, cache_dir, attempts),
        "sft": existing_metadata.get("sft") if args.only not in {"all", "sft"} else download_sft(args, cache_dir, attempts),
        "dpo": existing_metadata.get("dpo") if args.only not in {"all", "dpo"} else download_dpo(args, cache_dir, attempts),
        "grpo": existing_metadata.get("grpo") or {
            "success": True,
            "fallback": False,
            "source_dataset": "local_verifiable_reward",
            "note": "GRPO remains local verifiable reward data by design.",
        },
    }
    any_public = any(bool(metadata.get(key, {}).get("success")) for key in ["pretrain", "sft", "dpo"])
    metadata["any_public_success"] = any_public
    write_reports(out_dir, metadata, attempts)

    if not any_public:
        failed_lines = [
            "# Stage 8 Public Download Failed Report",
            "",
            "No public pretrain, SFT, or DPO dataset was downloaded successfully. No synthetic fallback was written by this retry downloader.",
            "",
            "## Attempts",
            "",
            "```json",
            json.dumps(attempts, indent=2, ensure_ascii=False)[:12000],
            "```",
            "",
        ]
        Path("audit_stage8/stage8_public_download_failed_report.md").write_text("\n".join(failed_lines), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
