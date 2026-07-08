# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.utils import ensure_dir, iter_jsonl, save_json


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl_limited(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for row in iter_jsonl(str(path)):
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def clean_text(text: str) -> Optional[str]:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 20 and "\ufffd" not in line]
    if not lines:
        return None
    text = "\n".join(lines)
    if len(text) < 40:
        return None
    return text


def dataset_info_record(name: str, config: Optional[str] = None) -> Dict[str, Any]:
    try:
        from datasets import load_dataset_builder  # type: ignore

        builder = load_dataset_builder(name, config, trust_remote_code=False) if config else load_dataset_builder(name, trust_remote_code=False)
        info = builder.info
        card_data = getattr(info, "cardData", None) or getattr(info, "card_data", None)
        license_value = card_data.get("license") if isinstance(card_data, dict) else getattr(info, "license", None)
        return {
            "name": name,
            "config": config,
            "license": license_value,
            "splits": list(getattr(info, "splits", {}) or []),
            "description_head": (info.description or "")[:300],
        }
    except Exception as exc:
        return {"name": name, "config": config, "error": repr(exc)}


def iter_public_text(name: str, config: Optional[str], split: str, field: str) -> Iterator[str]:
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset(name, config, split=split, streaming=True, trust_remote_code=False) if config else load_dataset(name, split=split, streaming=True, trust_remote_code=False)
    for item in dataset:
        text = clean_text(str(item.get(field, "")))
        if text:
            yield text


def build_public_pretrain(target_mb: float, out_path: Path) -> Dict[str, Any]:
    target_bytes = int(target_mb * 1024 * 1024)
    sources = [
        ("roneneldan/TinyStories", None, "train", "text"),
        ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", "text"),
    ]
    errors = []
    ensure_dir(str(out_path.parent))
    for name, config, split, field in sources:
        total = 0
        count = 0
        try:
            with out_path.open("w", encoding="utf-8") as f:
                for text in iter_public_text(name, config, split, field):
                    payload = text.strip() + "\n"
                    f.write(payload)
                    total += len(payload.encode("utf-8"))
                    count += 1
                    if total >= target_bytes:
                        break
            if total > 0:
                return {
                    "source": name,
                    "config": config,
                    "split": split,
                    "fallback": False,
                    "samples": count,
                    "bytes": total,
                    "lines": sum(1 for _ in out_path.open("r", encoding="utf-8")),
                    "target_mb": target_mb,
                    "info": dataset_info_record(name, config),
                    "errors": errors,
                }
        except Exception as exc:
            errors.append({"source": name, "config": config, "error": repr(exc)})
    stage7 = Path("data/stage7/raw/pretrain_corpus.txt")
    if not stage7.exists():
        raise RuntimeError("public pretrain failed and Stage 7 fallback corpus is missing")
    shutil.copyfile(stage7, out_path)
    return {
        "source": str(stage7),
        "fallback": True,
        "fallback_reason": errors,
        "samples": None,
        "bytes": out_path.stat().st_size,
        "lines": sum(1 for _ in out_path.open("r", encoding="utf-8")),
        "target_mb": target_mb,
    }


def normalize_alpaca(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "instruction": str(row.get("instruction", "")).strip(),
        "input": str(row.get("input", "") or "").strip(),
        "output": str(row.get("output", "")).strip(),
        "category": "alpaca",
    }


def build_public_sft(train_size: int, val_size: int, out_dir: Path, seed: int) -> Dict[str, Any]:
    errors = []
    rows: List[Dict[str, str]] = []
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=False)
        for item in ds:
            row = normalize_alpaca(item)
            if row["instruction"] and row["output"]:
                rows.append(row)
            if len(rows) >= train_size + val_size:
                break
    except Exception as exc:
        errors.append(repr(exc))
    fallback = False
    if not rows:
        fallback = True
        rows = read_jsonl_limited(Path("data/stage7/raw/sft_train.jsonl"), train_size) + read_jsonl_limited(Path("data/stage7/raw/sft_val.jsonl"), val_size)
    rng = random.Random(seed)
    rng.shuffle(rows)
    train = rows[: min(train_size, max(0, len(rows) - min(val_size, len(rows) // 10 or 1)))]
    val = rows[len(train) : len(train) + min(val_size, max(0, len(rows) - len(train)))]
    if not val and len(train) > 1:
        val = train[-max(1, len(train) // 20) :]
        train = train[: -len(val)]
    write_jsonl(train, out_dir / "sft_train.jsonl")
    write_jsonl(val, out_dir / "sft_val.jsonl")
    meta = {
        "source": "tatsu-lab/alpaca" if not fallback else "data/stage7/raw/sft_*.jsonl",
        "fallback": fallback,
        "errors": errors,
        "train": len(train),
        "val": len(val),
        "category_counts": dict(Counter(row.get("category", "unknown") for row in train)),
        "field_mapping": {"instruction": "instruction", "input": "input", "output": "output", "category": "alpaca"},
        "info": dataset_info_record("tatsu-lab/alpaca") if not fallback else None,
    }
    save_json(meta, str(out_dir / "sft_metadata.json"))
    return meta


def choose_pair_from_preference(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    instruction = str(row.get("instruction") or row.get("prompt") or "").strip()
    input_text = str(row.get("input") or "").strip()
    category = str(row.get("category") or "alpaca_farm")
    if "chosen" in row and "rejected" in row:
        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
    elif "output_1" in row and "output_2" in row:
        output_1 = str(row.get("output_1") or "").strip()
        output_2 = str(row.get("output_2") or "").strip()
        preference = row.get("preference") or row.get("winner") or row.get("preferred")
        if str(preference).strip() in {"1", "output_1", "A", "a"}:
            chosen, rejected = output_1, output_2
        elif str(preference).strip() in {"2", "output_2", "B", "b"}:
            chosen, rejected = output_2, output_1
        else:
            return None
    elif "completion_a" in row and "completion_b" in row:
        a = str(row.get("completion_a") or "").strip()
        b = str(row.get("completion_b") or "").strip()
        winner = str(row.get("winner") or row.get("preference") or "").lower()
        chosen, rejected = (a, b) if "a" in winner or "1" in winner else (b, a)
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
        "reason": "mapped from public preference fields",
    }


def iter_public_dpo_candidates() -> Iterator[Tuple[str, Optional[str], Dict[str, Any]]]:
    from datasets import get_dataset_config_names, load_dataset  # type: ignore

    dataset_name = "tatsu-lab/alpaca_farm"
    preferred_configs = [
        "alpaca_gpt4_preference",
        "alpaca_human_preference",
        "alpaca_farm_human_preference",
        "alpaca_farm_gpt4_preference",
    ]
    try:
        configs = get_dataset_config_names(dataset_name, trust_remote_code=False)
    except Exception:
        configs = []
    configs_to_try = [cfg for cfg in preferred_configs if cfg in configs] + [cfg for cfg in configs if "preference" in cfg.lower()]
    if not configs_to_try:
        configs_to_try = [None]
    for config in configs_to_try[:6]:
        try:
            ds = load_dataset(dataset_name, config, split="train", trust_remote_code=False) if config else load_dataset(dataset_name, split="train", trust_remote_code=False)
            for item in ds:
                yield dataset_name, config, dict(item)
        except Exception:
            continue


def build_public_dpo(train_size: int, val_size: int, out_dir: Path, seed: int) -> Dict[str, Any]:
    rows: List[Dict[str, str]] = []
    errors = []
    used_source = None
    used_config = None
    try:
        for source, config, raw in iter_public_dpo_candidates():
            used_source = source
            used_config = config
            mapped = choose_pair_from_preference(raw)
            if mapped:
                rows.append(mapped)
            if len(rows) >= train_size + val_size:
                break
    except Exception as exc:
        errors.append(repr(exc))
    fallback = False
    if not rows:
        fallback = True
        rows = read_jsonl_limited(Path("data/stage7/raw/dpo_train.jsonl"), train_size) + read_jsonl_limited(Path("data/stage7/raw/dpo_val.jsonl"), val_size)
        used_source = "data/stage7/raw/dpo_*.jsonl"
        used_config = None
    rng = random.Random(seed)
    rng.shuffle(rows)
    train = rows[: min(train_size, max(0, len(rows) - min(val_size, len(rows) // 10 or 1)))]
    val = rows[len(train) : len(train) + min(val_size, max(0, len(rows) - len(train)))]
    if not val and len(train) > 1:
        val = train[-max(1, len(train) // 20) :]
        train = train[: -len(val)]
    write_jsonl(train, out_dir / "dpo_train.jsonl")
    write_jsonl(val, out_dir / "dpo_val.jsonl")
    meta = {
        "source": used_source,
        "config": used_config,
        "fallback": fallback,
        "errors": errors,
        "train": len(train),
        "val": len(val),
        "category_counts": dict(Counter(row.get("category", "unknown") for row in train)),
        "rejected_type_counts": dict(Counter(row.get("rejected_type", "unknown") for row in train)),
        "field_mapping": "instruction/input/chosen/rejected; output_1/output_2 preference mapped when available",
        "info": dataset_info_record("tatsu-lab/alpaca_farm", used_config) if used_source == "tatsu-lab/alpaca_farm" else None,
    }
    save_json(meta, str(out_dir / "dpo_metadata.json"))
    return meta


def grpo_row(rng: random.Random, idx: int) -> Dict[str, str]:
    kind = idx % 7
    if kind == 0:
        word = rng.choice(["READY", "OK", "DONE", "SAFE", "PASS", "ACK"])
        return {"prompt": f"User: Output exactly the word {word} and nothing else.\nAssistant: ", "answer": word, "category": "exact_text", "reward_type": "exact_text", "keyword": word, "difficulty": "easy"}
    if kind == 1:
        keyword = rng.choice(["LoRA", "tokenizer", "SFT", "DPO", "reward", "RoPE"])
        return {"prompt": f"User: Write one short phrase containing the keyword {keyword}.\nAssistant: ", "answer": keyword, "category": "keyword", "reward_type": "keyword", "keyword": keyword, "difficulty": "easy"}
    if kind == 2:
        a, b = rng.randint(0, 80), rng.randint(0, 80)
        return {"prompt": f"User: Compute {a} + {b}. Answer with the final integer only.\nAssistant: ", "answer": str(a + b), "category": "math_add", "reward_type": "exact_integer", "keyword": "", "difficulty": "medium"}
    if kind == 3:
        a, b = rng.randint(20, 120), rng.randint(0, 80)
        return {"prompt": f"User: Compute {a} - {b}. Answer with the final integer only.\nAssistant: ", "answer": str(a - b), "category": "math_sub", "reward_type": "exact_integer", "keyword": "", "difficulty": "medium"}
    if kind == 4:
        a, b = rng.randint(2, 15), rng.randint(2, 15)
        return {"prompt": f"User: Compute {a} * {b}. Answer with the final integer only.\nAssistant: ", "answer": str(a * b), "category": "math_mul_small", "reward_type": "exact_integer", "keyword": "", "difficulty": "medium"}
    if kind == 5:
        a, b, c = rng.randint(1, 30), rng.randint(1, 30), rng.randint(1, 20)
        return {"prompt": f"User: Compute ({a} + {b}) - {c}. Answer with the final integer only.\nAssistant: ", "answer": str((a + b) - c), "category": "multi_step_arithmetic", "reward_type": "exact_integer", "keyword": "", "difficulty": "hard"}
    a, b = rng.randint(1, 30), rng.randint(1, 30)
    ans = a + b
    return {"prompt": f"User: Reply in the exact schema ANSWER=<integer>. Compute {a} + {b}.\nAssistant: ", "answer": str(ans), "category": "schema_math", "reward_type": "exact_integer", "keyword": "ANSWER=", "difficulty": "hard"}


def build_grpo(train_size: int, val_size: int, out_dir: Path, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    rows = [grpo_row(rng, idx) for idx in range(train_size + val_size)]
    rng.shuffle(rows)
    train, val = rows[:train_size], rows[train_size:]
    write_jsonl(train, out_dir / "grpo_train.jsonl")
    write_jsonl(val, out_dir / "grpo_val.jsonl")
    meta = {
        "source": "local_verifiable_reward",
        "fallback": False,
        "train": len(train),
        "val": len(val),
        "category_counts": dict(Counter(row.get("category", "unknown") for row in train)),
        "reward_type_counts": dict(Counter(row.get("reward_type", "unknown") for row in train)),
        "difficulty_counts": dict(Counter(row.get("difficulty", "unknown") for row in train)),
    }
    save_json(meta, str(out_dir / "grpo_metadata.json"))
    return meta


def line_count(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Stage 8 public dataset subsets.")
    parser.add_argument("--out-dir", default="data/stage8_public")
    parser.add_argument("--pretrain-target-mb", type=float, default=50.0)
    parser.add_argument("--sft-train-size", type=int, default=20000)
    parser.add_argument("--sft-val-size", type=int, default=2000)
    parser.add_argument("--dpo-train-size", type=int, default=10000)
    parser.add_argument("--dpo-val-size", type=int, default=1000)
    parser.add_argument("--grpo-train-size", type=int, default=3000)
    parser.add_argument("--grpo-val-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    base = Path(args.out_dir)
    raw = base / "raw"
    sft_dir = base / "sft"
    dpo_dir = base / "dpo"
    grpo_dir = base / "grpo"
    for directory in [raw, sft_dir, dpo_dir, grpo_dir, Path("audit_stage8")]:
        ensure_dir(str(directory))

    pretrain_path = raw / "pretrain_public.txt"
    pretrain_meta = build_public_pretrain(args.pretrain_target_mb, pretrain_path)
    save_json(pretrain_meta, str(raw / "pretrain_public_metadata.json"))

    sft_meta = build_public_sft(args.sft_train_size, args.sft_val_size, sft_dir, args.seed + 1)
    dpo_meta = build_public_dpo(args.dpo_train_size, args.dpo_val_size, dpo_dir, args.seed + 2)
    grpo_meta = build_grpo(args.grpo_train_size, args.grpo_val_size, grpo_dir, args.seed + 3)

    metadata = {
        "pretrain": pretrain_meta,
        "sft": sft_meta,
        "dpo": dpo_meta,
        "grpo": grpo_meta,
        "note": "Stage 8 uses public dataset subsets when accessible and explicit fallback otherwise. It is a migration/control experiment, not a capability benchmark.",
    }
    save_json(metadata, str(base / "dataset_metadata.json"))

    report = [
        "# Stage 8 Public Dataset Report",
        "",
        f"- Pretrain source: `{pretrain_meta.get('source')}`",
        f"- Pretrain fallback: `{pretrain_meta.get('fallback')}`",
        f"- Pretrain path: `{pretrain_path}`",
        f"- Pretrain bytes: `{pretrain_path.stat().st_size}`",
        f"- Pretrain lines: `{line_count(pretrain_path)}`",
        f"- SFT source: `{sft_meta.get('source')}`, fallback: `{sft_meta.get('fallback')}`, train/val: `{sft_meta.get('train')}` / `{sft_meta.get('val')}`",
        f"- DPO source: `{dpo_meta.get('source')}`, fallback: `{dpo_meta.get('fallback')}`, train/val: `{dpo_meta.get('train')}` / `{dpo_meta.get('val')}`",
        f"- GRPO source: `{grpo_meta.get('source')}`, train/val: `{grpo_meta.get('train')}` / `{grpo_meta.get('val')}`",
        "",
        "## SFT Distribution",
        "",
        f"`{sft_meta.get('category_counts')}`",
        "",
        "## DPO Distribution",
        "",
        f"- categories: `{dpo_meta.get('category_counts')}`",
        f"- rejected types: `{dpo_meta.get('rejected_type_counts')}`",
        "",
        "## GRPO Distribution",
        "",
        f"- categories: `{grpo_meta.get('category_counts')}`",
        f"- reward types: `{grpo_meta.get('reward_type_counts')}`",
        f"- difficulty: `{grpo_meta.get('difficulty_counts')}`",
        "",
        "This report records fallback explicitly so README claims do not imply unavailable public data was used.",
        "",
    ]
    Path("audit_stage8/stage8_public_dataset_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
