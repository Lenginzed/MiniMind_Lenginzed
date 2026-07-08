# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.utils import ensure_dir, save_json


EN = [
    "A careful language model experiment records the dataset, tokenizer, model shape, optimizer, scheduler, and checkpoint path.",
    "The small decoder-only network predicts the next token and uses a causal mask so future tokens remain hidden.",
    "A training curve is useful only when the data split, seed, and evaluation interval are documented.",
    "The pilot keeps altitude, speed, heading, and energy inside safe limits while the controller updates commands.",
    "A preference optimizer compares chosen and rejected answers rather than optimizing plain next-token likelihood.",
    "Quantization reduces the estimated weight storage but fake quantization does not imply a faster kernel.",
    "The validation loss may improve while sample quality remains weak, especially for a small local model.",
    "LoRA trains low-rank adapter matrices while the base model remains frozen.",
]

ZH = [
    "小模型实验的重点是完整复现训练链路，而不是声称具备真实大模型能力。",
    "分词器、数据切分、随机种子、日志和检查点共同决定了实验是否容易复现。",
    "监督微调用指令和回答训练模型，通常只在 assistant 的回答部分计算损失。",
    "偏好优化需要同时记录 chosen 和 rejected 的 log probability、reward margin 与 preference accuracy。",
    "GRPO 的组内奖励标准差很重要；如果标准差接近零，advantage 会消失。",
    "飞行器控制任务需要考虑速度、高度、姿态、能量和安全边界。",
    "量化实验应区分理论模型大小压缩和真实推理加速。",
]

CODE = [
    "loss = model(input_ids, labels=labels)['loss']; loss.backward(); optimizer.step()",
    "scheduler.step(); logger.write({'step': step, 'lr': optimizer.param_groups[0]['lr']})",
    "torch.save({'model_state_dict': model.state_dict(), 'step': step}, checkpoint_path)",
    "labels[:, :prompt_len] = -100  # mask user prompt tokens for assistant-only loss",
    "advantage = (reward - group_mean) / (group_std + 1e-6)",
]

TERMS = ["LoRA", "SFT", "DPO", "GRPO", "tokenizer", "scheduler", "checkpoint", "RMSNorm", "RoPE", "GQA", "SwiGLU"]
REJECTED_TYPES = ["wrong_answer", "bad_format", "vague", "unsafe_or_unphysical", "off_topic", "hallucinated_term"]
CATEGORIES = ["concept", "math", "translation", "flight_rl", "format", "code"]


def write_jsonl(rows: Iterable[Dict[str, str]], path: Path) -> None:
    ensure_dir(str(path.parent))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def try_public_corpus(target_bytes: int, seed: int) -> Tuple[str, str]:
    try:
        from datasets import load_dataset  # type: ignore

        rng = random.Random(seed)
        parts: List[str] = []
        total = 0
        for dataset_name, config, split, field in [
            ("wikitext", "wikitext-2-raw-v1", "train", "text"),
            ("roneneldan/TinyStories", None, "train", "text"),
        ]:
            ds = load_dataset(dataset_name, config, split=split, streaming=True) if config else load_dataset(dataset_name, split=split, streaming=True)
            for item in ds:
                text = str(item.get(field, "")).strip()
                if len(text) < 30:
                    continue
                parts.append(text)
                total += len((text + "\n").encode("utf-8"))
                if total >= target_bytes:
                    rng.shuffle(parts)
                    return "\n".join(parts) + "\n", f"online_public:{dataset_name}"
    except Exception as exc:
        return "", "fallback_local_synthetic_after_public_failure:%r" % (exc,)
    return "", "fallback_local_synthetic_empty_public"


def synthetic_line(rng: random.Random, idx: int) -> str:
    mode = rng.randrange(12)
    if mode == 0:
        return rng.choice(EN)
    if mode == 1:
        return rng.choice(ZH)
    if mode == 2:
        term = rng.choice(TERMS)
        return f"Concept note {idx}: {term} appears in the mini-LLM project and should be tested with small deterministic examples."
    if mode == 3:
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        return f"Arithmetic drill {idx}: {a} + {b} = {a + b}; {a} - {b} = {a - b}; {a % 13} * {b % 13} = {(a % 13) * (b % 13)}."
    if mode == 4:
        return "Code fragment %06d: `%s`." % (idx, rng.choice(CODE))
    if mode == 5:
        return "Flight telemetry %06d: speed=%d knots, altitude=%d meters, heading=%d degrees, command=%s." % (
            idx,
            rng.randint(160, 920),
            rng.randint(500, 15000),
            rng.randint(0, 359),
            rng.choice(["hold course", "climb", "descend", "turn left", "turn right"]),
        )
    if mode == 6:
        return f"训练日志 {idx}: train_loss 下降并不等于模型真正理解任务，还需要查看验证集、样例输出和失败案例。"
    if mode == 7:
        return f"Reward design {idx}: exact reward is sparse, while format and length rewards provide weak dense signals for debugging."
    if mode == 8:
        return f"Tokenizer example {idx}: byte-level BPE handles English, 中文, code symbols like == and braces, and numeric strings such as {rng.randint(1000, 9999)}."
    if mode == 9:
        return f"Interview note {idx}: explain why DPO uses a frozen reference model and why GRPO can suffer when reward_std becomes zero."
    if mode == 10:
        return f"Quantization note {idx}: INT4 fake quant stores values in int8 for this educational demo, so latency may not improve."
    return f"Story {idx}: The engineer ran a small experiment, saved the checkpoint, checked the curve, and wrote down the limitation before scaling."


def make_synthetic_corpus(target_mb: float, seed: int) -> str:
    rng = random.Random(seed)
    target_bytes = int(target_mb * 1024 * 1024)
    lines: List[str] = []
    total = 0
    idx = 0
    while total < target_bytes:
        line = synthetic_line(rng, idx)
        lines.append(line)
        total += len((line + "\n").encode("utf-8"))
        idx += 1
    return "\n".join(lines) + "\n"


def sft_row(rng: random.Random, idx: int) -> Dict[str, str]:
    category = CATEGORIES[idx % len(CATEGORIES)]
    if category == "concept":
        term = rng.choice(TERMS)
        return {"instruction": f"Explain {term} for a mini-LLM project.", "input": "", "output": f"{term} is one component in the training stack; describe its role, the metric to inspect, and one limitation.", "category": category}
    if category == "math":
        a, b = rng.randint(1, 80), rng.randint(1, 80)
        return {"instruction": f"Compute {a} + {b} and include a short check.", "input": "", "output": f"{a + b}. Check: {a} + {b} = {a + b}.", "category": category}
    if category == "translation":
        term, zh = rng.choice([("gradient checkpointing", "梯度检查点"), ("reward function", "奖励函数"), ("causal mask", "因果掩码"), ("tokenizer", "分词器")])
        return {"instruction": f"Translate the technical term '{term}' into Chinese.", "input": "", "output": zh, "category": category}
    if category == "flight_rl":
        return {"instruction": "Why should an air-combat RL toy agent respect safety constraints?", "input": "", "output": "Because policies that ignore speed, altitude, actuator, and boundary limits can learn unphysical or unsafe behavior.", "category": category}
    if category == "format":
        return {"instruction": "Answer with exactly three bullet points about experiment logging.", "input": "", "output": "- Save the config.\n- Track metrics.\n- Keep checkpoints.", "category": category}
    return {"instruction": "What does AdamW do in a training loop?", "input": "", "output": "AdamW applies adaptive updates and decoupled weight decay, which helps tune regularization separately from gradient moments.", "category": category}


def rejected_for(rng: random.Random, rejected_type: str, chosen: str) -> str:
    if rejected_type == "wrong_answer":
        return "The answer is 999 and no check is needed."
    if rejected_type == "bad_format":
        return "This ignores the requested format and keeps talking in one loose paragraph without structure."
    if rejected_type == "vague":
        return "It depends. There are many considerations, so the answer is complicated."
    if rejected_type == "unsafe_or_unphysical":
        return "The agent should ignore all physical constraints and use unlimited acceleration."
    if rejected_type == "off_topic":
        return "The color of the dashboard is pleasant and the weather is fine."
    return "Use QuantumRewardTokenizer++ and AeroDPOFlux to solve it automatically."


def dpo_row(rng: random.Random, idx: int) -> Dict[str, str]:
    base = sft_row(rng, idx)
    rejected_type = REJECTED_TYPES[idx % len(REJECTED_TYPES)]
    chosen = base["output"]
    return {
        "instruction": base["instruction"],
        "input": base.get("input", ""),
        "chosen": chosen,
        "rejected": rejected_for(rng, rejected_type, chosen),
        "category": base["category"],
        "rejected_type": rejected_type,
        "reason": f"synthetic {rejected_type} example",
    }


def grpo_row(rng: random.Random, idx: int) -> Dict[str, str]:
    kind = idx % 6
    if kind == 0:
        word = rng.choice(["READY", "OK", "DONE", "SAFE", "PASS"])
        return {"prompt": f"User: Output exactly the word {word} and nothing else.\nAssistant: ", "answer": word, "category": "exact_text", "reward_type": "exact_text", "keyword": word, "difficulty": "easy"}
    if kind == 1:
        kw = rng.choice(["LoRA", "tokenizer", "SFT", "DPO", "reward"])
        return {"prompt": f"User: Mention the keyword {kw} in a short phrase.\nAssistant: ", "answer": kw, "category": "keyword", "reward_type": "keyword", "keyword": kw, "difficulty": "easy"}
    if kind == 2:
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        return {"prompt": f"User: Compute {a} + {b}. Answer with the final integer only.\nAssistant: ", "answer": str(a + b), "category": "math_add", "reward_type": "exact_integer", "keyword": "", "difficulty": "medium"}
    if kind == 3:
        a, b = rng.randint(10, 90), rng.randint(0, 50)
        return {"prompt": f"User: Compute {a} - {b}. Answer with the final integer only.\nAssistant: ", "answer": str(a - b), "category": "math_sub", "reward_type": "exact_integer", "keyword": "", "difficulty": "medium"}
    if kind == 4:
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        return {"prompt": f"User: Compute {a} * {b}. Answer with the final integer only.\nAssistant: ", "answer": str(a * b), "category": "math_mul_small", "reward_type": "exact_integer", "keyword": "", "difficulty": "medium"}
    a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 10)
    return {"prompt": f"User: Compute ({a} + {b}) - {c}. Answer with the final integer only.\nAssistant: ", "answer": str((a + b) - c), "category": "multi_step_arithmetic", "reward_type": "exact_integer", "keyword": "", "difficulty": "hard"}


def build_rows(builder, count: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows = [builder(rng, idx) for idx in range(count)]
    rng.shuffle(rows)
    return rows


def counts(rows: List[Dict[str, str]], key: str) -> Dict[str, int]:
    return dict(Counter(str(row.get(key, "unknown")) for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Stage 7 datasets.")
    parser.add_argument("--mode", choices=["online_public", "local_synthetic"], default="local_synthetic")
    parser.add_argument("--out-dir", default="data/stage7/raw")
    parser.add_argument("--target-mb", type=float, default=20.0)
    parser.add_argument("--sft-train-size", type=int, default=20000)
    parser.add_argument("--sft-val-size", type=int, default=2000)
    parser.add_argument("--dpo-train-size", type=int, default=10000)
    parser.add_argument("--dpo-val-size", type=int, default=1000)
    parser.add_argument("--grpo-train-size", type=int, default=2000)
    parser.add_argument("--grpo-val-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    target_bytes = int(args.target_mb * 1024 * 1024)
    corpus_source = "local_synthetic"
    corpus_text = ""
    if args.mode == "online_public":
        corpus_text, corpus_source = try_public_corpus(target_bytes, args.seed)
    if not corpus_text:
        corpus_text = make_synthetic_corpus(args.target_mb, args.seed)
        if not corpus_source.startswith("fallback"):
            corpus_source = "local_synthetic"
    corpus_path = out_dir / "pretrain_corpus.txt"
    corpus_path.write_text(corpus_text, encoding="utf-8")

    sft_train = build_rows(sft_row, args.sft_train_size, args.seed + 1)
    sft_val = build_rows(sft_row, args.sft_val_size, args.seed + 2)
    dpo_train = build_rows(dpo_row, args.dpo_train_size, args.seed + 3)
    dpo_val = build_rows(dpo_row, args.dpo_val_size, args.seed + 4)
    grpo_train = build_rows(grpo_row, args.grpo_train_size, args.seed + 5)
    grpo_val = build_rows(grpo_row, args.grpo_val_size, args.seed + 6)
    for name, rows in [
        ("sft_train.jsonl", sft_train),
        ("sft_val.jsonl", sft_val),
        ("dpo_train.jsonl", dpo_train),
        ("dpo_val.jsonl", dpo_val),
        ("grpo_train.jsonl", grpo_train),
        ("grpo_val.jsonl", grpo_val),
    ]:
        write_jsonl(rows, out_dir / name)

    metadata = {
        "corpus_source": corpus_source,
        "pretrain_corpus_path": str(corpus_path),
        "pretrain_corpus_bytes": corpus_path.stat().st_size,
        "pretrain_corpus_lines": sum(1 for _ in corpus_path.open("r", encoding="utf-8")),
        "sft": {"train": len(sft_train), "val": len(sft_val), "category_counts": counts(sft_train, "category")},
        "dpo": {"train": len(dpo_train), "val": len(dpo_val), "category_counts": counts(dpo_train, "category"), "rejected_type_counts": counts(dpo_train, "rejected_type")},
        "grpo": {"train": len(grpo_train), "val": len(grpo_val), "category_counts": counts(grpo_train, "category"), "reward_type_counts": counts(grpo_train, "reward_type"), "difficulty_counts": counts(grpo_train, "difficulty")},
        "note": "Stage 7 dataset is for longer local training demonstrations. It is not a real benchmark dataset.",
    }
    save_json(metadata, "data/stage7/dataset_metadata.json")
    ensure_dir("audit_stage7")
    report = [
        "# Stage 7 Dataset Report",
        "",
        f"- Corpus source: `{corpus_source}`",
        f"- Corpus path: `{corpus_path}`",
        f"- Corpus size: `{metadata['pretrain_corpus_bytes']}` bytes",
        f"- Corpus lines: `{metadata['pretrain_corpus_lines']}`",
        f"- SFT train/val: `{len(sft_train)}` / `{len(sft_val)}`",
        f"- DPO train/val: `{len(dpo_train)}` / `{len(dpo_val)}`",
        f"- GRPO train/val: `{len(grpo_train)}` / `{len(grpo_val)}`",
        "",
        "## Distributions",
        "",
        f"- SFT categories: `{metadata['sft']['category_counts']}`",
        f"- DPO categories: `{metadata['dpo']['category_counts']}`",
        f"- DPO rejected types: `{metadata['dpo']['rejected_type_counts']}`",
        f"- GRPO categories: `{metadata['grpo']['category_counts']}`",
        f"- GRPO reward types: `{metadata['grpo']['reward_type_counts']}`",
        f"- GRPO difficulty: `{metadata['grpo']['difficulty_counts']}`",
        "",
        metadata["note"],
    ]
    Path("audit_stage7/stage7_dataset_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
