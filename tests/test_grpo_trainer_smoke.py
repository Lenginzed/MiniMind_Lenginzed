from __future__ import annotations

from pathlib import Path

import torch
import yaml

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.grpo_data import write_jsonl
from minillm.grpo_trainer import run_grpo
from minillm.tokenizer import MiniTokenizer


def setup_tiny_grpo_run(tmp_path: Path, lora: bool) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "User: 1 + 1?\nAssistant: 2\n"
            "User: say LoRA\nAssistant: LoRA\n"
            "User: output OK\nAssistant: OK\n"
        )
        * 120,
        encoding="utf-8",
    )
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    tokenizer_path = tmp_path / "tok.json"
    tokenizer.save(str(tokenizer_path))
    rows = [
        {
            "prompt": "User: Compute 1 + 1. Answer with the final integer only.\nAssistant: ",
            "answer": "2",
            "category": "math_add",
            "reward_type": "exact_integer",
            "keyword": "",
        },
        {
            "prompt": "User: Mention LoRA.\nAssistant: ",
            "answer": "LoRA",
            "category": "concept_keyword",
            "reward_type": "keyword",
            "keyword": "LoRA",
        },
        {
            "prompt": "User: Output exactly the word OK and nothing else.\nAssistant: ",
            "answer": "OK",
            "category": "format_echo",
            "reward_type": "exact_text",
            "keyword": "OK",
        },
    ] * 8
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(rows, str(train_path))
    write_jsonl(rows[:6], str(val_path))

    cfg = MiniLLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=40,
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        intermediate_size=64,
    )
    model = MiniLLMForCausalLM(cfg)
    ckpt = tmp_path / "sft_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": cfg.__dict__,
            "optimizer_state_dict": {},
            "step": 0,
            "best_eval_loss": 0.0,
        },
        ckpt,
    )
    config = {
        "seed": 123,
        "policy_checkpoint": str(ckpt),
        "tokenizer_path": str(tokenizer_path),
        "train_data_path": str(train_path),
        "val_data_path": str(val_path),
        "output_dir": str(tmp_path / ("grpo_lora" if lora else "grpo_full")),
        "prefer_cuda": False,
        "dtype": "fp32",
        "max_prompt_length": 28,
        "max_new_tokens": 4,
        "num_generations": 2,
        "clip_epsilon": 0.2,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
            "eval_interval": 1,
            "save_interval": 1,
            "log_interval": 1,
            "eval_examples": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "scheduler": "none",
            "warmup_steps": 0,
        },
    }
    if lora:
        config["lora"] = {"enabled": True, "r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
    config_path = tmp_path / ("grpo_lora.yaml" if lora else "grpo_full.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_full_grpo_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_grpo(str(setup_tiny_grpo_run(tmp_path, lora=False)))
    assert summary["mode"] == "full"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] == summary["parameter_count"]
    assert Path(summary["metrics_path"]).exists()
    assert Path(summary["rollout_samples_path"]).exists()
    assert Path(summary["best_checkpoint"]).exists()


def test_lora_grpo_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_grpo(str(setup_tiny_grpo_run(tmp_path, lora=True)))
    assert summary["mode"] == "lora"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] < summary["parameter_count"]
    assert Path(summary["metrics_path"]).exists()
    assert Path(summary["output_dir"], "adapters", "best_adapter.pt").exists()
