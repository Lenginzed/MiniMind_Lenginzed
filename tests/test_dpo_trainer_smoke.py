from __future__ import annotations

from pathlib import Path

import torch
import yaml

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.dpo_data import write_jsonl
from minillm.dpo_trainer import freeze_model, run_dpo
from minillm.tokenizer import MiniTokenizer


def setup_tiny_dpo_run(tmp_path: Path, lora: bool) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "User: Explain LoRA.\nAssistant: LoRA trains adapters.\n"
            "User: Compute 2 + 3.\nAssistant: 5.\n"
            "Chosen answers are better than rejected answers.\n"
        )
        * 120,
        encoding="utf-8",
    )
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    tokenizer_path = tmp_path / "tok.json"
    tokenizer.save(str(tokenizer_path))
    rows = [
        {
            "instruction": "Explain LoRA.",
            "input": "",
            "chosen": "LoRA trains small adapters.",
            "rejected": "It depends.",
            "category": "concept",
            "rejected_type": "vague",
            "reason": "too vague",
        },
        {
            "instruction": "Compute 2 + 3.",
            "input": "",
            "chosen": "5. Check: 2 + 3 = 5.",
            "rejected": "8. Check: 2 + 3 = 8.",
            "category": "math",
            "rejected_type": "wrong_answer",
            "reason": "wrong answer",
        },
    ] * 16
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(rows, str(train_path))
    write_jsonl(rows[:8], str(val_path))
    cfg = MiniLLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=48,
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        intermediate_size=64,
    )
    model = MiniLLMForCausalLM(cfg)
    base_ckpt = tmp_path / "sft_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": cfg.__dict__,
            "optimizer_state_dict": {},
            "step": 0,
            "best_eval_loss": 0.0,
        },
        base_ckpt,
    )
    config = {
        "seed": 123,
        "policy_checkpoint": str(base_ckpt),
        "reference_checkpoint": str(base_ckpt),
        "tokenizer_path": str(tokenizer_path),
        "train_data_path": str(train_path),
        "val_data_path": str(val_path),
        "output_dir": str(tmp_path / ("dpo_lora" if lora else "dpo_full")),
        "prefer_cuda": False,
        "dtype": "fp32",
        "max_length": 48,
        "beta": 0.1,
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
            "eval_interval": 1,
            "save_interval": 1,
            "log_interval": 1,
            "eval_batches": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "scheduler": "none",
            "warmup_steps": 0,
        },
        "sample_prompts": ["Explain LoRA."],
    }
    if lora:
        config["lora"] = {"enabled": True, "r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
    config_path = tmp_path / ("dpo_lora.yaml" if lora else "dpo_full.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_freeze_model_marks_reference_params_non_trainable() -> None:
    model = MiniLLMForCausalLM(MiniLLMConfig())
    freeze_model(model)
    assert all(not p.requires_grad for p in model.parameters())


def test_full_dpo_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_dpo(str(setup_tiny_dpo_run(tmp_path, lora=False)))
    assert summary["mode"] == "full"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] == summary["parameter_count"]
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["metrics_path"]).exists()


def test_lora_dpo_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_dpo(str(setup_tiny_dpo_run(tmp_path, lora=True)))
    assert summary["mode"] == "lora"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] < summary["parameter_count"]
    assert summary["trainable_ratio"] < 0.1
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["output_dir"], "adapters", "best_adapter.pt").exists()
