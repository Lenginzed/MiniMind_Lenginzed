from __future__ import annotations

from pathlib import Path

import torch
import yaml

from minillm.data import save_tokenized_splits
from minillm.tokenizer import MiniTokenizer
from minillm.trainer import run_pretrain


def test_resume_training_step_increases_and_checkpoint_loads(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("mini resume smoke test 小模型\n" * 200), encoding="utf-8")
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    processed = tmp_path / "processed"
    save_tokenized_splits(
        tokenizer,
        str(corpus),
        str(processed),
        block_size=16,
        val_ratio=0.1,
        split_mode="random_blocks",
        seed=123,
    )

    config = {
        "seed": 123,
        "output_dir": str(tmp_path / "out"),
        "tokenizer_path": str(tokenizer_path),
        "train_data_path": str(processed / "train.npy"),
        "val_data_path": str(processed / "val.npy"),
        "prefer_cuda": False,
        "dtype": "fp32",
        "data": {"block_size": 16},
        "model": {
            "vocab_size": 128,
            "context_length": 16,
            "n_layer": 1,
            "n_embd": 32,
            "n_head": 4,
            "n_kv_head": 2,
            "intermediate_size": 64,
            "dropout": 0.0,
            "use_gradient_checkpointing": False,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
            "eval_interval": 1,
            "save_interval": 1,
            "log_interval": 1,
            "eval_batches": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "scheduler": "linear",
            "warmup_steps": 1,
        },
        "sample_prompts": ["mini"],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    first = run_pretrain(str(config_path))
    assert first["final_step"] == 2
    checkpoint = Path(first["last_checkpoint"])
    assert checkpoint.exists()

    second = run_pretrain(str(config_path), resume_path=str(checkpoint), max_steps_override=4)
    assert second["resume_loaded"] is True
    assert second["start_step"] == 2
    assert second["final_step"] == 4
    loaded = torch.load(second["last_checkpoint"], map_location="cpu")
    assert loaded["step"] == 4
    assert "optimizer_state_dict" in loaded
