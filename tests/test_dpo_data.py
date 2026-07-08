from __future__ import annotations

from pathlib import Path

from minillm.dpo_data import DPODataset, dpo_collate_fn, write_jsonl
from minillm.sft_data import IGNORE_INDEX
from minillm.tokenizer import MiniTokenizer


def build_tokenizer(tmp_path: Path) -> MiniTokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "User: Explain LoRA.\nAssistant: LoRA trains low-rank adapters.\n"
            "Chosen responses are concise and correct. Rejected answers are wrong or vague.\n"
        )
        * 80,
        encoding="utf-8",
    )
    return MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)


def test_dpo_dataset_masks_prompt_and_keeps_eos(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    eos_id = tokenizer.special_token_ids["eos_token_id"]
    rows = [
        {
            "instruction": "Explain LoRA.",
            "input": "",
            "chosen": "LoRA trains low-rank adapters.",
            "rejected": "It depends.",
            "category": "concept",
            "rejected_type": "vague",
            "reason": "too vague",
        }
    ]
    path = tmp_path / "dpo.jsonl"
    write_jsonl(rows, str(path))
    ds = DPODataset(str(path), tokenizer, max_length=64)
    item = ds[0]

    for labels_name in ["chosen_labels", "rejected_labels"]:
        labels = item[labels_name].tolist()
        first_valid = next(i for i, value in enumerate(labels) if value != IGNORE_INDEX)
        assert first_valid > 0
        assert all(value == IGNORE_INDEX for value in labels[:first_valid])
        assert eos_id in labels[first_valid:]
        assert sum(value != IGNORE_INDEX for value in labels) > 0

    stats = ds.stats()
    assert stats["raw_examples"] == 1
    assert stats["effective_examples"] == 1
    assert stats["skipped_examples"] == 0
    assert stats["category_counts"]["concept"] == 1
    assert stats["rejected_type_counts"]["vague"] == 1


def test_dpo_collate_padding_labels_are_ignored(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    pad_id = tokenizer.special_token_ids["pad_token_id"]
    rows = [
        {
            "instruction": "Explain LoRA.",
            "input": "",
            "chosen": "LoRA trains adapters.",
            "rejected": "Vague.",
            "category": "concept",
            "rejected_type": "vague",
            "reason": "too vague",
        },
        {
            "instruction": "Explain DPO.",
            "input": "",
            "chosen": "DPO compares chosen and rejected completions before updating the policy.",
            "rejected": "Blue dashboard.",
            "category": "concept",
            "rejected_type": "off_topic",
            "reason": "off topic",
        },
    ]
    path = tmp_path / "dpo.jsonl"
    write_jsonl(rows, str(path))
    ds = DPODataset(str(path), tokenizer, max_length=80)
    batch = dpo_collate_fn([ds[0], ds[1]], int(pad_id))
    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["rejected_input_ids"].shape[0] == 2
    assert batch["chosen_input_ids"].shape == batch["chosen_labels"].shape
    assert batch["rejected_input_ids"].shape == batch["rejected_labels"].shape
    chosen_pad = batch["chosen_attention_mask"] == 0
    rejected_pad = batch["rejected_attention_mask"] == 0
    assert (batch["chosen_labels"][chosen_pad] == IGNORE_INDEX).all()
    assert (batch["rejected_labels"][rejected_pad] == IGNORE_INDEX).all()


def test_dpo_truncation_can_skip_when_response_is_fully_cut(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    rows = [
        {
            "instruction": "Explain LoRA with a very long prompt " * 40,
            "input": "",
            "chosen": "LoRA trains adapters.",
            "rejected": "Wrong.",
            "category": "concept",
            "rejected_type": "wrong_answer",
            "reason": "wrong",
        }
    ]
    path = tmp_path / "dpo.jsonl"
    write_jsonl(rows, str(path))
    try:
        DPODataset(str(path), tokenizer, max_length=8)
    except ValueError as exc:
        assert "all DPO examples were skipped" in str(exc)
    else:
        raise AssertionError("expected all examples to be skipped when response labels are truncated away")
