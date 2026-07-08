from __future__ import annotations

from minillm.grpo_rewards import (
    combined_reward,
    contains_keyword,
    extract_first_integer,
    integer_accuracy_reward,
    keyword_reward,
    length_penalty,
    normalize_text,
)


def test_integer_extraction_and_exact_reward() -> None:
    assert extract_first_integer("answer is -12, then 7") == -12
    assert extract_first_integer("no digits") is None
    assert integer_accuracy_reward("The answer is 19.", "19") == 1.0
    assert integer_accuracy_reward("The answer is 18.", "19") == 0.0


def test_keyword_and_normalize() -> None:
    assert normalize_text("  LoRA\nAdapter ") == "lora adapter"
    assert contains_keyword("This mentions a Tokenizer.", "tokenizer")
    assert keyword_reward("SFT is supervised fine-tuning.", "SFT") == 1.0
    assert keyword_reward("No relevant term.", "LoRA") == 0.0


def test_length_penalty() -> None:
    assert length_penalty("short", max_chars=10) == 0.0
    assert length_penalty("x" * 50, max_chars=10) < 0.0


def test_combined_reward_breakdown_for_math_and_keyword() -> None:
    math_example = {"category": "math_add", "reward_type": "exact_integer", "answer": "7", "keyword": ""}
    reward, breakdown = combined_reward("7", math_example)
    assert reward >= 1.0
    assert breakdown["exact_accuracy_reward"] == 1.0
    assert breakdown["exact_accuracy"] == 1.0

    keyword_example = {"category": "concept_keyword", "reward_type": "keyword", "answer": "LoRA", "keyword": "LoRA"}
    reward, breakdown = combined_reward("LoRA uses adapters.", keyword_example)
    assert reward >= 1.0
    assert breakdown["keyword_reward"] == 1.0
    assert breakdown["total_reward"] == reward
