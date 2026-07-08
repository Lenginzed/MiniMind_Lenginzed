from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional, Tuple


INTEGER_RE = re.compile(r"[-+]?\d+")


def extract_first_integer(text: str) -> Optional[int]:
    match = INTEGER_RE.search(text or "")
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_keyword(text: str, keyword: str) -> bool:
    if not keyword:
        return False
    return normalize_text(keyword) in normalize_text(text)


def is_reasonable_length(text: str, min_chars: int = 1, max_chars: int = 160) -> bool:
    length = len((text or "").strip())
    return min_chars <= length <= max_chars


def _replacement_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = text.count("\ufffd") + text.count("锟")
    return bad / max(1, len(text))


def format_reward(completion: str, example: Dict[str, str]) -> float:
    text = completion or ""
    reward = 0.0
    if is_reasonable_length(text, 1, 160):
        reward += 0.05
    if any(ch.isalnum() for ch in text):
        reward += 0.02
    if _replacement_char_ratio(text) > 0.08:
        reward -= 0.10
    if len(text.strip()) > 220:
        reward -= 0.10
    return reward


def integer_accuracy_reward(completion: str, answer: str) -> float:
    predicted = extract_first_integer(completion)
    expected = extract_first_integer(answer)
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0


def keyword_reward(completion: str, keyword: str) -> float:
    return 1.0 if contains_keyword(completion, keyword) else 0.0


def length_penalty(completion: str, max_chars: int = 160) -> float:
    length = len((completion or "").strip())
    if length <= max_chars:
        return 0.0
    excess = min(200, length - max_chars)
    return -0.05 - 0.0005 * excess


def exact_text_reward(completion: str, answer: str) -> float:
    completion_norm = normalize_text(completion)
    answer_norm = normalize_text(answer)
    if not completion_norm or not answer_norm:
        return 0.0
    if completion_norm == answer_norm:
        return 1.0
    if completion_norm.startswith(answer_norm):
        return 0.75
    if answer_norm in completion_norm:
        return 0.5
    return 0.0


def combined_reward(completion: str, example: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
    text = completion or ""
    reward_type = str(example.get("reward_type", ""))
    category = str(example.get("category", ""))
    answer = str(example.get("answer", ""))
    keyword = str(example.get("keyword", "")) or answer

    fmt = format_reward(text, example)
    dense_length = 0.001 * min(60, len(text.strip())) if text.strip() else 0.0
    number_presence = 0.0
    exact = 0.0
    keyword_score = 0.0
    text_exact = 0.0

    if reward_type == "exact_integer" or category.startswith("math_"):
        number_presence = 0.10 if extract_first_integer(text) is not None else 0.0
        exact = integer_accuracy_reward(text, answer)
    elif reward_type == "keyword":
        keyword_score = keyword_reward(text, keyword)
    elif reward_type == "exact_text":
        text_exact = exact_text_reward(text, answer)
        keyword_score = 0.25 if contains_keyword(text, answer) else 0.0
    else:
        keyword_score = 0.5 * keyword_reward(text, keyword)

    penalty = length_penalty(text)
    total = fmt + dense_length + number_presence + exact + keyword_score + text_exact + penalty
    breakdown = {
        "format_reward": float(fmt),
        "dense_length_reward": float(dense_length),
        "number_presence_reward": float(number_presence),
        "exact_accuracy_reward": float(exact),
        "keyword_reward": float(keyword_score),
        "exact_text_reward": float(text_exact),
        "length_penalty": float(penalty),
        "total_reward": float(total),
        "exact_accuracy": float(exact > 0.0 or keyword_score >= 1.0 or text_exact >= 1.0),
        "completion_length": float(len(text.strip())),
        "completion_empty": float(len(text.strip()) == 0),
    }
    return float(total), breakdown
