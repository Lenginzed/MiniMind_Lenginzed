# Stage 5 GRPO Data Report

## Dataset

- Train path: `data/grpo/grpo_train.jsonl`
- Val path: `data/grpo/grpo_val.jsonl`
- Train rows: 800
- Val rows: 100
- Note: Synthetic local reward data only for GRPO pipeline validation; it is not real RLHF/RLVR data.

## Category Distribution

Train: `{"concept_keyword": 160, "format_echo": 160, "math_add": 160, "math_mul_small": 160, "math_sub": 160}`

Val: `{"concept_keyword": 20, "format_echo": 20, "math_add": 20, "math_mul_small": 20, "math_sub": 20}`

## Reward Type Distribution

Train: `{"exact_integer": 480, "exact_text": 160, "keyword": 160}`

Val: `{"exact_integer": 60, "exact_text": 20, "keyword": 20}`

## Prompt Length Stats

- Train prompt length min/mean/max: 36 / 39.8062 / 49
- Val prompt length min/mean/max: 36 / 39.4800 / 49
- Truncated prompts train/val: 0 / 0
- Max prompt length: 96

## Answer Distribution Summary

- Train unique answers: 105
- Val unique answers: 52
- Train answer top10: `{"0": 31, "DONE": 31, "DPO": 31, "LoRA": 32, "OK": 29, "PASS": 34, "READY": 34, "SAFE": 32, "SFT": 33, "tokenizer": 37}`
- Val answer top10: `{"16": 3, "60": 3, "DONE": 3, "DPO": 3, "OK": 4, "READY": 3, "SAFE": 8, "SFT": 3, "reward": 3, "tokenizer": 9}`

This is synthetic local reward data for GRPO pipeline validation only, not real RLHF/RLVR data.
