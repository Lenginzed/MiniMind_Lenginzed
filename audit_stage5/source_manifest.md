# Stage 5 Source Manifest

## `minillm/grpo_data.py`

- Lines: 120
- Main symbols: def load_grpo_jsonl, class GRPODataset, def grpo_collate_fn, def write_jsonl

## `minillm/grpo_rewards.py`

- Lines: 134
- Main symbols: def extract_first_integer, def normalize_text, def contains_keyword, def is_reasonable_length, def _replacement_char_ratio, def format_reward, def integer_accuracy_reward, def keyword_reward, def length_penalty, def exact_text_reward, def combined_reward

## `minillm/grpo_trainer.py`

- Lines: 689
- Main symbols: def load_policy_model, def trainable_stats, def _pad_1d, def build_prompt_completion_batch, def token_logps_for_labels, def compute_group_advantages, def grpo_loss, def mean_float, def std_float, def rollout_batch, def evaluate_grpo, def save_grpo_checkpoint, def append_rollout_samples, def run_grpo

## `scripts/create_grpo_dataset.py`

- Lines: 148
- Main symbols: def math_add, def math_sub, def math_mul_small, def format_echo, def concept_keyword, def build_rows, def summarize, def main

## `scripts/train_grpo.py`

- Lines: 28
- Main symbols: def main

## `scripts/eval_grpo.py`

- Lines: 95
- Main symbols: def build_model, def main

## `tests/test_grpo_rewards.py`

- Lines: 44
- Main symbols: def test_integer_extraction_and_exact_reward, def test_keyword_and_normalize, def test_length_penalty, def test_combined_reward_breakdown_for_math_and_keyword

## `tests/test_grpo_advantage.py`

- Lines: 31
- Main symbols: def test_group_advantage_mean_zero_for_nonconstant_groups, def test_zero_std_group_advantage_is_zero_and_recorded, def test_invalid_group_size_raises

## `tests/test_grpo_loss.py`

- Lines: 62
- Main symbols: class FixedLogitModel, def test_build_prompt_completion_batch_masks_prompt, def test_token_logps_for_labels_only_response_tokens, def test_grpo_clipped_loss_finite_and_shapes, def test_grpo_loss_all_zero_advantage_does_not_crash

## `tests/test_grpo_trainer_smoke.py`

- Lines: 131
- Main symbols: def setup_tiny_grpo_run, def test_full_grpo_trainer_cpu_smoke, def test_lora_grpo_trainer_cpu_smoke

## `configs/grpo_full.yaml`

- Lines: 30
- Main symbols: YAML config

## `configs/grpo_lora.yaml`

- Lines: 39
- Main symbols: YAML config
