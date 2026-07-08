# Stage 4 Source Manifest

## `minillm/dpo_data.py`

- Lines: 168
- Main symbols: def load_dpo_jsonl, def dpo_prompt, def encode_prompt_response, class DPODataset, def _pad_1d, def dpo_collate_fn, def write_jsonl

## `minillm/dpo_trainer.py`

- Lines: 375
- Main symbols: def sequence_logps, def dpo_loss, def load_policy_model, def freeze_model, def trainable_stats, def dpo_batch_metrics, def mean_item, def evaluate_dpo, def save_dpo_checkpoint, def run_dpo

## `minillm/lora.py`

- Lines: 154
- Main symbols: class LoRALinear, def freeze_all_parameters, def _set_lora_trainable, def apply_lora, def lora_parameter_stats, def lora_state_dict, def save_lora_adapter, def load_lora_adapter

## `minillm/sft_trainer.py`

- Lines: 355
- Main symbols: def load_base_model, def sft_evaluate, def save_sft_checkpoint, def write_sft_samples, def run_sft

## `scripts/create_dpo_dataset.py`

- Lines: 240
- Main symbols: def concept_example, def math_example, def translation_example, def flight_rl_example, def format_example, def code_example, def rejected_for, def reason_for, def build_rows, def summarize, def main

## `scripts/train_dpo.py`

- Lines: 28
- Main symbols: def main

## `scripts/eval_dpo.py`

- Lines: 110
- Main symbols: def build_model, def main

## `scripts/eval_sft.py`

- Lines: 102
- Main symbols: def build_model_from_config, def main

## `tests/test_dpo_data.py`

- Lines: 115
- Main symbols: def build_tokenizer, def test_dpo_dataset_masks_prompt_and_keeps_eos, def test_dpo_collate_padding_labels_are_ignored, def test_dpo_truncation_can_skip_when_response_is_fully_cut

## `tests/test_dpo_loss.py`

- Lines: 48
- Main symbols: class FixedLogitModel, def test_sequence_logps_only_counts_unmasked_response_tokens, def test_dpo_loss_finite_and_prefers_better_policy_margin

## `tests/test_dpo_trainer_smoke.py`

- Lines: 130
- Main symbols: def setup_tiny_dpo_run, def test_freeze_model_marks_reference_params_non_trainable, def test_full_dpo_trainer_cpu_smoke, def test_lora_dpo_trainer_cpu_smoke

## `configs/dpo_full.yaml`

- Lines: 32
- Main symbols: YAML config

## `configs/dpo_lora.yaml`

- Lines: 41
- Main symbols: YAML config
