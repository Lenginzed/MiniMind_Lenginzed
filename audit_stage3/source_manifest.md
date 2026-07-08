# Stage 3 Source Manifest

## `minillm/sft_data.py`

- Lines: `177`
- Classes: SFTDataset
- Functions: format_prompt, format_full_text, load_sft_jsonl, encode_sft_example, sft_collate_fn, write_jsonl

## `minillm/lora.py`

- Lines: `154`
- Classes: LoRALinear
- Functions: freeze_all_parameters, _set_lora_trainable, apply_lora, lora_parameter_stats, lora_state_dict, save_lora_adapter, load_lora_adapter

## `minillm/sft_trainer.py`

- Lines: `351`
- Classes: none
- Functions: load_base_model, sft_evaluate, save_sft_checkpoint, write_sft_samples, run_sft

## `scripts/create_sft_dataset.py`

- Lines: `134`
- Classes: none
- Functions: make_examples, main

## `scripts/train_sft.py`

- Lines: `26`
- Classes: none
- Functions: main

## `scripts/eval_sft.py`

- Lines: `98`
- Classes: none
- Functions: build_model_from_config, main

## `configs/sft_full.yaml`

- Lines: `30`
- Classes: n/a
- Functions: n/a

## `configs/sft_lora.yaml`

- Lines: `39`
- Classes: n/a
- Functions: n/a

## `tests/test_sft_data.py`

- Lines: `61`
- Classes: none
- Functions: build_tokenizer, test_assistant_only_labels, test_sft_collate_padding_labels, test_sft_dataset_stats

## `tests/test_lora.py`

- Lines: `42`
- Classes: none
- Functions: test_lora_linear_shape_and_zero_b_equivalence, test_apply_lora_only_targets_q_v_and_freezes_base

## `tests/test_sft_trainer_smoke.py`

- Lines: `98`
- Classes: none
- Functions: setup_tiny_sft_run, test_full_sft_trainer_cpu_smoke, test_lora_sft_trainer_cpu_smoke
