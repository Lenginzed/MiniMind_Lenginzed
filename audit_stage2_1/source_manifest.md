# Stage 2.1 Source Manifest

## `minillm/model.py`

- Lines: `73`
- Classes: MiniLLMForCausalLM
- Functions: count_parameters

## `minillm/data.py`

- Lines: `201`
- Classes: CausalLMBlockDataset
- Functions: encode_text_file, split_tokens, validate_token_ids, validate_block_size, split_random_blocks, save_tokenized_splits, load_token_array, load_block_datasets

## `minillm/trainer.py`

- Lines: `540`
- Classes: none
- Functions: build_model_config, build_lr_scheduler, lr_scale_for_step, reset_optimizer_lr, cycle_loader, move_batch, current_lr, cuda_memory_record, evaluate, save_checkpoint, write_sample, _existing_first_train_loss, run_pretrain

## `minillm/utils.py`

- Lines: `105`
- Classes: none
- Functions: ensure_dir, load_yaml, save_yaml, save_json, load_json, append_jsonl, iter_jsonl, set_seed, get_device, safe_perplexity, count_lines, file_size_bytes, resolve_dtype, autocast_context

## `scripts/create_mixed_corpus.py`

- Lines: `137`
- Classes: none
- Functions: make_line, main

## `scripts/tokenize_corpus.py`

- Lines: `44`
- Classes: none
- Functions: main

## `scripts/train_pretrain.py`

- Lines: `28`
- Classes: none
- Functions: main

## `scripts/plot_training_curves.py`

- Lines: `57`
- Classes: none
- Functions: main

## `configs/pretrain_stage2_hardened.yaml`

- Lines: `44`
- Classes: n/a
- Functions: n/a

## `configs/pretrain_resume_smoke.yaml`

- Lines: `42`
- Classes: n/a
- Functions: n/a

## `tests/test_gradient_checkpointing.py`

- Lines: `39`
- Classes: none
- Functions: run_one_backward, test_gradient_checkpointing_true_forward_backward, test_gradient_checkpointing_false_forward_backward

## `tests/test_resume_training.py`

- Lines: `80`
- Classes: none
- Functions: test_resume_training_step_increases_and_checkpoint_loads

## `tests/test_scheduler.py`

- Lines: `26`
- Classes: none
- Functions: test_cosine_scheduler_warmup_then_decay, test_none_scheduler_returns_none

## `tests/test_data_pipeline.py`

- Lines: `88`
- Classes: none
- Functions: build_tokenizer, test_block_dataset_shape_and_labels, test_train_val_split_not_empty, test_save_tokenized_splits_metadata, test_random_blocks_split_reproducible, test_random_blocks_too_small_raises, test_block_size_larger_than_context_raises
