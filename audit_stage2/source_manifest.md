# Stage 2 Source Manifest

## `minillm/tokenizer.py`

- Lines: `107`
- Classes: MiniTokenizer
- Functions: discover_text_files

## `minillm/data.py`

- Lines: `119`
- Classes: CausalLMBlockDataset
- Functions: encode_text_file, split_tokens, save_tokenized_splits, load_token_array, load_block_datasets

## `minillm/trainer.py`

- Lines: `346`
- Classes: none
- Functions: build_model_config, cycle_loader, move_batch, evaluate, save_checkpoint, write_sample, run_pretrain

## `minillm/utils.py`

- Lines: `105`
- Classes: none
- Functions: ensure_dir, load_yaml, save_yaml, save_json, load_json, append_jsonl, iter_jsonl, set_seed, get_device, safe_perplexity, count_lines, file_size_bytes, resolve_dtype, autocast_context

## `scripts/create_toy_corpus.py`

- Lines: `79`
- Classes: none
- Functions: build_lines, main

## `scripts/train_tokenizer.py`

- Lines: `59`
- Classes: none
- Functions: main

## `scripts/tokenize_corpus.py`

- Lines: `40`
- Classes: none
- Functions: main

## `scripts/train_pretrain.py`

- Lines: `26`
- Classes: none
- Functions: main

## `scripts/plot_training_curves.py`

- Lines: `57`
- Classes: none
- Functions: main

## `scripts/eval_pretrain_smoke.py`

- Lines: `82`
- Classes: none
- Functions: main

## `configs/pretrain_tiny.yaml`

- Lines: `42`
- Classes: n/a
- Functions: n/a

## `configs/model_tiny.yaml`

- Lines: `12`
- Classes: n/a
- Functions: n/a

## `configs/model_50m.yaml`

- Lines: `12`
- Classes: n/a
- Functions: n/a

## `tests/test_tokenizer_pipeline.py`

- Lines: `35`
- Classes: none
- Functions: test_toy_tokenizer_train_load_encode_decode

## `tests/test_data_pipeline.py`

- Lines: `53`
- Classes: none
- Functions: build_tokenizer, test_block_dataset_shape_and_labels, test_train_val_split_not_empty, test_save_tokenized_splits_metadata

## `tests/test_pretrain_smoke.py`

- Lines: `30`
- Classes: none
- Functions: test_tiny_pretrain_one_batch_forward_backward_cpu
