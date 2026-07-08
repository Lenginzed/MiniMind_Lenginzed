# Stage 2.1 Code Review

## Gradient Checkpointing

- `MiniLLMForCausalLM.forward` now uses `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` for decoder layers when `config.use_gradient_checkpointing=True` and `model.training=True`.
- Eval and generation paths do not checkpoint because `model.training` is false.
- Tests cover forward/backward with checkpointing both true and false.

## Resume Training

- `scripts/train_pretrain.py` supports `--resume PATH` and `--max-steps` override.
- `run_pretrain` restores model state, optimizer state, scheduler state when compatible, step, best eval loss, and tokens seen.
- Checkpoints contain model, optimizer, scheduler, step, config, model_config, best_eval_loss, and tokens_seen.
- Resume appends a `resume` event to `metrics.jsonl` rather than creating a new run directory.

## Scheduler

- Supports `scheduler: none`, `linear`, and `cosine` with `warmup_steps`.
- LR is logged every step.
- Tests cover warmup/cosine behavior and none scheduler.

## Random Blocks Split

- `tokenize_corpus.py` supports `--split-mode contiguous|random_blocks`.
- `random_blocks` cuts full blocks first, shuffles block indices with a fixed seed, and writes flattened train/val arrays.
- Metadata records split mode, seed, total/train/val blocks, and dropped tail tokens.
- Tests cover non-empty splits and reproducibility.

## Metrics / Logging

- Per-step JSONL now records step, train/eval loss, train/eval perplexity, lr, grad_norm, tokens_seen, approximate_epoch, and CUDA memory allocated/reserved/max values when CUDA is available.
- CSV and TensorBoard logging are still written.
- Plotting ignores non-training event rows such as resume markers.

## Current Limits

- The mixed corpus is still locally generated and synthetic.
- The hardened model is still small and trained only for a short smoke run.
- This stage does not claim real language ability.
- SFT, LoRA, DPO, GRPO, real public datasets, KV cache, quantization, vLLM, flash-attn, DeepSpeed, and bitsandbytes remain out of scope.
