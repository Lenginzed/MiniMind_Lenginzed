# Stage 2 Code Review

## Tokenizer Pipeline

- Uses Hugging Face `tokenizers` with Byte-level BPE.
- Special tokens are explicit: `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- Supports train/save/load/encode/decode and reports vocab size plus special token ids.
- Training script saves a sample encode/decode JSON next to the tokenizer.

## Dataset / Block Packing

- Text is encoded line by line with BOS/EOS boundaries.
- Encoded ids are saved as `int32` `.npy` arrays plus metadata JSON.
- Train/val split is contiguous and non-empty; this is sufficient for smoke validation.
- `CausalLMBlockDataset` packs non-overlapping fixed-length blocks and returns `input_ids` plus identical `labels`; the model performs the causal shift internally.

## Train Loop

- Custom PyTorch loop, no Hugging Face Trainer.
- Supports YAML config, CUDA auto-selection, bf16/fp16/fp32 selection, AdamW, gradient accumulation, gradient clipping, train/eval loss, safe perplexity, TensorBoard, JSONL/CSV metrics, and checkpointing.
- Tiny smoke used CUDA + bf16 on RTX 4080 SUPER and ran 120 optimizer steps.

## Checkpointing

- Saves `checkpoints/last.pt` and `checkpoints/best.pt`.
- Checkpoint contains model state, optimizer state, step, resolved config, model config, and best eval loss.
- Resume is intentionally left as TODO for a later iteration.

## Metrics / Logging

- `metrics.jsonl` records every step.
- `metrics.csv` mirrors key fields for quick inspection.
- TensorBoard logs are written under `outputs/pretrain_tiny/logs/`.
- `plot_training_curves.py` creates `loss_curve.png` with train and eval loss.
- On this Windows/Anaconda environment, matplotlib initially hit a duplicate OpenMP runtime warning. The plot-only script now sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing matplotlib; this does not affect training kernels.

## Current Limitations

- Corpus is synthetic and tiny.
- Model is tiny and trained for only 120 steps.
- Output samples are only smoke-test artifacts and should not be read as model capability.
- No real tokenizer/data curation, no scheduler, no resume implementation, no distributed training.
- No SFT, LoRA, DPO, GRPO, GPTQ, SmoothQuant, vLLM, flash-attn, DeepSpeed, or bitsandbytes in this stage.
