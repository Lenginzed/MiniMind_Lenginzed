# Stage 3 Code Review

## Assistant-Only Labels

- `minillm/sft_data.py` builds `input_ids` as BOS + prompt + assistant output + EOS.
- Labels are `-100` for BOS and user/prompt tokens.
- Assistant output tokens and EOS remain valid labels.
- Padding collate pads `input_ids` with pad id, `labels` with `-100`, and `attention_mask` with 0.
- Tests verify prompt masking, valid assistant labels, EOS label, and padding behavior.

## LoRA Implementation

- `LoRALinear` wraps an existing `nn.Linear` and freezes the base layer.
- LoRA update is `base(x) + scaling * B(A(dropout(x)))` with `scaling = alpha / r`.
- A is Kaiming initialized and B is zero initialized, so initial wrapped output matches base output.
- `apply_lora` only replaces requested module names; Stage 3 targets `q_proj` and `v_proj`.

## Freeze / Trainable Params

- LoRA-SFT freezes the full base model and trains only `lora_A`/`lora_B` weights.
- Tests confirm trainable parameter names are only LoRA parameters.
- Full SFT sets all parameters trainable.

## SFT Trainer

- Loads Stage 2.1 pretrain checkpoint.
- Supports full SFT and LoRA-SFT from the same base checkpoint.
- Uses assistant-only labels, AdamW, scheduler/warmup, grad clipping, eval loss, metrics JSONL/CSV, TensorBoard, and samples.
- This is a custom PyTorch loop, not Hugging Face Trainer/PEFT/TRL.

## Checkpoint / Adapter Saving

- Full and LoRA runs save `checkpoints/best.pt` and `checkpoints/last.pt`.
- LoRA runs also save `adapters/best_adapter.pt` and `adapters/last_adapter.pt` plus adapter JSON config.
- Checkpoints include model state, optimizer state, scheduler state, config, model_config, best_eval_loss, and mode.

## TODO

- DPO
- GRPO
- quantization
- KV cache
- real public instruction data
- better generation/evaluation protocol
