# Stage 2 Train Report

## Command

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\train_pretrain.py --config configs\pretrain_tiny.yaml
```

## Config Summary

- Model: tiny decoder-only Causal LM
- Parameter count: `423552`
- Device: `cuda`
- Dtype: `bf16`
- Max steps: `120`
- Batch size: `16`
- Gradient accumulation: `1`
- Learning rate: `3e-4`
- Weight decay: `0.1`
- Grad clip: `1.0`

## Loss / Perplexity

- Initial train loss: `6.9878`
- Final train loss: `1.6551`
- Initial train perplexity: `1083.32`
- Final train perplexity: `5.23`
- Initial eval loss: `6.8485`
- Final/best eval loss: `1.6858`
- Initial eval perplexity: `942.46`
- Final/best eval perplexity: `5.40`

This loss drop is expected on a tiny repetitive toy corpus. It proves the training loop and data flow work; it is not evidence of general language ability.

## Artifacts

- Metrics JSONL: `outputs\pretrain_tiny\metrics.jsonl`
- Last checkpoint: `outputs\pretrain_tiny\checkpoints\last.pt`
- Best checkpoint: `outputs\pretrain_tiny\checkpoints\best.pt`
- Loss curve: `outputs/pretrain_tiny/plots/loss_curve.png`
- Before samples: `outputs\pretrain_tiny\samples\before.txt`
- After samples: `outputs\pretrain_tiny\samples\after.txt`
- Resolved config: `outputs/pretrain_tiny/train_config_resolved.yaml`
- TensorBoard logs: `outputs/pretrain_tiny/logs/`

## Resolved Config

```yaml
seed: 20260707
output_dir: outputs/pretrain_tiny
tokenizer_path: data/tokenizers/toy_tokenizer.json
train_data_path: data/processed/train.npy
val_data_path: data/processed/val.npy
prefer_cuda: true
dtype: auto
data:
  block_size: 64
model:
  vocab_size: 1000
  context_length: 64
  n_layer: 2
  n_embd: 128
  n_head: 4
  n_kv_head: 2
  intermediate_size: 256
  rms_norm_eps: 1.0e-06
  rope_theta: 10000.0
  dropout: 0.0
  tie_word_embeddings: true
  use_gradient_checkpointing: false
training:
  batch_size: 16
  gradient_accumulation_steps: 1
  max_steps: 120
  eval_interval: 20
  save_interval: 50
  log_interval: 10
  eval_batches: 10
  learning_rate: 0.0003
  weight_decay: 0.1
  grad_clip: 1.0
sample_prompts:
- Mini language models
- RoPE rotates
- 小模型
- The pilot checks
```
