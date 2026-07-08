# Stage 8 Training Preflight

- Python executable: `<local_python_executable>`
- Torch: `2.1.0+cu121`, CUDA: `12.1`
- CUDA available: `True`
- bf16 supported: `True`
- GPU: `{'ok': True, 'gpus': [{'name': 'NVIDIA GeForce RTX 4080 SUPER', 'memory_total_mib': 16376, 'memory_used_mib': 2157, 'memory_free_mib': 13891}], 'raw': 'NVIDIA GeForce RTX 4080 SUPER, 16376, 2157, 13891\n'}`
- Disk free: `713.43` GB / `875.28` GB
- Public tokenizer vocab_size: `12000`
- Train npy: `{'exists': True, 'shape': [2445312], 'dtype': 'int32', 'tokens': 2445312}`
- Val npy: `{'exists': True, 'shape': [49664], 'dtype': 'int32', 'tokens': 49664}`
- SFT train/val rows: `20000` / `2000`
- DPO train/val rows: `10000` / `1000`
- Model parameter count: `45631296`

## Output Status

- `pretrain`: `{'path': 'outputs\\stage8_public\\pretrain_public_long', 'exists': False, 'metrics_exists': False, 'metrics_size': None}`
- `sft_full`: `{'path': 'outputs\\stage8_public\\sft_public_full', 'exists': False, 'metrics_exists': False, 'metrics_size': None}`
- `sft_lora`: `{'path': 'outputs\\stage8_public\\sft_public_lora', 'exists': False, 'metrics_exists': False, 'metrics_size': None}`
- `dpo_full`: `{'path': 'outputs\\stage8_public\\dpo_public_full', 'exists': False, 'metrics_exists': False, 'metrics_size': None}`
- `dpo_lora`: `{'path': 'outputs\\stage8_public\\dpo_public_lora', 'exists': False, 'metrics_exists': False, 'metrics_size': None}`
- `grpo_full`: `{'path': 'outputs\\stage8_public\\grpo_public_full', 'exists': False, 'metrics_exists': False, 'metrics_size': None}`
- `grpo_lora`: `{'path': 'outputs\\stage8_public\\grpo_public_lora', 'exists': False, 'metrics_exists': False, 'metrics_size': None}`

## Recommendation

- Use the 50M-tier public config with gradient checkpointing.
- Start with pretrain batch_size=4 and grad_accum=8 in bf16.
- If OOM occurs, reduce batch_size to 2 and increase grad_accum to 16, then resume/restart with an explicit audit note.
- Do not overwrite existing Stage 7 outputs.
