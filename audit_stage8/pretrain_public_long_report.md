# Stage 8 Public Pretrain Report

- Command: `<local_python_executable> scripts\train_pretrain.py --config configs\stage8_public\pretrain_public_long.yaml`
- Parameter count: `45631296`
- Device/dtype: `cuda` / `bf16`
- Steps: `1500`
- Tokens seen: `12288000`
- Train loss: `9.5015` -> `4.2189`
- Eval loss: `9.5070` -> `4.7385`
- Best eval ppl: `114.2674`
- CUDA max allocated/reserved: `1049344000` / `1184890880`
- Metrics: `outputs\stage8_public\pretrain_public_long\metrics.jsonl`
- Best checkpoint: `outputs\stage8_public\pretrain_public_long\checkpoints\best.pt`
- Samples: `outputs\stage8_public\pretrain_public_long\samples\before.txt`, `outputs/stage8_public/pretrain_public_long/samples/mid.txt`, `outputs\stage8_public\pretrain_public_long\samples\after.txt`

This is public WikiText-2 subset pretraining for pipeline/result comparison, not a claim of real language-model capability.