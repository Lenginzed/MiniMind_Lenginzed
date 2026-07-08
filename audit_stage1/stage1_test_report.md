# Stage 1 Test Report

## Commands

```powershell
& '<local_python_executable>' -m pytest -q
& '<local_python_executable>' scripts\smoke_model_forward.py
```

## Pytest Result

- Exit code: `0`
- Summary: `23 passed in 1.64s`
- Passed/failed: `23 passed / 0 failed`

```text
.......................                                                  [100%]
23 passed in 1.64s

```

## Smoke Result

- Exit code: `0`
- Summary: tiny forward/loss/backward/generation completed; CUDA memory printed when available.

```text
device: cuda
tiny config: MiniLLMConfig(vocab_size=128, context_length=32, n_layer=2, n_embd=64, n_head=4, n_kv_head=2, intermediate_size=128, rms_norm_eps=1e-06, rope_theta=10000.0, dropout=0.0, tie_word_embeddings=True, use_gradient_checkpointing=False)
parameter count: 82240
logits shape: (2, 16, 128)
loss: 4.78904914855957
loss finite: True
backward: ok
finite gradient check: True
generated shape: (2, 8)
generated first row: [83, 105, 38, 50, 48, 83, 59, 111]
cuda memory allocated bytes: 17726464
cuda memory reserved bytes: 23068672

```