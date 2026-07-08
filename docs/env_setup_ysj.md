# YSJAirCombat Environment Audit Notes

## Original Audited Local Environment

The original audited local run used:

```powershell
D:\anaconda3\envs\YSJAirCombat\python.exe
```

This path belongs to the author's Windows workstation and is not required for external users. For the author's machine, `YSJAirCombat` was preferred because the Anaconda `base` environment had a broken PyTorch install: package metadata existed, but `import torch` failed while loading `fbgemm.dll`. That made `base` unsuitable for CUDA validation or later training work.

`YSJAirCombat` was suitable for the audited runs because PyTorch imported successfully, CUDA was visible, and RTX 4080 SUPER matmul smoke tests passed.

External users should create their own environment and install the project requirements from `requirements-minimal.txt`.

## Verify The Environment

Run:

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\verify_ysj_env.py
```

To regenerate the Stage 0.5 audit files:

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\verify_ysj_env.py --write-audit
```

The script prints:

- Python executable
- torch version
- `torch.version.cuda`
- `torch.cuda.is_available()`
- GPU name and VRAM
- fp32/fp16/bf16 CUDA matmul checks
- PyTorch SDPA availability
- missing minimal packages

## Minimal Dependencies

For the original audited local environment, missing dependencies could be installed with:

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' -m pip install -r requirements-minimal.txt
```

External users can use the generic README quick-start commands instead. Only install missing minimal packages when needed. At audit time, the local environment already had most required packages, but `pytest` was missing.

## If CUDA Is Healthy

For the author's local machine, if `verify_ysj_env.py` reports that torch imports, CUDA is available, and fp32/fp16/bf16 matmul tests pass, `YSJAirCombat` can be used for small GPU experiments.

Do not install large training stack extras yet. Add libraries in stages:

- Stage 1/2: `numpy`, `tqdm`, `matplotlib`, `tensorboard`, `pyyaml`, `pytest`, `safetensors`
- Stage 3+: evaluate `transformers`, `datasets`, `accelerate`, `peft`, `trl`
- QLoRA/quantization stage: evaluate `bitsandbytes`

## If CUDA Is Not Healthy

Only if `YSJAirCombat` also fails `import torch` or `torch.cuda.is_available()`, create a clean `mini_llm` environment. Do not create it preemptively.

## Deferred Libraries

Do not install these for the current stage:

- `vLLM`: useful later for serving-style inference, not needed for tiny model skeletons
- `flash-attn`: Windows installation can be high-friction; PyTorch SDPA is enough for Stage 1
- `DeepSpeed`: unnecessary for single-GPU mini models at this stage
- `bitsandbytes`: wait until Stage 3/QLoRA or quantization experiments
