# Environment Setup

## Recommended Open-Source Setup

Create a fresh environment for this repository:

```bash
conda create -n minimind python=3.10 -y
conda activate minimind
pip install -r requirements-minimal.txt
```

Or use `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
```

On Windows, activate a `venv` with:

```powershell
.venv\Scripts\activate
```

PyTorch CUDA builds are hardware-specific. If the default `pip install -r requirements-minimal.txt` does not install the right PyTorch build for your machine, install PyTorch separately using the official PyTorch instructions for your CUDA or CPU environment.

## Verify The Environment

Run the generic local environment verifier:

```bash
python scripts/verify_local_env.py
```

To regenerate local audit files:

```bash
python scripts/verify_local_env.py --write-audit
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

The minimal dependency file is intended for tests, scripts, public-data prep, plotting, and local training experiments:

```bash
pip install -r requirements-minimal.txt
```

It intentionally does not include vLLM, flash-attn, DeepSpeed, or bitsandbytes.

## Original Audit Context

The original experiments were audited on a Windows workstation with an RTX 4080 SUPER, working CUDA, and bf16 support. The exact local Conda path is not required for reproduction and is intentionally not documented as a public setup requirement.

For new machines, prefer a clean environment over reusing a pre-existing base environment. If `python scripts/verify_local_env.py` reports that PyTorch imports, CUDA is available, and fp32/fp16/bf16 checks pass, the environment is suitable for the small GPU experiments in this project.

## Deferred Libraries

Do not install these unless you intentionally extend the project:

- `vLLM`
- `flash-attn`
- `DeepSpeed`
- `bitsandbytes`
