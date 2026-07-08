# Artifact Policy

This repository should stay lightweight and reviewable. Large generated artifacts should not be committed to Git.

## Commit To The Repository

- Source code under `minillm/`.
- Training and evaluation scripts under `scripts/`.
- YAML configs under `configs/`.
- Tests under `tests/`.
- Documentation under `docs/`.
- Audit report markdown/json files when they are small and useful for reproducibility.
- Selected README-ready plots copied to `docs/assets/`.
- Small metadata files that describe datasets or runs, when they do not contain raw dataset payloads.

## Do Not Commit

- Checkpoints: `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`.
- LoRA adapter tensors: `adapters/*.pt`, `adapters/*.safetensors`.
- Raw datasets: downloaded parquet/json/text files and generated raw corpora.
- Processed arrays: `.npy`, `.npz`, `.bin`, `.arrow`.
- TensorBoard event files.
- Hugging Face or local dataset caches.
- `__pycache__`, `.pytest_cache`, IDE files, and virtual environments.
- Full `outputs/` directories.

## Why

The Stage 7/8 training outputs include many checkpoints in the 170-520 MB range, and the full `outputs/` directory is over 10 GB. Keeping those in Git would make the project difficult to clone and review. The repository should contain enough code, configs, docs, selected plots, and metadata to explain and reproduce the experiments, while large artifacts can be stored separately if needed.

## Recommended Release Layout

For GitHub:

- Commit `README.md`, `docs/`, `minillm/`, `scripts/`, `configs/`, `tests/`, `.gitignore`, requirements, and the final chosen `LICENSE`.
- Commit `docs/assets/*.png` for visual summaries.
- Do not commit `outputs/`, `data/**/raw`, `data/**/processed*`, or checkpoint files.
- If sharing checkpoints later, publish them as a separate release artifact or model repository with clear warnings about limited capability.
