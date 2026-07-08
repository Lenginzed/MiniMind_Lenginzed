# Release Checklist

This checklist summarizes what is ready for open-source release and what still needs a human decision.

## Current Status

| Item | Status | Notes |
| --- | --- | --- |
| README | Ready | Uses `docs/assets/*.png`, highlights public-data long run and limitations |
| Docs | Ready | Training stack, results, limitations, data sources, model card, artifact policy, claim audit |
| Configs | Ready | Stage configs are small and reproducible |
| Source code | Ready for final review | No training was rerun during this clean pass |
| Tests | Existing | Not rerun in this docs/artifact pass |
| `.gitignore` | Ready | Excludes checkpoints, adapters, raw data, processed arrays, logs, caches, outputs |
| Requirements | Ready | `requirements-minimal.txt` excludes vLLM, flash-attn, DeepSpeed, bitsandbytes |
| LICENSE | Pending user choice | Choose MIT or Apache-2.0 before publishing |

## Repository Size Audit

Measured before `.gitignore` filtering:

| Path | Approx size | Recommendation |
| --- | ---: | --- |
| `outputs/` | 10.47 GB | Do not commit |
| `data/` | 138.39 MB | Do not commit raw/processed data |
| `docs/` | 0.83 MB | Commit |
| `minillm/` | 0.28 MB | Commit |
| `scripts/` | 0.39 MB | Commit |
| `tests/` | 0.20 MB | Commit |
| `configs/` | 0.02 MB | Commit |
| `audit_stage7/` | 0.02 MB | Optional, useful for reproducibility |
| `audit_stage8/` | 0.06 MB | Optional, useful for reproducibility |

Largest files found:

| Pattern | Example size | Action |
| --- | ---: | --- |
| Stage 8 full checkpoints | ~522 MB each | Ignored |
| Stage 7 full checkpoints | ~496 MB each | Ignored |
| Stage 8 LoRA checkpoints | ~178 MB each | Ignored |
| Stage 7 LoRA checkpoints | ~171 MB each | Ignored |
| Downloaded public parquet/json | 14-23 MB | Ignored |
| Raw generated/public corpora | 10-20 MB | Ignored |
| Processed `.npy` arrays | 16 MB+ | Ignored |

## Recommended Commit Scope

Commit:

- `README.md`
- `.gitignore`
- `requirements-minimal.txt`
- `docs/`
- `minillm/`
- `scripts/`
- `configs/`
- `tests/`
- Optional small `audit_stage7/` and `audit_stage8/` markdown/json summaries

Do not commit:

- `outputs/`
- `data/**/raw`
- `data/**/processed*`
- `data/**/download_cache`
- Checkpoints and adapters
- TensorBoard logs
- Python caches

## License Decision

The repository currently has no existing license. Before `git init`, `commit`, or `push`, choose one:

- MIT: short and permissive.
- Apache-2.0: permissive and includes an explicit patent grant.

After choosing, add the corresponding `LICENSE` file at the repository root.

## Final Pre-Push Steps

1. Confirm license choice and add `LICENSE`.
2. Run `git status --ignored` after `git init` to verify large artifacts are ignored.
3. Review README rendered on GitHub, especially Mermaid and image paths.
4. Run tests if you want a fresh release badge/status: `python -m pytest -q`.
5. Keep model/checkpoint artifacts out of the repository unless published separately with the model card.
