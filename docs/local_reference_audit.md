# Local Reference Audit

This audit checks tracked repository files for local environment references and other information that should not appear in a public GitHub release.

## Scope

Only Git-tracked files were searched. Ignored local artifacts such as `outputs/`, `data/`, caches, checkpoints, adapters, and TensorBoard logs were not included.

## Search Terms

The following terms were searched with `git grep`:

- `YSJ`
- `YSJAirCombat`
- `D:\`
- `F:\`
- `C:\`
- `anaconda3`
- `Lenginzed\MiniMInd`
- `/mnt/data`
- `sandbox:`
- `18980494367`
- `黎镇东`
- `电子科技大学`
- `aircombat`
- `AirCombat`
- `JSBSim`
- `LAG`
- `env_setup_ysj`
- `verify_ysj`

## Initial Findings

The first scan found local environment references in:

- `README.md`
- `docs/env_setup_ysj.md`
- `docs/stage1_model_design.md`
- `scripts/verify_ysj_env.py`
- `scripts/build_stage7_report.py`
- Stage audit summaries under `audit_stage1/` through `audit_stage8/`

The recurring issue was the original local Python executable path and environment name used during the author's audited Windows run. Some Stage 8 network audit files also contained local Windows cache or SSL certificate paths.

No hits were found for personal phone number, Chinese name, university name, `/mnt/data`, `sandbox:`, unrelated air-combat terms, `JSBSim`, or `LAG`.

## Changes Made

- Renamed `docs/env_setup_ysj.md` to `docs/environment.md`.
- Renamed `scripts/verify_ysj_env.py` to `scripts/verify_local_env.py`.
- Updated README links to point to `docs/environment.md`.
- Rewrote `docs/environment.md` as a generic open-source environment setup guide.
- Replaced specific local Python paths with `<local_python_executable>` in historical audit summaries.
- Replaced local SSL/cache/process paths with placeholders such as `<local_ssl_cafile>`, `<hf_datasets_cache>`, `<openssl_cafile>`, `<openssl_capath>`, and `<windows_process_path>`.
- Replaced local environment-name wording with generic "local Conda environment" or "local Python environment" language.

## Final Result

Final tracked-file scans returned no hits for:

- `YSJ`
- `YSJAirCombat`
- `D:\`
- `F:\`
- `C:\`
- `anaconda3`
- `Lenginzed\MiniMInd`
- `/mnt/data`
- `sandbox:`
- personal phone/name/university strings
- unrelated air-combat/JSBSim/LAG terms

The repository still contains generic placeholders such as `<local_python_executable>` in historical audit summaries. These are intentionally retained to show where local commands were executed without exposing a machine-specific path.
