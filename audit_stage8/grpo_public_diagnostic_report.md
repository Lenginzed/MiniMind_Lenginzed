# Stage 8 Public-Policy GRPO Diagnostic Report

GRPO data is local verifiable reward by design; this is not public RLVR. The policy checkpoint is public Full SFT.

## Full GRPO Diagnostic

- Steps: `150`
- Params/trainable: `45631296` / `45631296`
- reward_mean: `0.1096` -> `0.2300`
- reward_std: `0.0215` -> `0.0000`
- zero_std: `0.5000` -> `1.0000`
- exact_accuracy: `0.0000` -> `0.0000`
- approx_kl final: `0.000338`
- rollout samples: `outputs\stage8_public\grpo_public_full\samples\rollout_samples.jsonl`

## GRPO-LoRA Diagnostic

- Steps: `150`
- Params/trainable: `45938496` / `307200`
- reward_mean: `0.1840` -> `0.2269`
- reward_std: `0.0513` -> `0.0083`
- zero_std: `0.0000` -> `0.5000`
- exact_accuracy: `0.0000` -> `0.0000`
- approx_kl final: `-0.000389`
- rollout samples: `outputs\stage8_public\grpo_public_lora\samples\rollout_samples.jsonl`

Diagnostic conclusion: Full GRPO collapsed to zero group reward std by the end; LoRA-GRPO retained some group diversity but did not improve exact-answer accuracy.