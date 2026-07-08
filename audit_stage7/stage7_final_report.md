# Stage 7 Final Report

Stage 7 runs are local long-run / resume-ready demonstrations for README and interview discussion. They do not claim real LLM capability.

## Run Summary

- Pretrain: steps `600`, train_loss `9.0892` -> `0.7405`
- Full SFT: steps `400`, train_loss `11.0005` -> `0.1809`
- LoRA-SFT: steps `400`, train_loss `10.9894` -> `2.0745`
- Full DPO: steps `250`, train_loss `0.6935` -> `0.0000`
- DPO-LoRA: steps `250`, train_loss `0.6931` -> `0.0001`
- Full GRPO: steps `80`, reward_mean `0.1586` -> `0.2300`
- GRPO-LoRA: steps `80`, reward_mean `0.1291` -> `0.1985`

## Plots

- `outputs/stage7/plots/pretrain_loss_ppl.png`
- `outputs/stage7/plots/sft_full_vs_lora_loss.png`
- `outputs/stage7/plots/dpo_full_vs_lora_margin_acc.png`
- `outputs/stage7/plots/grpo_reward_diagnostics.png`
- `outputs/stage7/plots/trainable_params_compare.png`
