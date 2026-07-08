# Stage 8 Public Dataset Interview Update

## One-Liner

After validating the mini-LLM stack on synthetic data, I moved the same from-scratch pipeline onto public WikiText-2, Alpaca, and AlpacaFarm subsets and compared the resulting training dynamics against the Stage 7 synthetic baseline.

## What I Ran

- WikiText-2 raw public pretraining with a 45.63M parameter decoder-only model.
- Alpaca assistant-only SFT and LoRA-SFT.
- AlpacaFarm DPO and DPO-LoRA with a frozen reference model.
- Public-SFT-policy GRPO diagnostics on local verifiable rewards.

## Interview Talking Points

- Public pretraining loss fell substantially, but public data remained harder than synthetic data.
- Alpaca SFT improved loss and produced more natural English-shaped completions, but outputs remain unreliable.
- AlpacaFarm DPO did not saturate preference accuracy, unlike synthetic DPO; this is a good sign that the public preference task is less trivial.
- GRPO diagnostics showed why reward variance matters: full GRPO reached zero group std, while LoRA kept some diversity but did not improve exact accuracy.
- LoRA trained only about 0.67% of parameters for q/v adapters, demonstrating parameter-efficient fine-tuning mechanics.

## README-Safe Result

The project now demonstrates the same self-built LLM lifecycle on public dataset subsets, with checkpoints, metrics, curves, and honest limitations.

## Do Not Overclaim

- Do not call the model capable or aligned.
- Do not describe GRPO as public RLVR.
- Do not treat generated examples as proof of instruction following.
- Do not claim DPO solved preference learning; accuracy stayed near chance-to-modest levels.

## Key Files

- `audit_stage8/stage8_public_final_report.md`
- `outputs/stage8_public/plots/pretrain_public_loss_ppl.png`
- `outputs/stage8_public/plots/sft_public_full_vs_lora.png`
- `outputs/stage8_public/plots/dpo_public_margin_acc.png`
- `outputs/stage8_public/plots/grpo_public_diagnostics.png`
