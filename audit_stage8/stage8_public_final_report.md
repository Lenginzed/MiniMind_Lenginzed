# Stage 8 Public Long-Run Final Report

Stage 8 trained the existing from-scratch mini-LLM pipeline on real public dataset subsets. It complements Stage 7 synthetic long-run results and does not claim real LLM capability.

## Public Data Sources

- Pretrain: `Salesforce/wikitext` / `wikitext-2-raw-v1` / `wikitext-2-raw-v1/train-00000-of-00001.parquet`, fallback=`False`, bytes=`10717316`, lines=`16929`
- SFT: `tatsu-lab/alpaca` / `data/train-00000-of-00001-a09b74b3ef9c3b56.parquet`, fallback=`False`, train/val=`20000`/`2000`
- DPO: `tatsu-lab/alpaca_farm` / `alpaca_gpt4_preference.json`, fallback=`False`, train/val=`10000`/`1000`
- GRPO: local verifiable reward data by design; not public RLVR.
- Tokenizer vocab/token blocks: `12000` vocab, `2495217` tokens, train/val blocks `9552`/`194`

## Training Results

- Public pretrain: `1500` steps, train loss `9.5015` -> `4.2189`, eval loss `9.5070` -> `4.7385`, best eval ppl `114.2674`
- Full SFT: `1500` steps, train loss `7.0393` -> `3.9252`, eval loss `4.0156`, ppl `55.4577`
- LoRA-SFT: `1500` steps, train loss `6.5874` -> `5.3978`, eval loss `5.2418`, trainable ratio `0.006687`
- Full DPO: `1000` steps, train loss `0.6926` -> `1.0169`, final eval margin `0.1615`, accuracy `0.5500`
- DPO-LoRA: `1000` steps, train loss `0.6929` -> `0.7124`, final eval margin `0.0582`, accuracy `0.5250`
- Full GRPO diagnostic: `150` steps, reward_mean `0.1096` -> `0.2300`, zero_std `0.5000` -> `1.0000`, exact `0.0000` -> `0.0000`
- GRPO-LoRA diagnostic: `150` steps, reward_mean `0.1840` -> `0.2269`, zero_std `0.0000` -> `0.5000`, exact `0.0000` -> `0.0000`

## Answers To Audit Questions

- Public pretrain loss/ppl clearly declined on WikiText-2, but final perplexity is still high because the model/data/steps are small.
- Public SFT samples are more natural English than Stage 7 synthetic in places, but still hallucinate and often fail task semantics.
- Public DPO did not immediately saturate to 1.0 accuracy; final eval accuracy was about 0.55 full and 0.525 LoRA.
- DPO reward margin stayed small/noisy compared with synthetic DPO, which is a useful realism signal.
- Public-policy GRPO did not solve exact-answer reward tasks; full GRPO collapsed to zero group reward std, while LoRA retained some diversity but exact accuracy stayed 0.
- README-safe claim: the self-built stack was exercised on public WikiText/Alpaca/AlpacaFarm subsets with reproducible metrics and comparison plots.
- Do not claim the model has real instruction-following, preference alignment, or mathematical reasoning ability.

## Plots

- `outputs/stage8_public/plots/pretrain_public_loss_ppl.png`
- `outputs/stage8_public/plots/public_vs_synthetic_compare.png`
- `outputs/stage8_public/plots/sft_public_full_vs_lora.png`
- `outputs/stage8_public/plots/dpo_public_margin_acc.png`
- `outputs/stage8_public/plots/grpo_public_diagnostics.png`
