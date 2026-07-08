# Data Sources

This document records the data sources used by the public-data long run and the boundary between public datasets and local diagnostic data.

## Stage 8 Public Sources

| Use | Source | Local conversion | Notes |
| --- | --- | --- | --- |
| Pretraining | `Salesforce/wikitext`, `wikitext-2-raw-v1` | `data/stage8_public/raw/pretrain_public.txt` | WikiText-2 raw train split, downloaded through direct Hugging Face resolve URL |
| SFT | `tatsu-lab/alpaca` | `data/stage8_public/sft/sft_train.jsonl`, `sft_val.jsonl` | Mapped to `{instruction, input, output, category}` |
| DPO | `tatsu-lab/alpaca_farm`, `alpaca_gpt4_preference.json` | `data/stage8_public/dpo/dpo_train.jsonl`, `dpo_val.jsonl` | Mapped from `output_1`, `output_2`, and `preference` to chosen/rejected pairs |
| GRPO | Local generated reward prompts | `data/stage8_public/grpo/*.jsonl` | Local verifiable reward diagnostics, not public RLVR |

Stage 8 metadata:

- Pretrain text: 16,929 lines, 10,717,316 bytes.
- Tokenizer vocab size: 12,000.
- Tokenization: 2,495,217 tokens, 9,552 train blocks, 194 validation blocks, block size 256.
- SFT: 20,000 train and 2,000 validation examples.
- DPO: 10,000 train and 1,000 validation preference pairs.

## Fallback History

Earlier Stage 8 attempts hit Proxy/TLS errors when using high-level Hugging Face access paths. The successful redownload used direct `urllib` Hugging Face resolve URLs. The final Stage 8 public-data reports mark pretrain, SFT, and DPO as `fallback=false`.

## GRPO Caveat

GRPO data remains local by design because this project uses rule-verifiable rewards for a controlled diagnostic loop. It should be described as "public-policy GRPO diagnostics" only when the policy checkpoint comes from public-data SFT. It should not be described as public RLVR.

## Do Not Commit Raw Data

The raw and processed data files are reproducible artifacts, not source code. They are intentionally covered by `.gitignore`. Keep metadata and conversion code in the repository; keep dataset payloads out of Git.
