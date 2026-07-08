# Stage 8 Fallback Decision Report

- Generated at: `2026-07-08T14:28:13`
- Decision: `pause full Stage 8 public long-run`
- Correct label for current state: `Stage 8 public-data access failed audit` / `Stage 8 fallback-mode data path sanity state`

## File Status

- `audit_stage8/public_data_preflight.md`: exists=`True`, bytes=`1782`
- `audit_stage8/public_data_preflight.json`: exists=`True`, bytes=`1524`
- `audit_stage8/stage8_public_dataset_report.md`: exists=`True`, bytes=`1356`
- `data/stage8_public/dataset_metadata.json`: exists=`True`, bytes=`2616`
- `data/stage8_public/raw/pretrain_public.txt`: exists=`True`, bytes=`21158154`
- `data/stage8_public/tokenizers/public_tokenizer.json`: exists=`True`, bytes=`800441`
- `data/stage8_public/processed_pretrain/train.npy`: exists=`True`, bytes=`17604736`
- `data/stage8_public/processed_pretrain/val.npy`: exists=`True`, bytes=`358528`
- `data/stage8_public/processed_pretrain/metadata.json`: exists=`True`, bytes=`644`
- `configs/stage8_public/pretrain_public_long.yaml`: exists=`True`, bytes=`767`
- `outputs/stage8_public/pretrain_public_long/metrics.jsonl`: exists=`False`, bytes=`None`
- `audit_stage8/public_recovery_probe.md`: exists=`True`, bytes=`1290`
- `audit_stage8/public_recovery_probe.json`: exists=`True`, bytes=`1752`

## Public Data Access Result

- Hugging Face preflight network: `{'elapsed_sec': 5.122, 'error': "URLError(SSLZeroReturnError(6, 'TLS/SSL connection has been closed (EOF) (_ssl.c:1135)'))", 'ok': False}`
- Preflight target status:
  - `roneneldan/TinyStories`: accessible=`False`, error=`datasets missing or Hugging Face network check failed`
  - `Salesforce/wikitext`: accessible=`False`, error=`datasets missing or Hugging Face network check failed`
  - `tatsu-lab/alpaca`: accessible=`False`, error=`datasets missing or Hugging Face network check failed`
  - `tatsu-lab/alpaca_farm`: accessible=`False`, error=`datasets missing or Hugging Face network check failed`
  - `OpenAssistant/oasst1`: accessible=`False`, error=`datasets missing or Hugging Face network check failed`

## Dataset Source Judgment

- Pretrain source: `data\stage7\raw\pretrain_corpus.txt`, fallback=`True`
- Pretrain fallback reason: `[{'config': None, 'error': 'ConnectionError("Couldn\'t reach \'roneneldan/TinyStories\' on the Hub (ProxyError)")', 'source': 'roneneldan/TinyStories'}, {'config': 'wikitext-2-raw-v1', 'error': 'ConnectionError("Couldn\'t reach \'Salesforce/wikitext\' on the Hub (ProxyError)")', 'source': 'Salesforce/wikitext'}]`
- SFT source: `data/stage7/raw/sft_*.jsonl`, fallback=`True`, errors=`['ConnectionError("Couldn\'t reach \'tatsu-lab/alpaca\' on the Hub (ProxyError)")']`
- DPO source: `data/stage7/raw/dpo_*.jsonl`, fallback=`True`, errors=`[]`
- GRPO source: `local_verifiable_reward`, fallback=`False`

## Generated Fallback Data

- Fallback pretrain corpus bytes/lines: `21158154` / `186574`
- SFT train/val: `20000` / `2000`
- DPO train/val: `10000` / `1000`
- GRPO train/val: `3000` / `500`
- Tokenizer vocab size: `12000`
- Tokenized total/train/val tokens: `4490944` / `4401152` / `89600`
- Tokenized train/val blocks: `17192` / `350`

## Recovery Probe

- Any public success: `False`
- `Salesforce/wikitext`: ok=`False`, timed_out=`False`, error=`ConnectionError("Couldn't reach 'Salesforce/wikitext' on the Hub (ProxyError)")`
- `tatsu-lab/alpaca`: ok=`False`, timed_out=`False`, error=`ConnectionError("Couldn't reach 'tatsu-lab/alpaca' on the Hub (ProxyError)")`

## Why Full Stage 8 Long-Run Should Not Start Now

- All public-dependent data sources failed and were replaced by Stage 7 synthetic fallback data.
- A long `pretrain_public_long` run would mostly duplicate Stage 7 synthetic training under a Stage 8 path and could mislead README or interview claims.
- The objective of Stage 8 is public dataset migration/control; that objective is not satisfied while TinyStories/WikiText/Alpaca/AlpacaFarm are inaccessible.
- It is acceptable to run at most a 50-100 step fallback sanity check later to verify independent Stage 8 paths, but it should be labeled fallback sanity only.

## Recommended Next Step

- Pause Stage 8 full long-run until Hugging Face access is fixed or public subsets are manually downloaded.
- Manual pretrain target: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `train`, field `text`, saved as `data/stage8_public/raw/pretrain_public.txt`.
- Manual SFT target: `tatsu-lab/alpaca`, fields `instruction/input/output`, saved as `data/stage8_public/sft/*.jsonl`.
- Manual DPO target: AlpacaFarm preference data with `chosen/rejected` or `output_1/output_2/preference` mapping, saved as `data/stage8_public/dpo/*.jsonl`.
- After public data is available, regenerate tokenizer/tokenization and then start public long-run from a clean Stage 8 output directory.
