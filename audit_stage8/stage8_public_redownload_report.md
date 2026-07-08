# Stage 8 Public Redownload Report

- Generated at: `2026-07-08T14:52:32`
- Training launched this round: `False`
- New dependencies installed: `False`
- Main workaround: `urllib` direct HF API/resolve downloads, because `requests` and `huggingface_hub` hit ProxyError.

## Network Diagnose Summary

- Python executable: `D:\anaconda3\envs\YSJAirCombat\python.exe`
- datasets version: `3.1.0`
- huggingface_hub version: `0.36.2`
- urllib huggingface.co: `True`
- urllib WikiText API: `True`
- urllib Alpaca API: `True`
- requests huggingface.co: `False`; error: `ProxyError(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: / (Caused by ProxyError('Unable to connect to proxy', SSLError(SSLZeroReturnError(6, 'TLS/SSL connection has been closed (EOF) (_ssl.c:1135)'))))"))`
- HF cache: `C:\Users\user\.cache\huggingface\datasets`

## Dataset Results

- TinyStories: probe success=`True`, downloaded probe bytes=`1048576`, not selected as main corpus because WikiText-2 completed first.
- WikiText-2 raw: success=`True`, source_file=`wikitext-2-raw-v1/train-00000-of-00001.parquet`, samples=`16929`, bytes=`10717316`.
- Alpaca SFT: success=`True`, train/val=`20000`/`2000`, mapping=`{'category': 'alpaca', 'input': 'input', 'instruction': 'instruction', 'output': 'output'}`.
- AlpacaFarm DPO: success=`True`, train/val=`10000`/`1000`, source_file=`alpaca_gpt4_preference.json`, mapping=`instruction/input plus chosen/rejected or output_1/output_2/preference`.
- GRPO: source=`local_verifiable_reward`, by design local verifiable reward data.

## Fallback Status

- Public pretrain fallback: `False`
- Public SFT fallback: `False`
- Public DPO fallback: `False`
- Earlier fallback files/reports are superseded by this redownload report; do not use old fallback decision as the current Stage 8 state.

## Tokenizer / Tokenization

- Re-trained tokenizer: `True`
- Re-tokenized pretrain corpus: `True`
- vocab_size: `12000`
- total tokens: `2495217`
- train/val tokens: `2445312` / `49664`
- train/val blocks: `9552` / `194`

## Attempts / Notes

- `datasets` streaming path still emitted ProxyError through requests/huggingface_hub.
- Direct `urllib` parquet/json/txt downloads succeeded for the public data used here.
- DPO was retried separately with known AlpacaFarm JSON direct URLs after the first AlpacaFarm attempt hit TLS EOF.
- No long-run training was started. Next step should be ChatGPT audit of data quality and metadata before training.

## Key Files

- `data/stage8_public/raw/pretrain_public.txt`
- `data/stage8_public/sft/sft_train.jsonl`
- `data/stage8_public/dpo/dpo_train.jsonl`
- `data/stage8_public/tokenizers/public_tokenizer.json`
- `data/stage8_public/processed_pretrain/metadata.json`
- `audit_stage8/network_diagnose.md`
- `audit_stage8/stage8_public_dataset_report.md`
