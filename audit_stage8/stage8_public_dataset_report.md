# Stage 8 Public Dataset Report

This report supersedes the earlier fallback decision report. Public data was successfully downloaded through urllib/direct HF URLs after requests/huggingface_hub failed with ProxyError.

## pretrain

- success: `True`
- fallback: `False`
- source: `Salesforce/wikitext`
- config/split: `wikitext-2-raw-v1` / `train`
- train/val: `None` / `None`
- samples/lines/bytes: `16929` / `16929` / `10717316`
- method: `urllib_parquet`
- field mapping: `None`

## sft

- success: `True`
- fallback: `False`
- source: `tatsu-lab/alpaca`
- config/split: `None` / `train`
- train/val: `20000` / `2000`
- samples/lines/bytes: `None` / `None` / `None`
- method: `urllib_parquet`
- field mapping: `{'category': 'alpaca', 'input': 'input', 'instruction': 'instruction', 'output': 'output'}`

## dpo

- success: `True`
- fallback: `False`
- source: `tatsu-lab/alpaca_farm`
- config/split: `None` / `None`
- train/val: `10000` / `1000`
- samples/lines/bytes: `None` / `None` / `None`
- method: `urllib_json`
- field mapping: `instruction/input plus chosen/rejected or output_1/output_2/preference`

## grpo

- success: `True`
- fallback: `False`
- source: `local_verifiable_reward`
- config/split: `None` / `None`
- train/val: `None` / `None`
- samples/lines/bytes: `None` / `None` / `None`
- method: `None`
- field mapping: `None`

## TinyStories Probe

- ok: `True`
- bytes read: `1048576`
- path: `data\stage8_public\download_cache\tinystories_probe_valid_head.txt`
- note: TinyStories was probed successfully but WikiText-2 remains the selected pretrain source for this run.

## Tokenization

- vocab_size: `12000`
- total/train/val tokens: `2495217` / `2445312` / `49664`
- train/val blocks: `9552` / `194`
- block_size: `256`