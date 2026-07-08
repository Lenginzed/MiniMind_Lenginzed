# Stage 8 Public Data Preflight

- Python executable: `<local_python_executable>`
- datasets importable: `True`
- huggingface_hub importable: `True`
- Hugging Face network: `{'ok': False, 'error': "URLError(SSLZeroReturnError(6, 'TLS/SSL connection has been closed (EOF) (_ssl.c:1135)'))", 'elapsed_sec': 5.122}`
- HF datasets cache: `<hf_datasets_cache>`
- Disk free: `713.51` GB / `875.28` GB

## Target Dataset Access

### roneneldan/TinyStories

- config: `None`
- accessible: `False`
- license: `None`
- splits: `None`
- configs: `None`
- error: `datasets missing or Hugging Face network check failed`

### Salesforce/wikitext

- config: `wikitext-2-raw-v1`
- accessible: `False`
- license: `None`
- splits: `None`
- configs: `None`
- error: `datasets missing or Hugging Face network check failed`

### tatsu-lab/alpaca

- config: `None`
- accessible: `False`
- license: `None`
- splits: `None`
- configs: `None`
- error: `datasets missing or Hugging Face network check failed`

### tatsu-lab/alpaca_farm

- config: `None`
- accessible: `False`
- license: `None`
- splits: `None`
- configs: `None`
- error: `datasets missing or Hugging Face network check failed`

### OpenAssistant/oasst1

- config: `None`
- accessible: `False`
- license: `None`
- splits: `None`
- configs: `None`
- error: `datasets missing or Hugging Face network check failed`

## Recommendation

- If TinyStories or WikiText access succeeds, use it for public pretraining.
- If Alpaca access succeeds, use it for public SFT.
- If Alpaca Farm preference conversion fails, fallback to Stage 7 synthetic DPO and mark fallback explicitly.
- Do not download large models or unbounded dataset splits in Stage 8.
