# Stage 8 Public Recovery Probe

- Python executable: `D:\anaconda3\envs\YSJAirCombat\python.exe`
- Per-dataset timeout: `90.0` seconds
- Any public dataset success: `False`
- Successful datasets: `[]`

## Probe Results

### Salesforce/wikitext

- config: `wikitext-2-raw-v1`
- split: `train`
- field: `text`
- ok: `False`
- timed_out: `False`
- sample_count: `0`
- nonempty_count: `0`
- fields: `[]`
- error: `ConnectionError("Couldn't reach 'Salesforce/wikitext' on the Hub (ProxyError)")`

### tatsu-lab/alpaca

- config: `None`
- split: `train`
- field: `output`
- ok: `False`
- timed_out: `False`
- sample_count: `0`
- nonempty_count: `0`
- fields: `[]`
- error: `ConnectionError("Couldn't reach 'tatsu-lab/alpaca' on the Hub (ProxyError)")`

## Manual Recovery Notes

- Pretrain text should become `data/stage8_public/raw/pretrain_public.txt`, one cleaned text segment per line or paragraph.
- Alpaca SFT should be converted to JSONL rows with `instruction`, `input`, `output`, `category`.
- Preference data should be converted to JSONL rows with `instruction`, `input`, `chosen`, `rejected`, `category`, and `rejected_type` or `reason`.
- After manually placing public data, rerun tokenizer/tokenization and then start Stage 8 public long-run.
