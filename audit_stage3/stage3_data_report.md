# Stage 3 SFT Data Report

## Dataset

- Train path: `data/sft/sft_train.jsonl`
- Val path: `data/sft/sft_val.jsonl`
- Metadata path: `data/sft/sft_metadata.json`
- Train rows: `3000`
- Val rows: `300`
- Max length: `128`
- Source: local synthetic templates/rules only; no network download.
- Purpose: SFT/LoRA pipeline validation and method reproduction, not real instruction-tuning data quality.

## Category Distribution

Train categories:

```json
{
  "code": 489,
  "concept": 504,
  "flight_rl": 498,
  "format": 503,
  "math": 497,
  "translation": 509
}
```

Val categories:

```json
{
  "code": 61,
  "concept": 46,
  "flight_rl": 52,
  "format": 47,
  "math": 53,
  "translation": 41
}
```

## Encoded SFT Stats

- Effective train samples: `3000`
- Effective val samples: `300`
- Skipped train samples: `0`
- Skipped val samples: `0`
- Truncated train samples: `0`
- Truncated val samples: `0`
- Avg assistant label tokens/train: `33.83`
- Avg assistant label tokens/val: `34.05`

Assistant-only labels are used: prompt/user tokens are masked with `-100`; assistant output tokens and EOS participate in loss.
