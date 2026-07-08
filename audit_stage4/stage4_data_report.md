# Stage 4 DPO Data Report

## Dataset

- Train path: `data/dpo/dpo_train.jsonl`
- Val path: `data/dpo/dpo_val.jsonl`
- Train rows: 3000
- Val rows: 300
- Max length: 128
- Data note: Synthetic local preference data only for DPO pipeline validation; it is not real human preference data.

## Category Distribution

Train: `{"code": 500, "concept": 500, "flight_rl": 500, "format": 500, "math": 500, "translation": 500}`

Val: `{"code": 50, "concept": 50, "flight_rl": 50, "format": 50, "math": 50, "translation": 50}`

## Rejected Type Distribution

Train: `{"bad_format": 504, "hallucinated_term": 498, "off_topic": 498, "unsafe_or_unphysical": 498, "vague": 498, "wrong_answer": 504}`

Val: `{"bad_format": 54, "hallucinated_term": 48, "off_topic": 48, "unsafe_or_unphysical": 48, "vague": 48, "wrong_answer": 54}`

## Encoded Dataset Stats

- Train effective/skipped/truncated: 3000 / 0 / 0
- Val effective/skipped/truncated: 300 / 0 / 0
- Train avg chosen response label tokens: 38.6830
- Train avg rejected response label tokens: 42.8210
- Val avg chosen response label tokens: 38.4200
- Val avg rejected response label tokens: 42.8767

This is local synthetic preference data for DPO pipeline validation only. It should not be treated as real human preference data.
