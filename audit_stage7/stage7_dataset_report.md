# Stage 7 Dataset Report

- Corpus source: `local_synthetic`
- Corpus path: `data\stage7\raw\pretrain_corpus.txt`
- Corpus size: `21158154` bytes
- Corpus lines: `186574`
- SFT train/val: `20000` / `2000`
- DPO train/val: `10000` / `1000`
- GRPO train/val: `2000` / `300`

## Distributions

- SFT categories: `{'translation': 3333, 'math': 3334, 'flight_rl': 3333, 'format': 3333, 'concept': 3334, 'code': 3333}`
- DPO categories: `{'math': 1667, 'concept': 1667, 'code': 1666, 'format': 1666, 'flight_rl': 1667, 'translation': 1667}`
- DPO rejected types: `{'bad_format': 1667, 'wrong_answer': 1667, 'hallucinated_term': 1666, 'off_topic': 1666, 'unsafe_or_unphysical': 1667, 'vague': 1667}`
- GRPO categories: `{'math_mul_small': 333, 'math_add': 333, 'math_sub': 333, 'keyword': 334, 'multi_step_arithmetic': 333, 'exact_text': 334}`
- GRPO reward types: `{'exact_integer': 1332, 'keyword': 334, 'exact_text': 334}`
- GRPO difficulty: `{'medium': 999, 'easy': 668, 'hard': 333}`

Stage 7 dataset is for longer local training demonstrations. It is not a real benchmark dataset.