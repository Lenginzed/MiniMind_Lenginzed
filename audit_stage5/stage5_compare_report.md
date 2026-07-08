# Stage 5 GRPO Compare Report

Both Full GRPO and GRPO-LoRA were run. These are smoke runs on synthetic local reward data and tiny models.

| Run | Steps | Total params | Trainable params | Trainable ratio | Initial reward | Final reward | Final zero-std frac | Eval reward | Eval exact acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full GRPO | 60 | 1,564,992 | 1,564,992 | 100.00% | 0.1114 | 0.1800 | 1.0000 | 0.2572 | 0.0800 |
| GRPO-LoRA | 50 | 1,580,352 | 15,360 | 0.9719% | 0.1272 | 0.1397 | 0.0000 | 0.1182 | 0.0000 |

Full GRPO had a higher separate eval reward in this smoke run, partly because a few eval samples hit exact reward. GRPO-LoRA kept a nonzero group reward std through the final step and trained only 15,360 parameters. Neither result should be interpreted as real mathematical reasoning or RL alignment.

Main diagnosis: the reward design provides enough dense signal to exercise GRPO mechanics, but exact accuracy remains low and Full GRPO can lose signal when all completions in a group receive equal reward.
