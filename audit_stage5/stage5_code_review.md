# Stage 5 Code Review

## Online Generation

`minillm/grpo_trainer.py` samples `num_generations` completions per prompt with the existing local `generate` function. Generation is done per prompt rather than padded batches, which avoids attention-mask issues in the current minimal model. Rollout samples are saved to `outputs/grpo_*/samples/rollout_samples.jsonl`.

## Reward Functions

`minillm/grpo_rewards.py` implements integer extraction, text normalization, keyword matching, length checks, format reward, exact integer reward, keyword reward, exact text reward, and a mild length penalty. `combined_reward` includes small dense signals so weak-model completions do not all receive zero reward.

## Group Advantage

`compute_group_advantages` reshapes rewards into prompt groups, computes group mean/std, normalizes rewards within group, and sets advantage to zero when group std is too small. It records `frac_reward_zero_std`; this surfaced an important Full GRPO final-step signal collapse.

## Token-Level GRPO Loss

`build_prompt_completion_batch` masks prompt tokens with `-100` and labels only completion tokens. `token_logps_for_labels` returns token-level logprobs under the response mask. `grpo_loss` uses detached old token logps, token-level ratios, sequence-level advantages broadcast to response tokens, PPO-style clipping, `approx_kl`, `clip_ratio`, and `mean_ratio`.

## Zero Reward / Zero Std Diagnosis

Rewards were not all zero because dense format/length/number rewards fired. However, Full GRPO ended with `frac_reward_zero_std=1.0`, `advantage_std=0.0`, and `grad_norm=0.0`, so the final update had no useful policy-gradient signal. GRPO-LoRA did not show this final zero-std collapse in the run.

## Checkpoints / Metrics

Both runs saved best/last checkpoints, metrics JSONL/CSV, TensorBoard logs, rollout samples, and eval reports. GRPO-LoRA also saved best/last LoRA adapters.

## Tests

`D:\anaconda3\envs\YSJAirCombat\python.exe -m pytest -q` passed with 64 tests. Stage 5 tests cover reward helpers, advantage normalization, clipped loss behavior including zero advantage, and CPU smoke runs for Full GRPO and GRPO-LoRA.

## TODO

Still not implemented: DAPO, GSPO, value model/critic, reference KL penalty, batch generation with attention masks, KV cache, quantization, real RLVR/RLHF data, and larger 40M-50M policy runs.
