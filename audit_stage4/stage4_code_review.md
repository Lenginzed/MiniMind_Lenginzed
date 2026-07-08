# Stage 4 Code Review

## DPO Data Masking

`minillm/dpo_data.py` builds one prompt and two prompt+response sequences for each example. Prompt labels are `-100`, response labels use token ids, EOS is appended to the response and participates in loss/logprob, and padding labels are `-100`. The dataset records raw/effective/skipped/truncated counts plus category and rejected-type distributions.

## sequence_logps

`sequence_logps` forwards the model once, shifts logits and labels exactly like causal LM loss, gathers token log probabilities only where shifted labels are not `-100`, and returns per-sample response logprob sums, token counts, and mean logprobs. Prompt tokens are excluded by the label mask.

## DPO Loss

`dpo_loss` implements `-logsigmoid(beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)))`. It also logs chosen/rejected rewards, reward margin, preference accuracy, policy/reference logps, and logits.

## Reference Model Freeze

Full DPO and DPO-LoRA both load the reference model from `outputs/sft_full/checkpoints/best.pt`, call `freeze_model`, and keep it in eval mode. Reference logprobs are computed under `torch.no_grad()`.

## LoRA-DPO Freeze / Trainable Params

DPO-LoRA applies self-implemented LoRA only to `q_proj` and `v_proj`, freezes the base model, moves injected adapters to the training device, and optimizes only trainable LoRA parameters. The run trained 15,360 / 1,580,352 params (0.9719%).

## Checkpoints / Adapter Saving

Full DPO saves best/last model checkpoints. DPO-LoRA saves best/last full policy checkpoints and best/last adapter files under `outputs/dpo_lora/adapters/`. Each checkpoint stores model state, optimizer state, scheduler state, step, config, model config, mode, and best eval loss.

## Tests

`D:\anaconda3\envs\YSJAirCombat\python.exe -m pytest -q` passed with 51 tests. New Stage 4 tests cover DPO data masking/collation, sequence logprob masking, DPO loss, reference freeze, and CPU smoke runs for Full DPO and DPO-LoRA.

## TODO

Still not implemented in this project stage: GRPO, quantization, KV cache, real tokenizer/data curation, real preference datasets, larger 40M-50M training, vLLM, flash-attn, DeepSpeed, and bitsandbytes.
