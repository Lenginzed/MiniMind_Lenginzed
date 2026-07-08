# Interview Notes

## 60-Second Pitch

I built a from-scratch mini-LLM stack that covers the core lifecycle of modern language-model training: tokenizer and data pipeline, decoder-only transformer, pretraining, assistant-only SFT, LoRA, DPO, GRPO-style diagnostics, and educational quantization. I first validated it on synthetic data, then reran the same pipeline on public WikiText-2, Alpaca, and AlpacaFarm subsets to make the results more credible. The project is not about claiming a strong model; it is about showing I understand and can implement the training mechanics, metrics, checkpoints, diagnostics, and limitations.

## What To Emphasize

1. The model architecture is built from scratch: GQA, RoPE, SwiGLU, RMSNorm, causal masking, shifted loss, and generation.
2. The training loops are custom PyTorch loops, not Hugging Face Trainer, TRL, or PEFT wrappers.
3. LoRA is self-implemented and compared against full fine-tuning.
4. DPO is implemented from sequence log probabilities with a frozen reference model.
5. GRPO is treated as a diagnostic RL loop, with reward variance and zero-std failure modes surfaced explicitly.
6. Quantization is educational and measured honestly, without claiming real deployment speedups.
7. Stage 8 public data made the curves less perfect but more believable.

## Good Interview Answers

### Why build a mini-LLM from scratch?

Because it exposes the full training stack instead of hiding it behind high-level libraries. I can explain how tokens become blocks, how the transformer computes logits, how the causal loss is shifted, how assistant-only labels mask prompts, how LoRA freezes the base model, how DPO compares chosen and rejected log probabilities, and why GRPO needs reward variance.

### What changed when moving from synthetic data to public data?

The public runs were harder. Pretrain and SFT losses remained much higher, DPO preference accuracy did not saturate, and GRPO exposed reward-diversity problems. That is useful because it makes the result more realistic and prevents overclaiming.

### What did LoRA demonstrate?

LoRA updated only about 0.67% of parameters in the public runs. It gave a much cheaper training path and reused the same evaluation/reporting infrastructure, but full fine-tuning reached lower SFT loss in this setup. I would present it as a parameter-efficiency tradeoff, not as a quality win.

### How do you interpret public DPO?

Full DPO ended around 0.55 preference accuracy, while DPO-LoRA ended around 0.525. That is near chance-to-modest, not solved. The important part is that the frozen-reference DPO pipeline works, logs reward margin and preference accuracy, and behaves more realistically on public preference data than on synthetic data.

### Why did GRPO struggle?

The reward functions are local and sparse/dense toy signals. Full GRPO collapsed to zero group reward standard deviation, which means completions within each group stopped receiving meaningfully different rewards. Since group-relative advantage depends on reward variance, this is a training-signal problem. LoRA-GRPO retained more diversity but still did not improve exact accuracy.

### What would you improve next?

I would add a larger public corpus, better data cleaning, KV cache for generation, improved GRPO reward shaping, hyperparameter sweeps, longer public training, and real packed quantization or production inference integration after the educational path stays correct.

## Resume-Friendly Summary

Built an end-to-end mini-LLM training stack from scratch in PyTorch, including decoder-only transformer architecture, tokenizer/data pipeline, pretraining, assistant-only SFT, self-implemented LoRA, DPO with frozen reference model, GRPO-style online reward diagnostics, and educational quantization. Ran synthetic and public-data long-run experiments on WikiText-2, Alpaca, and AlpacaFarm subsets with reproducible configs, checkpoints, metrics, plots, and audit reports.

## What Not To Say

- Do not say the model is aligned.
- Do not say it has real instruction-following capability.
- Do not say it can reason mathematically.
- Do not say GRPO solved RLVR.
- Do not say fake quantization gives production speedup.

The strongest version of the project is honest: it demonstrates implementation depth and diagnostic thinking.
