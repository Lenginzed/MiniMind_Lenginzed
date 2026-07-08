# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM, generate
from minillm.model import count_parameters


def build_tiny_config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
    )


def main() -> int:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = build_tiny_config()
    model = MiniLLMForCausalLM(config).to(device)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print("device:", device)
    print("tiny config:", config)
    print("parameter count:", count_parameters(model))

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]
    logits = outputs["logits"]
    if loss is None:
        raise RuntimeError("loss was not computed")

    print("logits shape:", tuple(logits.shape))
    print("loss:", float(loss.detach().cpu().item()))
    print("loss finite:", bool(torch.isfinite(loss).detach().cpu().item()))
    loss.backward()
    print("backward: ok")
    finite_grads = []
    for param in model.parameters():
        if param.grad is not None:
            finite_grads.append(bool(torch.isfinite(param.grad).all().detach().cpu().item()))
    print("finite gradient check:", bool(finite_grads and all(finite_grads)))

    prompt = input_ids[:, :4]
    generated = generate(
        model,
        prompt,
        max_new_tokens=4,
        temperature=1.0,
        top_k=16,
        top_p=0.9,
        do_sample=True,
    )
    print("generated shape:", tuple(generated.shape))
    print("generated first row:", generated[0].detach().cpu().tolist())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("cuda memory allocated bytes:", torch.cuda.memory_allocated(device))
        print("cuda memory reserved bytes:", torch.cuda.memory_reserved(device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
