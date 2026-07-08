# Stage 8 Public SFT Report

## Full SFT

- Steps: `1500`
- Params/trainable: `45631296` / `45631296`
- Train loss: `7.0393` -> `3.9252`
- Eval loss: `4.0156`
- Eval ppl: `55.4577`
- Sample path: `outputs\stage8_public\sft_public_full\samples\after.txt`

## LoRA-SFT

- Steps: `1500`
- Params/trainable: `45938496` / `307200`
- Trainable ratio: `0.006687`
- Train loss: `6.5874` -> `5.3978`
- Eval loss: `5.2418`
- Eval ppl: `189.0164`
- LoRA modules: `20`
- Sample path: `outputs\stage8_public\sft_public_lora\samples\after.txt`

This is public Alpaca-format SFT on a small educational model; sample quality should be treated as diagnostic output, not real instruction-following ability.