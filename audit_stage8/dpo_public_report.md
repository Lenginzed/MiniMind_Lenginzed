# Stage 8 Public DPO Report

## Full DPO

- Steps: `1000`
- Params/trainable: `45631296` / `45631296`
- Train loss: `0.6926` -> `1.0169`
- Best eval loss: `0.6919`
- Final eval margin: `0.1615`
- Final eval preference accuracy: `0.5500`
- Sample path: `outputs\stage8_public\dpo_public_full\samples\after.txt`

## DPO-LoRA

- Steps: `1000`
- Params/trainable: `45938496` / `307200`
- Trainable ratio: `0.006687`
- Train loss: `0.6929` -> `0.7124`
- Best eval loss: `0.6718`
- Final eval margin: `0.0582`
- Final eval preference accuracy: `0.5250`
- Sample path: `outputs\stage8_public\dpo_public_lora\samples\after.txt`

Public AlpacaFarm preference data did not saturate like synthetic DPO; metrics should be described as noisy and more realistic, not solved.