# Stage 8 Public Training Report

- Pretrain: steps `1500`, train_loss `9.5015` -> `4.2189`
- SFT full: steps `1500`, train_loss `7.0393` -> `3.9252`
- SFT LoRA: steps `1500`, train_loss `6.5874` -> `5.3978`
- DPO full: steps `1000`, train_loss `0.6926` -> `1.0169`
- DPO LoRA: steps `1000`, train_loss `0.6929` -> `0.7124`
- GRPO full: steps `150`, reward_mean `0.1096` -> `0.2300`
- GRPO LoRA: steps `150`, reward_mean `0.1840` -> `0.2269`

## Summaries

```json
{
  "pretrain": {
    "after_samples": "outputs\\stage8_public\\pretrain_public_long\\samples\\after.txt",
    "before_samples": "outputs\\stage8_public\\pretrain_public_long\\samples\\before.txt",
    "best_checkpoint": "outputs\\stage8_public\\pretrain_public_long\\checkpoints\\best.pt",
    "best_eval_loss": 4.73854169845581,
    "best_eval_ppl": 114.26744372838256,
    "cuda_max_memory_allocated": 1049344000,
    "cuda_max_memory_reserved": 1184890880,
    "cuda_memory_allocated": 600456192,
    "cuda_memory_reserved": 1184890880,
    "device": "cuda",
    "dtype": "bf16",
    "final_step": 1500,
    "first_train_loss": 9.501470565795898,
    "gradient_checkpointing": true,
    "last_checkpoint": "outputs\\stage8_public\\pretrain_public_long\\checkpoints\\last.pt",
    "last_eval_loss": 4.73854169845581,
    "last_train_loss": 4.218937128782272,
    "max_steps": 1500,
    "metrics_path": "outputs\\stage8_public\\pretrain_public_long\\metrics.jsonl",
    "output_dir": "outputs/stage8_public/pretrain_public_long",
    "parameter_count": 45631296,
    "resume_loaded": false,
    "scheduler": "cosine",
    "start_step": 0,
    "tokens_seen": 12288000,
    "warmup_steps": 100
  },
  "sft_full": {
    "base_checkpoint": "outputs/stage8_public/pretrain_public_long/checkpoints/best.pt",
    "best_checkpoint": "outputs\\stage8_public\\sft_public_full\\checkpoints\\best.pt",
    "best_eval_loss": 4.015621447563172,
    "best_eval_ppl": 55.45774877873924,
    "device": "cuda",
    "dtype": "bf16",
    "first_train_loss": 7.039257168769836,
    "last_checkpoint": "outputs\\stage8_public\\sft_public_full\\checkpoints\\last.pt",
    "last_eval_loss": 4.015621447563172,
    "last_train_loss": 3.9251615405082703,
    "lora": {},
    "max_steps": 1500,
    "metrics_path": "outputs\\stage8_public\\sft_public_full\\metrics.jsonl",
    "mode": "full",
    "output_dir": "outputs/stage8_public/sft_public_full",
    "parameter_count": 45631296,
    "sample_path": "outputs\\stage8_public\\sft_public_full\\samples\\after.txt",
    "trainable_params": 45631296,
    "trainable_ratio": 1.0
  },
  "sft_lora": {
    "base_checkpoint": "outputs/stage8_public/pretrain_public_long/checkpoints/best.pt",
    "best_checkpoint": "outputs\\stage8_public\\sft_public_lora\\checkpoints\\best.pt",
    "best_eval_loss": 5.241833615303039,
    "best_eval_ppl": 189.0163681547349,
    "device": "cuda",
    "dtype": "bf16",
    "first_train_loss": 6.587377071380615,
    "last_checkpoint": "outputs\\stage8_public\\sft_public_lora\\checkpoints\\last.pt",
    "last_eval_loss": 5.241833615303039,
    "last_train_loss": 5.397764444351196,
    "lora": {
      "alpha": 32,
      "dropout": 0.05,
      "lora_module_count": 20,
      "lora_params": 307200,
      "r": 16,
      "replaced_modules": [
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
        "layers.2.self_attn.q_proj",
        "layers.2.self_attn.v_proj",
        "layers.3.self_attn.q_proj",
        "layers.3.self_attn.v_proj",
        "layers.4.self_attn.q_proj",
        "layers.4.self_attn.v_proj",
        "layers.5.self_attn.q_proj",
        "layers.5.self_attn.v_proj",
        "layers.6.self_attn.q_proj",
        "layers.6.self_attn.v_proj",
        "layers.7.self_attn.q_proj",
        "layers.7.self_attn.v_proj",
        "layers.8.self_attn.q_proj",
        "layers.8.self_attn.v_proj",
        "layers.9.self_attn.q_proj",
        "layers.9.self_attn.v_proj"
      ],
      "target_modules": [
        "q_proj",
        "v_proj"
      ],
      "total_params": 45938496,
      "trainable_params": 307200,
      "trainable_ratio": 0.006687201949319368
    },
    "max_steps": 1500,
    "metrics_path": "outputs\\stage8_public\\sft_public_lora\\metrics.jsonl",
    "mode": "lora",
    "output_dir": "outputs/stage8_public/sft_public_lora",
    "parameter_count": 45938496,
    "sample_path": "outputs\\stage8_public\\sft_public_lora\\samples\\after.txt",
    "trainable_params": 307200,
    "trainable_ratio": 0.006687201949319368
  },
  "dpo_full": {
    "best_checkpoint": "outputs\\stage8_public\\dpo_public_full\\checkpoints\\best.pt",
    "best_eval_loss": 0.6918969064950943,
    "beta": 0.1,
    "device": "cuda",
    "dtype": "bf16",
    "first_train_loss": 0.6925822794437408,
    "last_checkpoint": "outputs\\stage8_public\\dpo_public_full\\checkpoints\\last.pt",
    "last_eval": {
      "chosen_rewards": -2.4015418589115143,
      "chosen_token_count": 78.65,
      "logits": 1.614944338798523,
      "loss": 0.7726494774222374,
      "policy_chosen_logps": -339.6576324462891,
      "policy_chosen_mean_logps": -4.479648852348328,
      "policy_rejected_logps": -327.5865919113159,
      "policy_rejected_mean_logps": -4.509071660041809,
      "preference_accuracy": 0.55,
      "ref_chosen_logps": -315.6422145843506,
      "ref_rejected_logps": -301.95622658729553,
      "rejected_rewards": -2.5630362302064897,
      "rejected_token_count": 76.575,
      "reward_margin": 0.16149440184235572
    },
    "last_train_loss": 1.0168637335300446,
    "lora": {},
    "max_steps": 1000,
    "metrics_path": "outputs\\stage8_public\\dpo_public_full\\metrics.jsonl",
    "mode": "full",
    "output_dir": "outputs/stage8_public/dpo_public_full",
    "parameter_count": 45631296,
    "policy_checkpoint": "outputs/stage8_public/sft_public_full/checkpoints/best.pt",
    "reference_checkpoint": "outputs/stage8_public/sft_public_full/checkpoints/best.pt",
    "sample_path": "outputs\\stage8_public\\dpo_public_full\\samples\\after.txt",
    "trainable_params": 45631296,
    "trainable_ratio": 1.0
  },
  "dpo_lora": {
    "best_checkpoint": "outputs\\stage8_public\\dpo_public_lora\\checkpoints\\best.pt",
    "best_eval_loss": 0.6717841804027558,
    "beta": 0.1,
    "device": "cuda",
    "dtype": "bf16",
    "first_train_loss": 0.6929437518119812,
    "last_checkpoint": "outputs\\stage8_public\\dpo_public_lora
```