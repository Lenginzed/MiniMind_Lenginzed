# Stage 2.1 Resume Report

## Commands

```powershell
& '<local_python_executable>' scripts\train_pretrain.py --config configs\pretrain_resume_smoke.yaml --max-steps 20
& '<local_python_executable>' scripts\train_pretrain.py --config configs\pretrain_resume_smoke.yaml --resume outputs/pretrain_resume_smoke/checkpoints/last.pt --max-steps 30
```

## Result

- Resume event recorded: `True`
- Resume before step: `20`
- Resume after/final step: `30`
- Optimizer state restored: checkpoint contains `optimizer_state_dict` and resume training continued without error.
- Scheduler state behavior: when max_steps changes from 20 to 30, scheduler progress is rebuilt for the new target while optimizer moments are preserved.
- LR samples after resume: `[(21, 0.00010799999999999998), (22, 9.599999999999998e-05), (23, 8.4e-05), (30, 0.0)]`
- Tokens seen after resume: `30720`
- Last checkpoint: `outputs\pretrain_resume_smoke\checkpoints\last.pt`
- Best checkpoint: `outputs\pretrain_resume_smoke\checkpoints\best.pt`

No resume crash or checkpoint loading issue was observed.
