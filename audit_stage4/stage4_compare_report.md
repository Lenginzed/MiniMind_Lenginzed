# Stage 4 DPO Comparison Report

## Params

| Run | Total params | Trainable params | Trainable ratio |
| --- | ---: | ---: | ---: |
| Full DPO | 1,564,992 | 1,564,992 | 100.00% |
| DPO-LoRA | 1,580,352 | 15,360 | 0.9719% |

## Metrics

| Run | Initial loss | Final loss | Initial margin | Final margin | Initial acc | Final acc | Best eval loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full DPO | 0.692218 | 0.000044 | 0.001869 | 12.134012 | 0.5000 | 1.0000 | 0.000486 |
| DPO-LoRA | 0.693117 | 0.005183 | 0.000064 | 6.874298 | 0.3750 | 1.0000 | 0.017536 |

## Sample Output Summary

Full DPO completions:

- COMPLETION: A scheduler检 learning rate��.
- COMPLETION: . , is to,.
- COMPLETION: .m.Sbatch,, scheduler a,,,, is speed,,.
- COMPLETION:  step,The records into andh is speed,=.
- COMPLETION:  and�. ing.  ph next-. ming.etaad andc a� berg.

DPO-LoRA completions:

- COMPLETION:  + 20 = 8.
- COMPLETION: A tokenizeres.
- COMPLETION: .检 learning. 模型、策略更新、策略更新、、配置文件.
- COMPLETION: Ain checkpointes andou2.
- COMPLETION: . t ptrara.v the cross. . 模型. -anin.  m and and.�.utils.n. .C model and tokens se. .geT.ies.on. f.�.et tokenizer.  and a a observes, before decay

## Interpretation

Both runs optimize the synthetic preference objective and quickly reach high preference accuracy on this local generated dataset. Full DPO changes all model weights and drives the training/eval loss lower. DPO-LoRA trains only 15,360 adapter parameters, so it is much cheaper and still validates the LoRA-DPO pipeline.

Current limitations: synthetic preference data, tiny model, short training, no real human preference signal, no claim of real RLHF/DPO alignment or useful instruction-following capability.
