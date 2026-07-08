# Claim Audit

This audit checks high-risk wording before open-source release. The goal is to keep README and docs honest: this repository demonstrates a from-scratch mini-LLM engineering stack, not a capable deployed model.

## Search Terms

The release clean pass searched:

- `instruction-following`
- `instruction following`
- `alignment`
- `aligned`
- `reasoning`
- `capable`
- `solved`
- `RLVR`
- `production`
- `speedup`

## Conclusion

No high-risk claim was found in a positive capability context. The terms appear in caveats, limitations, diagnostic explanations, or explicit "do not claim" sections.

## Notable Safe Uses

| Term | Context | Assessment |
| --- | --- | --- |
| `instruction-following` | README and docs say not to claim real instruction-following behavior | Safe |
| `alignment` / `aligned` | Limitations say synthetic DPO is not evidence of real alignment | Safe |
| `reasoning` | Limitations say GRPO does not demonstrate mathematical reasoning | Safe |
| `capable` | Docs warn not to present the small model as generally capable | Safe |
| `solved` | DPO/GRPO docs say public metrics are not solved | Safe |
| `RLVR` | Docs state GRPO is not public RLVR | Safe |
| `production` / `speedup` | Quantization docs say fake quantization is not production acceleration | Safe |

## Release-Safe Wording

Use:

- "from-scratch mini-LLM stack"
- "public-data subset long run"
- "diagnostic metrics"
- "training dynamics"
- "educational quantization"

Avoid:

- "aligned model"
- "real instruction-following model"
- "reasoning model"
- "GRPO solved RLVR"
- "production speedup"

## Files Reviewed

- `README.md`
- `docs/training_stack.md`
- `docs/experiment_results.md`
- `docs/limitations.md`
- `docs/interview_notes.md`
- Stage 7/8 audit summaries used as source material
