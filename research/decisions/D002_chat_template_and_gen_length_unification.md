---
name: D002-chat-template-and-gen-length-unification
description: Unify HumanEval eval protocol across all baselines — chat template ON (Fast-dLLM/Elastic-Cache convention) and gen_length=512 (paper standard). Recorded 2026-05-20.
metadata:
  type: decision
---

# D002 — HumanEval protocol: chat ON + gen_length=512 for all baselines

**Date**: 2026-05-20
**Status**: decided, re-runs pending
**Supersedes**: prior commit `dfacebda` (Junwon, 2026-03-13) that added `use_chat_template = is_instruct and not is_humaneval` (skip chat template for HumanEval)

## Context

While running Elastic-Cache baseline (Section 5/6 of [[session-state-2026-05-20]]), found pass@1 = 0.470 — *higher* than Ours (0.427). On investigation, two protocol inconsistencies surfaced:

1. **Chat template**: our `eval_llada.py` and `eval_dream.py` skipped the chat template for HumanEval (`use_chat_template = is_instruct and not is_humaneval`), but Elastic-Cache's `eval_llada.py` applies it uniformly. `git blame` showed the "skip for HumanEval" branch was a 2026-03-13 modification by Junwon (not a Fast-dLLM upstream convention). Original Fast-dLLM applies chat template to all instruct-model tasks.
2. **Gen length**: existing tables.tex HumanEval rows mixed `gen_length=256` (Vanilla, Fast-dLLM, Ours) and `gen_length=512` (Dynamic-dLLM, Elastic-Cache). Speedup numbers across rows were not apples-to-apples.

## Decision

Adopt the **published baseline convention** uniformly:

- **Chat template ON** for HumanEval on instruct models (matches Fast-dLLM upstream + Elastic-Cache + most recent caching papers).
- **gen_length = 512, steps = 512, block_length = 32** for HumanEval (matches LLaDA paper, Fast-dLLM paper, Dynamic-DLLM Table 4 standard).

GSM8K protocol stays as currently used (`gen_length=256`, chat ON — was already consistent across baselines).

## Code changes (applied 2026-05-20)

`eval_llada.py` and `eval_dream.py`: removed the `is_humaneval` / `use_chat_template = ... and not is_humaneval` branch. Chat template now applied to all instruct-model tasks.

```python
# OLD (Junwon 2026-03-13):
is_humaneval = (hasattr(req, "doc") and req.doc is not None
                and "task_id" in req.doc
                and str(req.doc["task_id"]).lower().startswith("humaneval"))
use_chat_template = self.is_instruct and not is_humaneval

# NEW (2026-05-20, restored to Fast-dLLM upstream convention):
if self.is_instruct:
    m = [{"role": "user", "content": question}]
    user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
```

## Re-runs required

All LLaDA-8B-Instruct HumanEval baselines + Dream-v0-Instruct-7B HumanEval baselines must be re-run with the new protocol. Elastic-Cache already used this protocol (skip).

| Method | Prior wall (chat OFF, gen=256 or 512) | Expected new wall (chat ON, gen=512) |
|--------|---------------------------------------|--------------------------------------|
| LLaDA Vanilla | 2211s (gen=256) | ~4000s (gen=512) |
| LLaDA Fast-dLLM | 647s (gen=256) | ~1500s (gen=512) |
| LLaDA Dynamic-dLLM full (DCU+APD, B=64) | 1220s (gen=512) | ~1500s |
| LLaDA Dynamic-dLLM DCU only (no APD, B=64) | 3786s (gen=512) | ~4000s |
| LLaDA Ours (cheap E3 + parallel) | 899s (gen=256) | ~1500s |
| Dream Vanilla | 1935s (gen=256) | ~3500s (gen=512) |
| Dream Ours | 735s (gen=256) | ~1500s |
| LLaDA Elastic-Cache | already 1653s (gen=512, chat ON) | — keep 0.470 |

Total compute ~5 hours sequential, ~2.5 hours on 2-GPU parallel.

## Why this matters

- The 0.470 → ? gap on Elastic-Cache vs Ours will resolve once all baselines use the same protocol. If Ours' chat-ON pass@1 is e.g. 0.50, the Pareto picture changes substantively.
- Without this fix, the paper's HumanEval table is unsound — different gen budgets and prompt formats across rows.
- Honest framing: D002 also documents that the prior chat-OFF choice was *ours*, not a Fast-dLLM convention. Important for paper appendix transparency.

## Open question

Why did Junwon originally switch to chat OFF on 2026-03-13? Working hypothesis: chat template's "Sure, here's the code: ..." preamble may have confused the sanitize step in `postprocess_code.py`, dropping pass@1. If true, switching back to chat ON might require improving the sanitize regex.

**Action**: after re-runs, inspect Vanilla samples for conversational preambles. If sanitize fails on a non-trivial fraction, file a follow-up decision.

## Related

- [[session-state-2026-05-20]] (Section 8 pending actions — re-run list moves here)
- [[fast_dllm_2505_22618]] (Fast-dLLM paper)
- [[elastic_cache_2510_14973]] (Elastic-Cache paper)
