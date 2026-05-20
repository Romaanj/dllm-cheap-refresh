# dllm-cheap-refresh

Research on **cheap-refresh inference acceleration for diffusion language models (dLLMs)**, comparing against published caching baselines (Fast-dLLM, Dynamic-dLLM, Elastic-Cache) on LLaDA-8B-Instruct and Dream-v0-Instruct-7B.

Our method: **E3** = lag-1 + remaining-unmask budget ($U_t$) + window-4, paired with sparse-Q forward and KV cache, composed with $\theta{=}0.9$ confidence-threshold parallel decoding.

## Entry point

**Start here**: [`research/notes/session_state_2026_05_20.md`](research/notes/session_state_2026_05_20.md) — captures current state, all baseline numbers, in-flight job status, code diagnoses, and open decisions.

Then [`research/INDEX.md`](research/INDEX.md) for the full notes index (literature reviews, hypotheses, phase notes, decisions).

## Current contents (Phase 1 — research artifacts only)

```
research/      # 12 notes, 17 lit reviews, 4 hypotheses, 1 decision + INDEX
tables.tex     # LaTeX: main full-benchmark table + Dynamic-DLLM component ablation
CLAUDE.md      # Research workflow rules (used by Claude Code in this repo)
```

## Planned layout (Phase 2 — code scaffolding)

To be set up when the project moves to a new environment:

```
methods/
  ours/          # E3 cheap refresh: lag-1 + U_t + window-4, sparse-Q forward, KV cache
  fast_dllm/     # vendored from huggingface/Fast-dLLM @ <sha>
  dynamic_dllm/  # vendored from ICLR 2026 repo @ <sha>
  elastic_cache/ # vendored from upstream @ <sha>
evals/           # unified lm-eval adapters covering all four methods
results/         # JSON summaries only (raw samples → release artifact)
paper/           # method.tex, supplementary
scripts/         # launch helpers (single-GPU runs, sweeps)
```

See decision D1 in [`research/notes/session_state_2026_05_20.md`](research/notes/session_state_2026_05_20.md) for rationale and vendoring rules.

## Source

This repo replaces work that previously lived inside a fork of `NVlabs/Fast-dLLM`. Vendored baselines should be re-cloned fresh from their respective origins rather than carried over.
