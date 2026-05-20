# Attention Sinks in Diffusion Language Models

- **arxiv**: 2510.15731 (Oct 2025, revised Dec 2025)
- **URL**: https://arxiv.org/abs/2510.15731
- **Authors**: Maximo Eduardo Rulli, Simone Petruzzi, Edoardo Michielon, Fabrizio
  Silvestri, Simone Scardapane, Alessio Devoto

## Summary

First paper to characterize "attention sinks" in DLMs. Studies LLaDA-8B,
Dream-7B, MMaDA-8B and contrasts them with Llama-3.1-8B (ARM). Empirical /
descriptive — **no clustering, no per-head fractions, no intervention method**;
robustness analysis only.

## Definition (Sec 4.1)

- A token `j` is a sink at layer `l`, head `h` if its mean attention column
  `Ā_j^{(l,h)}` exceeds the average over all other tokens by at least `ε`.
- `ε = 3` chosen empirically; "filter[s] out at least the 96% of tokens in
  sequence" (Sec 4.1 / Appendix B Fig 10).

## Quantitative claims per model

### LLaDA-8B (Sec 4.2)

- **Token semantics (Table 2 frequencies)**: top sink tokens are `Ċ`
  (whitespace) at **0.368** and `<|mdm_mask|>` at **0.366**, then punctuation /
  EOS. Quote: "sinks consistently form on punctuation marks (periods, commas),
  whitespace, and end-of-sequence tokens."
- **Crucially**: mask token is the **#2 most common sink** (~37%) — only just
  below whitespace. The paper's prose downplays this; the table shows it.
- **Per-layer (Fig 5)**: "As we progress to deeper layers, the number of sinks
  decreases, converging to **one or two sinks per layer**." Deepest layers
  show "masked and unmasked tokens maintain **separate** attention sinks"
  (Fig 6a).
- **No per-head head-count statistics**, no per-layer index breakdown
  (Fig 5 is heat-map / averaged).

### Dream-7B (Sec 4.2)

- **Top sink tokens (Table 2)**: `<|mask|>` **0.321**, `Ċ` 0.090, period 0.046.
  → mask token dominates by mass; the pattern is "primarily positional rather
  than semantic."
- "Sinks often originate at the rightmost masked token and shift leftward as
  tokens are progressively unmasked." No quantitative fraction of steps /
  heads given — Fig 8(b) shows only steps 32→33 as illustration.
- **No mention of a static position-256 phenomenon.** They emphasize
  shifting; our position-256 observation appears to be **novel / unreported**.

### MMaDA-8B (Sec 4.2)

- **Top sink tokens (Table 2)**: `Ċ` 0.180, `<|mdm_mask|>` 0.166.
- "Most stable sink behaviour…sinks that are generally static and less
  frequent…remain fixed at their initial positions throughout the entire
  generation process." Closest to AR / Llama behavior.

### Llama-3.1-8B baseline (Table 1)

- Sinks fixed at position 0 / BOS — standard streaming-LLM picture.

## Comparison (Sec 4.2/4.3)

**Universal across DLMs**: sinks exist; they form on a mix of semantic
(whitespace/punct) and structural (mask) tokens; removing 1-2 sinks barely
hurts performance.

**Model-specific**:
| Model | Sink character | Dominant token |
|---|---|---|
| LLaDA-8B | semantic + mask; *moving* (multi-step persistence then vanishes) | whitespace `Ċ` 0.368 / `<mask>` 0.366 |
| Dream-7B | positional (rightmost-mask, shifts leftward) | `<mask>` 0.321 |
| MMaDA-8B | static / AR-like | `Ċ` 0.180 / `<mask>` 0.166 |
| Llama-3.1-8B | static at pos 0 | BOS |

## Robustness intervention (Table 1, Sec 4.3)

Mask top-`k` sinks (zero attention column), measure task performance.

**GSM8K** (unmasked → mask 1 → 2 → 3 sinks):
- Dream-7B: 0.82 → 0.79 → 0.78 → 0.75
- LLaDA-8B: 0.76 → 0.75 → 0.73 → 0.55
- MMaDA-8B: 0.54 → 0.53 → 0.54 → 0.37
- Llama-3.1-8B: 0.85 → **0.02** → 0.02 → 0.01

**HumanEval**:
- Dream-7B: 0.60 → 0.64 → 0.61 → 0.57
- LLaDA-8B: 0.37 → 0.37 → 0.39 → 0.35
- MMaDA-8B: 0.16 → 0.16 → 0.18 → 0.09
- Llama-3.1-8B: 0.66 → **0.00** → 0.00 → 0.00

→ Headline: "Masking one sink leads to a degradation in performance smaller
than 1%" for DLMs vs catastrophic collapse for the ARM.

## What this paper does NOT do (gaps we can exploit)

1. **No per-head clustering** — they aggregate over heads in Fig 5/6.
2. **No fraction-of-heads statistics** per pattern.
3. **No cross-task head-identity IoU** (uses GSM8K + HumanEval but never
   compares which heads exhibit a pattern across tasks).
4. **No causal intervention isolating mask-attention** — they only zero the
   sink token's column, not the masked-query → mask-key channel.
5. **No KV-cache / sparse-attention method**. Conclusion explicitly notes
   "worth exploring whether sinks could be exploited for acceleration or
   compression" — open direction.
6. **No quantification of Dream's leftward shift** (no fraction of steps,
   no head count).
7. **No discussion of static high positions** (e.g., our position-256
   observation in Dream-7B).
8. **No mechanistic story** — Logit-Lens / circuit analysis listed as future
   work.

## LLaDA architecture (2502.09992) — relevant cross-ref

- "LLaDA employs a Transformer as the mask predictor…does not use a causal
  mask."
- **No explicit sink / BOS / register tokens added.** EOS tokens are appended
  for length control only.
- Uses **vanilla multi-head attention** (not GQA) "due to LLaDA's
  incompatibility with KV caching." → any sink behavior is emergent, not
  architectural.

## Overlap with our `mask_binder` finding

- **Partial overlap on phenomenology**: Dream-7B sinks shifting onto masked
  positions is the closest published observation. But the paper attributes
  this to a positional pattern and provides **no head-level decomposition**.
- **For LLaDA**: Table 2 shows `<mask>` at frequency 0.366 — nearly tied with
  whitespace — yet the paper's prose ("punctuation, whitespace, EOS") elides
  the mask-token contribution. Our mask_binder cluster is consistent with
  this large-but-prose-ignored mass.
- **Our novel angles**:
  - per-head clustering identifying a discrete `mask_binder` population
  - ~20% head fraction (vs 0% reported quantification anywhere in this paper)
  - 73-79% cross-task IoU of head identities (not measured by them)
  - layer-depth concentration profile (they only say "deeper → fewer sinks")
  - causal R_curr intervention isolating the mask→mask channel
  - position-256 static dominance in Dream-7B (unreported)
