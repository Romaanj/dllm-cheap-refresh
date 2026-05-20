# H003 — What is mask_binder computing?

Status: **partial_supported** as of 2026-05-17 — Q5, Q2, B, C, A run. Core finding: mask_binder is task-invariant local-proximity mask aggregation in deep layers.

## Updated findings (2026-05-17)

**Q5 (layer distribution)**: mask_binder fraction by layer is r=0.91-0.97 correlated across GSM 0-shot, GSM 5-shot, HumanEval. Deep (L≥14) has 2.5-3.3x more than shallow. → **task/shot invariant layer pattern**.

**B (dip layers)**: L7, L10, L13, L16, L22, L27 are punct_sink-dominated (80-90% of heads). Jump layers L11, L12, L14, L15, L23, L28 are mask_binder-rich. → **layer-level alternation between punct-routing and mask-aggregation modes**, task-invariant.

**Q2/C (where attention goes, time-varying)**: top1 lands in current block ~70-75% across settings. mask_mass drops from 0.74 (block start) to 0.34 (block end). → mask_binder is **step-dependent role**: peaks early in block, weakens as masks reveal.

**A (self vs neighbor)**: within current block, top1 offset distribution is concentrated at ±1, ±2, ±3 (66% within |offset|≤4 excluding self; only 13.6% is self). → mask_binder is **proximity-based local mask aggregation**, NOT block-aware. Block boundary visibility in Q2 is an artifact of current-block masks being the nearest contiguous mask span to query.

## Synthesized mechanism

LLaDA's MDM training emerges a population of ~20% deep-layer heads that aggregate information across **neighboring mask positions** (effective window ±4-5). These heads do diffuse local-mask consensus, not long-range planning or single-position lookup. The "block-aware" appearance comes purely from the geometric coincidence: current-block masks are the closest contiguous mask span.

## Updated 2026-05-17 (Round 2: E, F, G, L, M, N)

**E (effective window)**: ±4 captures 80%, ±5 captures 85%, ±7 captures 90% of mask_binder's pooled in-current-block attention.

**F (mass breakdown)**: 60% mask + 20% punct + 15% content + 5% special. Positional split: 55% left, 36% right, 9% self.

**G (cross-block continuity)**: At end-of-block steps (28-31), 7-14% of mask_binder top1 lands in NEXT block, while only ~1% lands in further future blocks. → block boundary is irrelevant to mask_binder; what matters is proximity (next-block adjacent masks are within ±5 window).

**L (left bias is architectural)**: Within symmetric ±W windows, L/R ratio is consistently 1.20-1.24 across W=1..15. Geometric null is 1.00. → **inductive left bias confirmed**, ~20% directional preference.

**M (layer-specific window)**: Most deep layers have W80=4-5. Tight layers: L8, L12, L14, L16, L19, L30 (W80≤3). Wide layers: L22, L23, L29 (W80≥6). DIP-layer mask_binders (when present) have W80=4 — same scale as JUMP layers.

**N (cluster uniqueness)**: mask_binder is the ONLY cluster with tight local window (W80=5). punct_sink W80=24, bidir_asym_L W80=18, others W80=20+. → mask_binder's locality is a **unique architectural signature**, not shared with any other head cluster.

## Fingerprint (paper-relevant quantitative summary)

dLLM-intrinsic property of LLaDA-8B:
- Population: ~20% (218±26 heads)
- Layer locus: deep (L11+), alternating with punct-rich layers
- Window: W80 = 5 positions around query  
- Direction: L/R = 1.22 (left-biased)
- Block boundary: irrelevant (cross-block 13%)
- Targets: mask (60%) + punct (20%) + content (15%) + special (5%)

None of these 6 dimensions are reported in MaskKV (2510.09309), Attention Sinks in DLMs (2510.15731), Elastic-Cache (2510.14973), or DPad (2508.14148).

Status: **strongly supported** as a quantitative characterization of mask_binder mechanism. Open: causal validation (knockout) + cross-model verification (Dream-7B, MMaDA).

## Observations so far (raw)

1. ~20% of LLaDA-8B heads (≈200/1024) put >40% of their attention mass on
   mask key positions when queried from currently-masked positions. We
   call this cluster `mask_binder`.
2. Cross-task / cross-shot stability: GSM8K 0-shot, GSM8K 5-shot,
   HumanEval 0-shot all show 18–23% mask_binder fraction, with cell-by-cell
   IoU 73–79% (same physical heads).
3. Layer distribution (qualitative inspection of probe output): mask_binder
   density rises with layer index — concentrated in L12+ (mid to deep
   layers), almost absent in L0-L7.
4. Adding current-block mask positions to the cache floor (R_curr) recovers
   +2.4pp on GSM8K full 1319 — consistent with these heads needing mask
   keys to be cached.
5. Recency window stacked on R_curr adds nothing on full 1319 (was N=50
   noise).

## What the observations do NOT yet tell us

- **Computational role**: do these heads carry information, or are they
  near-no-ops (token-pass-through with degenerate output)?
- **Information content**: what does the head output encode? Position
  structure? Counts? Inter-mask aggregation? Random?
- **Cross-step behavior**: does the head's output change as block-internal
  steps progress and some mask positions get unmasked?
- **Mask key preference inside the cluster**: when mask_binder attends to
  mask positions, is the attention uniform over all mask positions, or
  biased toward nearby mask (within current block), or distant mask (future
  blocks)?
- **Causal contribution**: how much does generation quality degrade if we
  zero out mask_binder head outputs at inference?

## Candidate sub-questions (drill choices)

Each can be turned into a small probe. None proposes a method — all are
**diagnostic observations**.

### Q1. Causal — knockout test
*Does mask_binder actually compute something useful?*
- Zero out output of mask_binder heads only (e.g., scale o_proj contribution
  to 0) and measure GSM8K accuracy.
- Compare against: zero out same number of random heads, zero out
  punct_sink heads, zero out bidir_asym_L heads.
- **Pre-registered outcome**:
  - If knockout of mask_binder drops accuracy ≥5pp more than random
    knockout → mask_binder carries real information.
  - If drop ≤ random → mask_binder is functionally near-no-op; our R_curr
    intervention may be helping for the wrong mechanistic reason.

### Q2. Information content — mask key preference within mask_binder
*Are mask_binder heads attending UNIFORMLY across mask, or with structure?*
- For each mask_binder head, compute attention distribution over mask
  positions only (renormalized): is it flat, peaked at current block,
  peaked at far blocks, or peaked at recently-unmasked-edge?
- **Pre-registered outcome**:
  - Flat → mask_binder is just "averaging mask positions" (structural
    summary).
  - Peaked at current block → it's local mask aggregation.
  - Peaked at far blocks → it's planning summary.

### Q3. Cross-step dynamics — does output change with decoding?
*Is mask_binder content-dependent or purely positional?*
- Take a single (layer, head) mask_binder. Across 32 block-internal steps,
  measure ||head_output|| and cosine similarity between adjacent steps.
- **Pre-registered outcome**:
  - Cosine ~1 across steps → output is structural/positional, content-free.
  - Cosine << 1 with drift → output is tracking content as mask positions
    flip to content.

### Q4. Value vector content
*What does mask token's V_h actually encode? Same vector everywhere?*
- For a given layer, are V vectors at all mask positions identical (modulo
  rotary on Q-K side, not V)? If V_h is identical per mask position →
  mask_binder output is purely an aggregation of identical vectors → can
  only encode count/position info via attention WEIGHTS, not via Values.
- **Pre-registered outcome**:
  - V_mask identical across positions → mask_binder is using attention
    weight distribution to encode something. Output structure depends on
    distribution.
  - V_mask differs → there's per-position content in V (perhaps via
    upstream layers' mixing).

### Q5. Layer-position structure
*Where are mask_binders, and is there a pattern?*
- Plot mask_binder count by layer. Is it monotonic, bimodal, or alternating?
- **Pre-registered outcome**:
  - Concentrated in deep layers → "high-level planning" interpretation.
  - Scattered uniformly → distributed function.
  - Bimodal → two roles (early structure + late planning).

## Next decision

Which sub-question to drill first depends on what we care about.

- **Q1 (knockout)** is the most fundamental — settles whether mask_binder is
  actually doing work. ~1 hour of code + 30min eval.
- **Q2 (mask-key preference)** uses existing probe data, just re-analysis.
  ~30min, no GPU.
- **Q3 (step dynamics)** needs a per-step probe modification. ~1 hour.
- **Q4 (V vectors)** uses existing model + one forward pass. ~30min.
- **Q5 (layer map)** uses existing probe data, just plotting. ~10min.

Order suggestion (cheap-first): Q5 → Q2 → Q4 → Q3 → Q1.
