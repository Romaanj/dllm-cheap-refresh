---
title: Phase 12 (Dream-7B) — Cascade-respecting refresh oracle 포팅
date: 2026-05-18
status: complete
related: [[phase12_cascade_oracle]], [[phase14a_e2e_results]], [[H004_cascade_active_set]]
---

# TL;DR

**Dream-7B에도 cascade containment가 LLaDA만큼 (혹은 더) 강하게 성립.**
K=128 (10% L)에서 oracle argmax_shift = 0.963, K=384 (28% L)에서 0.982.
모든 K에서 LLaDA보다 +1~5pp 더 높음. 즉 refresh-only paradigm 자체는
Dream에서도 작동.

따라서 **Dream cheap mode -10pp 정확도 하락은 oracle ceiling 때문이 아님**.
원인 후보는:
- (a) lag-1 estimator가 Dream에서 oracle 신호를 충분히 못 잡음
- (b) sparse forward mechanism (KV cache 누적 stale-state) 이 Dream의
      attention pattern과 안 맞음

다음 단계: Phase 13b (Dream) 로 (a) 검증.

# Setup

- Script: `phase12_cascade_oracle_dream.py`
- Model: Dream-org/Dream-v0-Instruct-7B (28 layers, GQA 32/32, MASK_ID=151666)
- 8 GSM8K test samples, 5-shot, gen=256, block=32, steps=256, no DualCache
- Probe stride 16 → 15 probe step pairs/sample → 120 total
- K grid: {0, 32, 64, 128, 256, 384, 512, 768, 1024, full(=T)} uniform
- All 28 layers' block_in / block_out captured per step
- Hybrid forward at each (step, K):
  - A_l = top-K positions by ||cur_block_out[l] − prev_block_out[l]||₂
  - layer l output: out_l[A_l] (refreshed) ⊕ prev_block_out[l][~A_l] (reused)
- Logit shift `cat([logits[:, :1], logits[:, :-1]], dim=1)` 적용 → 실제
  decoding과 동일 컨벤션. shift / no-shift 둘 다 측정.

# Results

## K-curve (Dream — shift-corrected)

| K | K/T | cos_shift | arg_shift | arg_shift_p10 | top5_shift |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.9865 | 0.9143 | 0.7500 | 0.9924 |
| 32 | 0.024 | 0.9931 | 0.9245 | 0.7500 | 0.9971 |
| 64 | 0.048 | 0.9962 | 0.9445 | 0.8125 | 0.9982 |
| 128 | 0.096 | 0.9987 | **0.9630** | 0.8750 | 0.9997 |
| 256 | 0.192 | 0.9995 | 0.9776 | 0.9062 | 1.0000 |
| 384 | 0.288 | 0.9995 | **0.9820** | 0.9375 | 1.0000 |
| 512 | 0.383 | 0.9995 | 0.9823 | 0.9375 | 1.0000 |
| 768 | 0.575 | 0.9996 | 0.9844 | 0.9375 | 1.0000 |
| 1024 | 0.767 | 0.9995 | 0.9846 | 0.9375 | 1.0000 |
| full | 1.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Cross-model 비교 (argmax, mean across probes)

| K | K/T | LLaDA | Dream (shift) | Δ |
|---:|---:|---:|---:|---:|
| 0 | 0% | 0.865 | 0.914 | +5pp |
| 32 | ~2% | 0.883 | 0.925 | +4pp |
| 64 | ~5% | 0.899 | 0.945 | +5pp |
| 128 | ~10% | 0.929 | 0.963 | +3.4pp |
| 256 | ~19% | 0.945 | 0.978 | +3.3pp |
| 384 | ~28% | 0.966 | 0.982 | +1.6pp |
| 1024 | ~77% | 0.982 | 0.985 | +0.3pp |

**Dream의 cascade containment는 LLaDA보다 약간 더 강함.** K=0 (pure
stale reuse)부터 이미 91% argmax 일치. 또한 **K=384에서 0.982로 saturate**
— 1.5%는 oracle도 못 메꿈 (이건 LLaDA도 동일한 ceiling 패턴).

## Per-layer top-K coverage of ||ΔH_l|| (Dream)

| layer | K=32 | K=128 | K=256 | K=512 | K=1024 |
|---:|---:|---:|---:|---:|---:|
| 0  | 0.331 | 0.430 | 0.522 | 0.671 | 0.902 |
| 7  | 0.323 | 0.486 | 0.579 | 0.698 | 0.896 |
| 14 | 0.278 | 0.538 | 0.661 | 0.775 | 0.925 |
| 21 | 0.239 | 0.521 | 0.668 | 0.793 | 0.934 |
| 27 | 0.220 | 0.479 | 0.637 | 0.766 | 0.926 |

**LLaDA 대비 차이**:
- LLaDA L=0 K=32 coverage 0.345, Dream 0.331 — 비슷
- LLaDA L=31 K=32 coverage 0.176, Dream L=27 K=32 0.220 → **Dream의
  깊은 층은 LLaDA보다 더 집중적**

LLaDA stepped schedule (shallow 0.5K / mid K / deep 0.75K) 의 가정 — "shallow는 집중 / deep는 분산" — 이 Dream에서 약간 깨짐.
Dream 깊은 층 (L≥14) 의 K=128 coverage 0.48–0.54로 LLaDA 깊은 층 (L=16: 0.328) 보다 더 집중. **Dream에서는 stepped 대신 uniform K가 더 적절할 수 있음**.

## Per-layer active count (positions with |ΔH_l[p]| > c × layer mean)

| | shallow L<8 | mid 8≤L<20 | deep L≥20 |
|---|---:|---:|---:|
| count > 1× mean (avg) | 158 (12% L) | 224 (17% L) | 253 (19% L) |
| count > 2× mean (avg) | 44 | 124 | 156 |
| count > 4× mean (avg) | 22 | 56 | 64 |

LLaDA와 유사: **deep로 갈수록 active 위치 증가** (cascade 누적 효과 같음).
Dream에서도 H004의 "deep에 active 더 많음" 관찰 재현.

# H004 verdict (Dream)

- **H004a (K ≤ 0.4L → cos ≥ 0.99)**: **PASS** — K=32만 써도 cos_shift 0.993.
- **H004b (deep ≤ 30% true-active)**: **PASS** — deep max ~19%.
- **H004c (monotone, no cliff)**: **PASS** — smooth concave curve, no cliff.
- **H004d (compute saving)**: K=128 (10% L) → argmax 0.963 strict ≥0.95.
  10× speedup ceiling. K=384 (28% L) → 0.982, 3.5× ceiling.

**Dream의 ceiling이 LLaDA보다 살짝 더 너그러움** — 같은 K 예산으로 더
큰 argmax 회복 가능. 그럼에도 cheap mode end-to-end는 -10pp drop → oracle
ceiling이 아닌 다른 곳에서 비용 누적.

# 다음 진단 후보

1. **Phase 13b (Dream)**: lag-1 estimator vs oracle 갭 측정. LLaDA에서
   K=128일 때 oracle 0.929 → E3 lag-1 0.897 (gap 3.2pp). Dream에서 같은
   gap이면 → mechanism이 문제. Dream gap이 훨씬 크면 → lag-1 신호가 약함.

2. **Cheap mode KV cache trace**: full-baseline → cheap-warmup 시점에 KV
   cache를 init한 후 N steps 동안 ~A_input_changed 위치의 K, V는 frozen.
   이 정렬오차가 Dream에서 더 빨리 누적되는지.

3. **K-schedule 재검토**: Dream의 deep layer는 LLaDA보다 더 집중되어
   있음에도 stepped (0.5K shallow / K mid / 0.75K deep)을 그대로 적용 중.
   uniform으로 바꾸면 차이 있을 수 있음.

# Files

- `phase12_cascade_oracle_dream.py` — measurement script
- `analyze_phase12_dream_oracle.py` — verdict computation
- `results_phase12_cascade_oracle_dream/{gpu0,gpu1}/cascade_oracle.jsonl` — 1200 rows total
- `analysis_phase12_dream_oracle.txt`
