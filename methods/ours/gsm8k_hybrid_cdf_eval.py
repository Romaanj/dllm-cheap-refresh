"""
GSM8K Hybrid-CDF Equal-Mass Chunking Evaluation
=================================================
Attention Rollout Score의 CDF를 직선(uniform CDF)과 혼합하여
블록 경계를 결정하는 Hybrid-CDF 방식.

핵심 수식:
  hybrid_cdf[i] = λ * attention_cdf[i] + (1-λ) * (i / gen_length)

λ=1.0 → 순수 attention-based equal-mass (기존 방식)
λ=0.0 → 완전 균등 분할 (fixed-block과 동일)
0<λ<1 → attention 분포에 uniform 보정을 주어서
         너무 크거나 작은 블록이 자연스럽게 억제됨.

장점:
  - min_block_size / max_block_size 하이퍼파라미터가 필요 없음
  - 마지막 블록 병합 같은 ad-hoc 로직이 불필요
  - λ 하나로 attention 의존도를 부드럽게 조절
"""

import argparse
import csv
import json
import os
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM

try:
    from cap_partition import context_aware_partition
except ImportError:
    context_aware_partition = None


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions (gsm8k_equal_mass_eval.py와 공유)
# ═══════════════════════════════════════════════════════════════════════════

def extract_answer(text: str) -> str:
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    device = block_mask_index.device
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(torch.long).clone()
    cols = torch.arange(steps, device=device).unsqueeze(0)
    num_transfer_tokens = num_transfer_tokens + (cols < rem.unsqueeze(1)).to(torch.long)
    return num_transfer_tokens


# ═══════════════════════════════════════════════════════════════════════════
# Rollout Score (기존과 동일)
# ═══════════════════════════════════════════════════════════════════════════

def get_depth_adaptive_rollout(
    attentions: Tuple[torch.Tensor, ...],
    invert_depth: bool = False,
) -> torch.Tensor:
    with torch.no_grad():
        if attentions[0].dim() == 4:
            avg_attn = [a.mean(dim=1) for a in attentions]
        else:
            avg_attn = list(attentions)

        L = len(avg_attn)
        res_attn = []

        for i, a in enumerate(avg_attn):
            I = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)
            slope = 0.5
            mid = L / 2
            depth_arg = (mid - i) if invert_depth else (i - mid)
            alpha = 0.5 * torch.sigmoid(
                torch.tensor(slope * depth_arg, device=a.device, dtype=a.dtype)
            )
            res_attn.append((1.0 - alpha) * I + alpha * a)

        rollout = res_attn[0]
        for i in range(1, len(res_attn)):
            rollout = torch.matmul(res_attn[i], rollout)

        return rollout[0].sum(dim=0)


def get_baseline_rollout(
    attentions: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    with torch.no_grad():
        if attentions[0].dim() == 4:
            avg_attn = [a.mean(dim=1) for a in attentions]
        else:
            avg_attn = list(attentions)

        res_attn = [
            0.5 * torch.eye(a.size(-1), device=a.device, dtype=a.dtype) + 0.5 * a
            for a in avg_attn
        ]
        rollout = res_attn[0]
        for i in range(1, len(res_attn)):
            rollout = torch.matmul(res_attn[i], rollout)
        return rollout[0].sum(dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Rollout — forward hook 기반 메모리 효율적 rollout 계산
# ═══════════════════════════════════════════════════════════════════════════

class StreamingRollout:
    """Forward hook으로 attention rollout을 레이어별 즉시 계산.

    기존 방식: 32 레이어 attention matrix를 전부 메모리에 올린 뒤 rollout 계산
    이 방식:   레이어 하나씩 hook에서 rollout에 누적 → 즉시 해제 (메모리 ~32x 절감)
    """

    def __init__(self, num_layers: int, mode: str = "sigmoid", invert_depth: bool = False):
        self.num_layers = num_layers
        self.mode = mode
        self.invert_depth = invert_depth
        self.rollout: Optional[torch.Tensor] = None
        self._layer_idx = 0
        self._hooks: list = []

    def _hook_fn(self, module, input, output):
        x, cache, attn_weights = output
        if attn_weights is None:
            self._layer_idx += 1
            return output

        with torch.no_grad():
            a = attn_weights.mean(dim=1) if attn_weights.dim() == 4 else attn_weights
            I = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)

            if self.mode in ("sigmoid", "sigmoid_inverted"):
                slope = 0.5
                mid = self.num_layers / 2
                do_invert = self.invert_depth or (self.mode == "sigmoid_inverted")
                depth_arg = (mid - self._layer_idx) if do_invert else (self._layer_idx - mid)
                alpha = 0.5 * torch.sigmoid(
                    torch.tensor(slope * depth_arg, device=a.device, dtype=a.dtype)
                )
                res_a = (1.0 - alpha) * I + alpha * a
            else:
                res_a = 0.5 * I + 0.5 * a

            if self.rollout is None:
                self.rollout = res_a
            else:
                self.rollout = torch.matmul(res_a, self.rollout)

        self._layer_idx += 1
        return (x, cache, None)

    def register(self, blocks) -> None:
        for block in blocks:
            h = block.register_forward_hook(self._hook_fn)
            self._hooks.append(h)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_scores(self) -> Optional[torch.Tensor]:
        if self.rollout is None:
            return None
        return self.rollout[0].sum(dim=0)


class StreamingValueNorm:
    """Forward hook으로 attention value projection norm을 레이어별 수집."""

    def __init__(self):
        self.records: List[torch.Tensor] = []
        self._hooks: list = []

    def _sequential_hook_fn(self, module, input, output):
        x = input[0]
        with torch.no_grad():
            x_norm = module.attn_norm(x)
            qkv = module.att_proj(x_norm)
            _, _, v = qkv.split(module.fused_dims, dim=-1)
            norms = torch.linalg.vector_norm(v.float(), ord=2, dim=-1)[0]
            self.records.append(norms.detach().cpu())
        return output

    def _llama_hook_fn(self, module, input, output):
        x = input[0]
        with torch.no_grad():
            x_norm = module.attn_norm(x)
            v = module.v_proj(x_norm)
            norms = torch.linalg.vector_norm(v.float(), ord=2, dim=-1)[0]
            self.records.append(norms.detach().cpu())
        return output

    def register(self, blocks) -> None:
        for block in blocks:
            if hasattr(block, "att_proj") and hasattr(block, "fused_dims"):
                h = block.register_forward_hook(self._sequential_hook_fn)
            elif hasattr(block, "v_proj"):
                h = block.register_forward_hook(self._llama_hook_fn)
            else:
                raise TypeError(f"Unsupported block type for value norm collection: {type(block)}")
            self._hooks.append(h)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_layer_norms(self) -> Optional[torch.Tensor]:
        if not self.records:
            return None
        return torch.stack(self.records, dim=0).to(torch.float64)

    def get_mean_norms(self) -> Optional[torch.Tensor]:
        layer_norms = self.get_layer_norms()
        if layer_norms is None:
            return None
        return layer_norms.mean(dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-CDF Equal-Mass Chunking
# ═══════════════════════════════════════════════════════════════════════════

def hybrid_cdf_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
    lam: float = 0.5,
    inverse: bool = False,
) -> List[Tuple[int, int]]:
    """
    Hybrid-CDF 기반 Equal-Mass Chunking.

    hybrid_cdf[i] = λ * attention_cdf[i] + (1-λ) * (i+1)/gen_length

    λ=1 → 순수 attention CDF,  λ=0 → uniform (fixed-block과 동일)
    min/max block size, 마지막 블록 병합 로직이 필요 없음.

    inverse=True 일 때:
      scores의 역수를 사용하여 attention이 집중된 곳에 큰 블록을 배정.
      inv_scores = 1 / (scores + ε) → 정규화 → CDF
      high attention → low inv_score → CDF가 천천히 상승 → 큰 블록
    """
    gen_length = gen_scores.numel()
    if num_blocks <= 0 or num_blocks > gen_length:
        return [(0, gen_length)]

    scores = gen_scores.detach().cpu().to(torch.float64)
    scores = scores.clamp(min=0)

    total_mass = scores.sum().item()

    # attention score가 거의 0이면 uniform fallback
    if total_mass < 1e-12:
        block_size = gen_length // num_blocks
        return [
            (i * block_size, min((i + 1) * block_size, gen_length))
            for i in range(num_blocks)
        ]

    if inverse:
        inv_scores = 1.0 / (scores + 1e-10)
        attn_cdf = torch.cumsum(inv_scores, dim=0) / inv_scores.sum()
    else:
        attn_cdf = torch.cumsum(scores, dim=0) / total_mass

    # uniform CDF: (i+1) / gen_length → [1/T, 1]
    uniform_cdf = torch.arange(1, gen_length + 1, dtype=torch.float64) / gen_length

    # hybrid CDF
    hybrid_cdf = lam * attn_cdf + (1.0 - lam) * uniform_cdf

    # CDF를 N등분: threshold = k/N (k=1,...,N-1)
    boundaries = [0]
    for k in range(1, num_blocks):
        threshold = k / num_blocks
        candidates = torch.where(hybrid_cdf >= threshold)[0]
        if candidates.numel() > 0:
            boundary = candidates[0].item() + 1
        else:
            boundary = gen_length

        if boundary >= gen_length:
            break

        boundaries.append(boundary)

    if boundaries[-1] != gen_length:
        boundaries.append(gen_length)

    boundaries = sorted(set(boundaries))
    blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    return blocks


def _edge_corrected_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average without zero-padding bias at sequence boundaries."""
    if window <= 1:
        return values.astype(np.float64, copy=True)

    kernel = np.ones(int(window), dtype=np.float64)
    numer = np.convolve(values, kernel, mode="same")
    denom = np.convolve(np.ones_like(values, dtype=np.float64), kernel, mode="same")
    denom = np.clip(denom, 1.0, None)
    return numer / denom


def smoothed_inverse_cdf_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
    window: int = 32,
) -> List[Tuple[int, int]]:
    """
    Smoothed Inverse-CDF partition.
    1. Edge-corrected moving average smoothing (window)
    2. Inverse: 1/(smoothed + ε) → high attention 구간에 큰 블록
    3. CDF equal-mass partition
    """
    gen_length = gen_scores.numel()
    if num_blocks <= 0 or num_blocks > gen_length:
        return [(0, gen_length)]

    scores = gen_scores.detach().cpu().numpy().astype(np.float64)
    scores = np.clip(scores, 0, None)

    if scores.sum() < 1e-12:
        block_size = gen_length // num_blocks
        return [
            (i * block_size, min((i + 1) * block_size, gen_length))
            for i in range(num_blocks)
        ]

    # 1. Smoothing → peak의 영향을 주변으로 퍼뜨리되, 경계에서 0-padding bias를 제거.
    smoothed = _edge_corrected_moving_average(scores, window)

    # 2. Inverse → 높은 곳에 큰 블록
    inv = 1.0 / (smoothed + 1e-8)

    # 3. CDF equal-mass partition
    cumsum = np.cumsum(inv)
    cumsum = cumsum / cumsum[-1]

    boundaries = [0]
    for k in range(1, num_blocks):
        boundary = int(np.searchsorted(cumsum, k / num_blocks))
        if boundary >= gen_length:
            break
        boundaries.append(boundary)

    if boundaries[-1] != gen_length:
        boundaries.append(gen_length)

    boundaries = sorted(set(boundaries))
    blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    return blocks


def composite_smoothed_inverse_cdf_chunking(
    rollout_scores: torch.Tensor,
    value_norms: torch.Tensor,
    num_blocks: int,
    window: int = 32,
    eps: float = 1e-8,
) -> Tuple[List[Tuple[int, int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Composite Smoothed Inverse-CDF partition.

    composite = rollout * (vnorm / mean_vnorm), then smooth, invert, and
    equal-mass partition on the inverse scores.
    """
    gen_length = int(rollout_scores.numel())
    if num_blocks <= 0 or num_blocks > gen_length:
        composite = rollout_scores.detach().cpu().to(torch.float64).clamp(min=0)
        smoothed = composite.clone()
        inv = 1.0 / (smoothed + float(eps))
        return [(0, gen_length)], composite, smoothed, inv

    rollout = rollout_scores.detach().cpu().to(torch.float64).clamp(min=0)
    vnorm = value_norms.detach().cpu().to(torch.float64).clamp(min=0)
    if int(vnorm.numel()) != gen_length:
        raise ValueError(
            f"value_norms length mismatch: got {int(vnorm.numel())}, expected {gen_length}"
        )

    mean_v = float(vnorm.mean().item())
    if mean_v < float(eps):
        vnorm_scaled = torch.ones_like(vnorm)
    elif manual_block_sizes is None:
        vnorm_scaled = vnorm / mean_v
    composite = (rollout * vnorm_scaled).clamp(min=0)

    if float(composite.sum().item()) < float(eps):
        block_size = gen_length // num_blocks
        blocks = [
            (i * block_size, min((i + 1) * block_size, gen_length))
            for i in range(num_blocks)
        ]
        return blocks, composite, composite.clone(), torch.ones_like(composite)

    smoothed_np = _edge_corrected_moving_average(composite.numpy().astype(np.float64), window)
    smoothed_np = np.clip(smoothed_np, 0.0, None)
    inv_np = 1.0 / (smoothed_np + float(eps))

    cumsum = np.cumsum(inv_np)
    cumsum = cumsum / cumsum[-1]

    boundaries = [0]
    for k in range(1, num_blocks):
        boundary = int(np.searchsorted(cumsum, k / num_blocks))
        if boundary >= gen_length:
            break
        boundaries.append(boundary)

    if boundaries[-1] != gen_length:
        boundaries.append(gen_length)

    boundaries = sorted(set(boundaries))
    blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    smoothed = torch.tensor(smoothed_np, dtype=torch.float64)
    inv = torch.tensor(inv_np, dtype=torch.float64)
    return blocks, composite, smoothed, inv



def improved_composite_smoothed_inverse_cdf_chunking(
    rollout_scores: torch.Tensor,
    value_norms: torch.Tensor,
    num_blocks: int,
    window: int = 32,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[List[Tuple[int, int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Improved Composite Smoothed Inverse-CDF partition.

    1. Compute bounded value gain using robust z-score + sigmoid.
    2. Smooth rollout only, then apply position-wise value gain.
    3. Partition by equal mass over inverse composite scores.
    """
    gen_length = int(rollout_scores.numel())
    beta = float(beta)

    if num_blocks <= 0 or num_blocks > gen_length:
        composite = rollout_scores.detach().cpu().to(torch.float64).clamp(min=0)
        return [(0, gen_length)], composite, composite.clone(), 1.0 / (composite + float(eps))

    rollout = rollout_scores.detach().cpu().to(torch.float64).clamp(min=0)
    vnorm = value_norms.detach().cpu().to(torch.float64).clamp(min=0)

    if int(vnorm.numel()) != gen_length:
        raise ValueError(
            f"value_norms length mismatch: got {int(vnorm.numel())}, expected {gen_length}"
        )

    median_v = torch.median(vnorm)
    mad = torch.median(torch.abs(vnorm - median_v)) + float(eps)
    z = (vnorm - median_v) / mad
    value_gain = 1.0 + beta * (2.0 * torch.sigmoid(z) - 1.0)

    rollout_smoothed_np = _edge_corrected_moving_average(
        rollout.numpy().astype(np.float64), window
    )
    rollout_smoothed_np = np.clip(rollout_smoothed_np, 0.0, None)
    rollout_smoothed = torch.tensor(rollout_smoothed_np, dtype=torch.float64)

    composite = (rollout_smoothed * value_gain).clamp(min=0)
    composite_np = composite.numpy().astype(np.float64)

    if float(composite_np.sum()) < float(eps):
        block_size = gen_length // num_blocks
        blocks = [
            (i * block_size, min((i + 1) * block_size, gen_length))
            for i in range(num_blocks)
        ]
        return blocks, composite, rollout_smoothed, torch.ones_like(composite)

    inv_np = 1.0 / (composite_np + float(eps))
    cumsum = np.cumsum(inv_np)
    cumsum = cumsum / cumsum[-1]

    boundaries = [0]
    for k in range(1, num_blocks):
        boundary = int(np.searchsorted(cumsum, k / num_blocks))
        if boundary >= gen_length:
            break
        boundaries.append(boundary)
    if boundaries[-1] != gen_length:
        boundaries.append(gen_length)
    boundaries = sorted(set(boundaries))

    blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    inv = torch.tensor(inv_np, dtype=torch.float64)
    return blocks, composite, rollout_smoothed, inv


def lowest_score_boundary_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
) -> List[Tuple[int, int]]:
    """
    Select boundaries from the lowest rollout-score token positions.

    For num_blocks=N, pick N-1 boundaries and build N contiguous blocks.
    Token index i is converted to boundary i+1, and only internal
    boundaries in (0, gen_length) are allowed.
    """
    gen_length = int(gen_scores.numel())
    if num_blocks <= 0 or num_blocks > gen_length:
        return [(0, gen_length)]
    if num_blocks == 1:
        return [(0, gen_length)]

    scores = gen_scores.detach().cpu().to(torch.float64).clamp(min=0)
    score_list = [float(v) for v in scores.tolist()]

    ranked_token_idx = sorted(range(gen_length), key=lambda i: (score_list[i], i))
    need = num_blocks - 1
    picked: List[int] = []

    for idx in ranked_token_idx:
        boundary = idx + 1
        if 1 <= boundary < gen_length:
            picked.append(boundary)
            if len(picked) == need:
                break

    if len(picked) < need:
        for boundary in range(1, gen_length):
            if boundary not in picked:
                picked.append(boundary)
                if len(picked) == need:
                    break

    boundaries = [0] + sorted(set(picked)) + [gen_length]
    blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    return blocks


def high_score_boundary_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
    boundary_side: str,
    top_k: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Select boundaries from top-k highest rollout-score token positions.

    boundary_side:
      - "before": boundary at token index i (just before token i)
      - "after":  boundary at token index i+1 (just after token i)
    """
    gen_length = int(gen_scores.numel())
    if num_blocks <= 0 or num_blocks > gen_length:
        return [(0, gen_length)]
    if num_blocks == 1:
        return [(0, gen_length)]

    need = num_blocks - 1
    side = str(boundary_side).lower()
    if side not in {"before", "after"}:
        raise ValueError(f"Unknown boundary_side='{boundary_side}'. Use 'before' or 'after'.")

    if top_k is None:
        top_k = need
    top_k = max(int(top_k), need)

    scores = gen_scores.detach().cpu().to(torch.float64)
    score_list = [float(v) for v in scores.tolist()]

    ranked_token_idx = sorted(range(gen_length), key=lambda i: (-score_list[i], i))
    ranked_token_idx = ranked_token_idx[: min(top_k, len(ranked_token_idx))]

    picked: List[int] = []
    for idx in ranked_token_idx:
        boundary = idx if side == "before" else (idx + 1)
        if 1 <= boundary < gen_length and boundary not in picked:
            picked.append(boundary)
            if len(picked) == need:
                break

    if len(picked) < need:
        for idx in sorted(range(gen_length), key=lambda i: (-score_list[i], i)):
            boundary = idx if side == "before" else (idx + 1)
            if 1 <= boundary < gen_length and boundary not in picked:
                picked.append(boundary)
                if len(picked) == need:
                    break

    if len(picked) < need:
        for boundary in range(1, gen_length):
            if boundary not in picked:
                picked.append(boundary)
                if len(picked) == need:
                    break

    boundaries = [0] + sorted(set(picked)) + [gen_length]
    blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    return blocks


def _sizes_to_blocks(sizes: List[int], gen_length: int) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    cur = 0
    for s in sizes:
        nxt = min(cur + int(s), gen_length)
        if nxt > cur:
            blocks.append((cur, nxt))
        cur = nxt
        if cur >= gen_length:
            break
    if not blocks:
        return [(0, gen_length)]
    if blocks[-1][1] < gen_length:
        blocks[-1] = (blocks[-1][0], gen_length)
    return blocks


def _build_block_rollout_stats(
    gen_scores: torch.Tensor, blocks: List[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    scores = gen_scores.detach().cpu().to(torch.float64)
    for block_idx, (start, end) in enumerate(blocks):
        block_scores = scores[start:end]
        if block_scores.numel() == 0:
            mass = 0.0
            mean = 0.0
            max_v = 0.0
            min_v = 0.0
        else:
            mass = float(block_scores.sum().item())
            mean = float(block_scores.mean().item())
            max_v = float(block_scores.max().item())
            min_v = float(block_scores.min().item())
        stats.append(
            {
                "block_index": int(block_idx),
                "start": int(start),
                "end": int(end),
                "size": int(end - start),
                "rollout_mass": mass,
                "rollout_mean": mean,
                "rollout_min": min_v,
                "rollout_max": max_v,
            }
        )
    return stats


def _rollout_summary(gen_scores: torch.Tensor, top_k: int = 8) -> Dict[str, Any]:
    scores = gen_scores.detach().cpu().to(torch.float64)
    if scores.numel() == 0:
        return {
            "num_tokens": 0,
            "sum": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "topk": [],
        }
    k = min(int(top_k), int(scores.numel()))
    top_vals, top_idx = torch.topk(scores, k=k, largest=True)
    topk = [
        {"index": int(i.item()), "score": float(v.item())}
        for i, v in zip(top_idx, top_vals)
    ]
    return {
        "num_tokens": int(scores.numel()),
        "sum": float(scores.sum().item()),
        "mean": float(scores.mean().item()),
        "std": float(scores.std(unbiased=False).item()),
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
        "topk": topk,
    }


def _sample_balanced_sizes(
    gen_length: int,
    num_blocks: int,
    min_size: int,
    max_size: int,
    rng: random.Random,
) -> List[int]:
    if num_blocks <= 0:
        return [gen_length]
    min_size = max(1, int(min_size))
    max_size = max(min_size, int(max_size))
    if min_size * num_blocks > gen_length or max_size * num_blocks < gen_length:
        base = gen_length // num_blocks
        rem = gen_length - base * num_blocks
        sizes = [base] * num_blocks
        for i in range(rem):
            sizes[i] += 1
        return sizes

    sizes: List[int] = []
    remain = gen_length
    for i in range(num_blocks - 1):
        left_blocks = num_blocks - i - 1
        lo = max(min_size, remain - left_blocks * max_size)
        hi = min(max_size, remain - left_blocks * min_size)
        if lo > hi:
            lo = hi = max(1, remain - left_blocks * min_size)
        s = rng.randint(lo, hi)
        sizes.append(s)
        remain -= s
    sizes.append(remain)
    return sizes


def balanced_random_chunking(
    gen_length: int,
    num_blocks: int,
    min_size: int = 28,
    max_size: int = 32,
    scheduler_seed: int = 0,
    sample_index: int = 0,
) -> List[Tuple[int, int]]:
    rng = random.Random(int(scheduler_seed) + int(sample_index))
    sizes = _sample_balanced_sizes(
        gen_length=gen_length,
        num_blocks=num_blocks,
        min_size=min_size,
        max_size=max_size,
        rng=rng,
    )
    return _sizes_to_blocks(sizes, gen_length)


def anchor_partition(
    gen_length: int,
    num_blocks: int,
    anchor_size: int,
    anchor_pos: int,
    min_block_size: int = 26,
    pos_type: str = "center",
    all_right: bool = False,
) -> List[Tuple[int, int]]:
    """
    Anchor-based block partition: 하나의 큰 anchor 블록을 지정 위치에 배치하고,
    나머지 토큰을 (num_blocks - 1)개 블록으로 균등 분배.

    Parameters
    ----------
    gen_length : int
        생성 영역 전체 길이 (예: 256).
    num_blocks : int
        anchor 포함 총 블록 수 (예: 8).
    anchor_size : int
        anchor 블록 크기 (예: 64).
    anchor_pos : int
        anchor 위치 (0-indexed within gen region).
        pos_type="center" → anchor_pos는 블록 중심.
        pos_type="start"  → anchor_pos는 블록 시작.
    min_block_size : int
        non-anchor 블록의 최소 크기 (default 26).
    pos_type : str
        "center" 또는 "start".
    all_right : bool
        True이면 anchor 좌측 잔여를 anchor에 흡수하고,
        모든 non-anchor 블록을 우측에 배치 (structural stress test).

    Returns
    -------
    List[Tuple[int, int]]
        (start, end) 쌍의 정렬된 리스트. 전체 [0, gen_length)를 빈틈없이 커버.

    Raises
    ------
    ValueError
        min_block_size 제약을 만족할 수 없을 때.
    """
    if num_blocks <= 1:
        return [(0, gen_length)]

    # Clamp anchor_size
    anchor_size = min(anchor_size, gen_length)

    # Compute anchor_start based on pos_type
    if pos_type == "center":
        anchor_start = anchor_pos - anchor_size // 2
    else:  # "start"
        anchor_start = anchor_pos
    anchor_start = max(0, min(anchor_start, gen_length - anchor_size))
    anchor_end = anchor_start + anchor_size

    # Residual lengths
    left_len = anchor_start
    right_len = gen_length - anchor_end

    # all_right mode: absorb left residual into anchor, put all blocks right
    if all_right:
        if left_len > 0:
            anchor_start = 0
            anchor_end = anchor_size  # keep original anchor_size, shift left
            if anchor_end > gen_length:
                anchor_end = gen_length
            left_len = 0
            right_len = gen_length - anchor_end

    n_rem = num_blocks - 1

    # Pre-validation: can we fit n_rem blocks with min_block_size?
    total_remaining = left_len + right_len
    if n_rem > 0 and total_remaining < n_rem * min_block_size:
        raise ValueError(
            f"Cannot fit {n_rem} non-anchor blocks with min_block_size={min_block_size}: "
            f"remaining={total_remaining} < {n_rem * min_block_size}. "
            f"anchor_size={anchor_size}, gen_length={gen_length}"
        )

    # Build n_rem uniform-sized blocks across [0, gen_length) minus the anchor region,
    # then insert the anchor block at the correct sorted position.
    # This avoids left/right imbalance issues with proportional splitting.
    if n_rem == 0 or total_remaining == 0:
        return [(anchor_start, anchor_end)]

    if all_right:
        # All non-anchor blocks go after the anchor
        anchor_start = 0
        anchor_end = anchor_size
        if anchor_end > gen_length:
            anchor_end = gen_length
        right_len = gen_length - anchor_end
        base = right_len // n_rem
        remainder = right_len % n_rem
        blocks = [(anchor_start, anchor_end)]
        cur = anchor_end
        for i in range(n_rem):
            size = base + (1 if i < remainder else 0)
            blocks.append((cur, cur + size))
            cur += size
        return blocks

    # General case: create n_rem uniform blocks spanning left+right regions,
    # with sizes computed from total_remaining.
    base = total_remaining // n_rem
    remainder = total_remaining % n_rem
    sizes = [base + (1 if i < remainder else 0) for i in range(n_rem)]

    # Greedily fill blocks: left side first, then anchor, then right side.
    result: List[Tuple[int, int]] = []
    cur = 0
    block_idx = 0

    # Fill left side
    while block_idx < len(sizes) and cur + sizes[block_idx] <= anchor_start:
        result.append((cur, cur + sizes[block_idx]))
        cur += sizes[block_idx]
        block_idx += 1

    # Handle any remaining left-side tokens that don't fit a full block
    if cur < anchor_start:
        # Absorb leftover into anchor by shifting anchor_start
        anchor_start = cur
        anchor_end = anchor_start + anchor_size
        if anchor_end > gen_length:
            anchor_end = gen_length

    # Insert anchor
    result.append((anchor_start, anchor_end))
    cur = anchor_end

    # Recompute sizes for right side from actual remaining space
    n_right = n_rem - block_idx
    actual_right = gen_length - cur
    if n_right > 0 and actual_right > 0:
        r_base = actual_right // n_right
        r_rem = actual_right % n_right
        for i in range(n_right):
            sz = r_base + (1 if i < r_rem else 0)
            result.append((cur, cur + sz))
            cur += sz
    elif actual_right > 0:
        # No blocks left but space remains — extend anchor
        s, _ = result[-1]
        result[-1] = (s, gen_length)

    # Safety: ensure last block reaches gen_length
    if result[-1][1] != gen_length:
        s, _ = result[-1]
        result[-1] = (s, gen_length)

    return result


def inverse_permuted_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
    lam: float = 1.0,
    scheduler_seed: int = 0,
    sample_index: int = 0,
) -> List[Tuple[int, int]]:
    inv_blocks = hybrid_cdf_chunking(
        gen_scores=gen_scores,
        num_blocks=num_blocks,
        lam=lam,
        inverse=True,
    )
    sizes = [e - s for s, e in inv_blocks]
    rng = random.Random(int(scheduler_seed) + int(sample_index))
    rng.shuffle(sizes)
    return _sizes_to_blocks(sizes, gen_scores.numel())


def inverse_head_rescaled_tail_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
    first_block_size: int,
    lam: float = 1.0,
) -> Tuple[List[Tuple[int, int]], List[int], int]:
    inv_blocks = hybrid_cdf_chunking(
        gen_scores=gen_scores,
        num_blocks=num_blocks,
        lam=lam,
        inverse=True,
    )
    base_sizes = [e - s for s, e in inv_blocks]
    if len(base_sizes) <= 1:
        return _sizes_to_blocks(base_sizes, gen_scores.numel()), base_sizes, int(first_block_size)

    gen_length = int(gen_scores.numel())
    k = len(base_sizes)
    max_head = max(1, gen_length - (k - 1))
    clamped_head = max(1, min(int(first_block_size), max_head))
    remain = gen_length - clamped_head

    tail_base = base_sizes[1:]
    tail_sum = sum(tail_base)
    if tail_sum <= 0:
        base = remain // (k - 1)
        rem = remain - base * (k - 1)
        tail_new = [base] * (k - 1)
        for i in range(rem):
            tail_new[i] += 1
    else:
        raw = [remain * (float(s) / float(tail_sum)) for s in tail_base]
        tail_new = [int(x) for x in raw]
        used = sum(tail_new)
        left = remain - used
        if left > 0:
            frac_rank = sorted(
                range(len(raw)),
                key=lambda i: (raw[i] - float(tail_new[i])),
                reverse=True,
            )
            for i in range(left):
                tail_new[frac_rank[i % len(frac_rank)]] += 1
        elif left < 0:
            frac_rank = sorted(
                range(len(raw)),
                key=lambda i: (raw[i] - float(tail_new[i])),
            )
            need = -left
            ptr = 0
            while need > 0 and ptr < len(frac_rank) * 4:
                idx = frac_rank[ptr % len(frac_rank)]
                if tail_new[idx] > 0:
                    tail_new[idx] -= 1
                    need -= 1
                ptr += 1

    sizes = [clamped_head] + tail_new
    blocks = _sizes_to_blocks(sizes, gen_length)
    final_sizes = [e - s for s, e in blocks]
    return blocks, base_sizes, clamped_head


# ═══════════════════════════════════════════════════════════════════════════
# Transfer Index Selection
# ═══════════════════════════════════════════════════════════════════════════

def select_transfer_index_threshold(
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    transfer_index = mask_index & (confidence >= threshold)
    max_conf_idx = torch.argmax(confidence, dim=1, keepdim=True)
    force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_idx, True)
    transfer_index = (transfer_index | force_mask) & mask_index
    return transfer_index


def select_transfer_index_topk(
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
) -> torch.Tensor:
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    _, idx = torch.sort(confidence, dim=1, descending=True)
    B, L = confidence.shape
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded

    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return transfer_index


# ═══════════════════════════════════════════════════════════════════════════
# Generation: Hybrid-CDF + Confidence-based Parallel Decoding
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_hybrid_cdf(
    model,
    tokenizer,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    num_blocks: int,
    steps_per_block: int,
    lam: float = 0.5,
    temperature: float = 0.0,
    threshold: Optional[float] = 0.9,
    rollout_mode: str = "sigmoid",
    inverse: bool = False,
    control_mode: str = "none",
    control_min_size: int = 28,
    control_max_size: int = 32,
    scheduler_seed: int = 0,
    first_block_size: int = -1,
    manual_block_sizes: Optional[List[int]] = None,
    high_score_top_k: Optional[int] = None,
    prompt_len_chars: Optional[int] = None,
    sample_index: int = 0,
    verbose: bool = False,
    strategy: str = "hybrid_cdf_sigmoid",
    cap_alpha: float = 1.0,
    cap_b_min: int = 8,
    cap_max_iter: int = 50,
    # anchor partition params (used when strategy="anchor_score")
    anchor_size: int = 64,
    anchor_min_block_size: int = 26,
    anchor_pos_type: str = "center",
    anchor_all_right: bool = False,
    smoothing_window: int = 32,
    composite_beta: float = 0.5,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """
    Hybrid-CDF Equal-Mass Chunking + Confidence-based Parallel Decoding.

    Step 0: Forward pass → Rollout Score → Hybrid CDF → 블록 경계 결정
    Step 1+: 블록별 Confidence-based decoding (앞→뒤)

    inverse=True: attention 역수 기반 CDF (high attention → big block)
    """
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    nfe = 0
    rollout_summary: Optional[Dict[str, Any]] = None
    value_norm_summary: Optional[Dict[str, Any]] = None
    composite_summary: Optional[Dict[str, Any]] = None
    composite_smoothed_summary: Optional[Dict[str, Any]] = None
    composite_inverse_summary: Optional[Dict[str, Any]] = None
    block_composite_stats: Optional[List[Dict[str, Any]]] = None
    initial_inverse_blocks: Optional[List[Tuple[int, int]]] = None
    size_permutation: Optional[List[int]] = None
    permuted_from_inverse = False
    head_rescaled_from_inverse = False
    head_rescaled_clamped_first: Optional[int] = None
    schedule_source = "rollout_hybrid_cdf"
    high_score_boundary_side: Optional[str] = None
    high_score_top_k_used: Optional[int] = None
    block_rollout_stats: List[Dict[str, Any]] = []
    initial_block_rollout_stats: Optional[List[Dict[str, Any]]] = None
    use_cap_context = str(strategy).startswith("cap_context")
    use_composite_smoothed_inverse_cdf = str(strategy).startswith("composite_smoothed_inverse_cdf")
    use_improved_composite_smoothed_inverse_cdf = str(strategy).startswith("improved_composite_smoothed_inverse_cdf")
    blocks: List[Tuple[int, int]]
    rollout_scores = None
    value_norm_scores = None

    if manual_block_sizes is not None:
        manual_sizes = [int(x) for x in manual_block_sizes]
        if any(v <= 0 for v in manual_sizes):
            raise ValueError("manual_block_sizes must contain only positive integers")
        if sum(manual_sizes) != gen_length:
            raise ValueError(
                f"sum(manual_block_sizes)={sum(manual_sizes)} must equal gen_length={gen_length}"
            )
        blocks = _sizes_to_blocks(manual_sizes, gen_length)
        schedule_source = "manual_blocks"
    else:
        # ── Step 0: Rollout Score 계산 + Hybrid-CDF Chunking ──
        # Manual schedules already provide boundaries, so only adaptive schedules
        # need the expensive output_attentions=True rollout pass.
        torch.cuda.empty_cache()

        invert_depth = rollout_mode == "sigmoid_inverted"
        hook_mode = "baseline" if rollout_mode == "baseline" else "sigmoid"

        core_model = model.model if hasattr(model, "model") else model
        blocks_list = core_model.transformer.blocks
        num_layers = len(blocks_list)

        streaming = StreamingRollout(
            num_layers=num_layers, mode=hook_mode, invert_depth=invert_depth,
        )
        value_streaming: Optional[StreamingValueNorm] = None
        streaming.register(blocks_list)
        if use_composite_smoothed_inverse_cdf or use_improved_composite_smoothed_inverse_cdf:
            value_streaming = StreamingValueNorm()
            value_streaming.register(blocks_list)

        try:
            outputs = model(x, output_attentions=True)
            nfe += 1
        finally:
            streaming.remove()
            if value_streaming is not None:
                value_streaming.remove()

        rollout_scores = streaming.get_scores()
        value_norm_scores = value_streaming.get_mean_norms() if value_streaming is not None else None
        del outputs, streaming, value_streaming
        torch.cuda.empty_cache()

    if rollout_scores is not None:
        gen_scores = rollout_scores.to(torch.float64)[prompt_len: prompt_len + gen_length]
        rollout_summary = _rollout_summary(gen_scores)
        use_smoothed_inverse_cdf = str(strategy).startswith("smoothed_inverse_cdf")
        use_lowest_score_boundary = str(strategy).startswith("lowest_score_boundary")
        use_high_score_before_boundary = str(strategy).startswith("high_score_boundary_before")
        use_high_score_after_boundary = str(strategy).startswith("high_score_boundary_after")
        use_anchor_score = str(strategy) == "anchor_score"
        if use_improved_composite_smoothed_inverse_cdf:
            if value_norm_scores is None:
                raise RuntimeError("improved_composite_smoothed_inverse_cdf requires value norm scores")
            gen_value_norms = value_norm_scores.to(torch.float64)[prompt_len: prompt_len + gen_length]
            value_norm_summary = _rollout_summary(gen_value_norms)
            blocks, composite_scores, composite_smoothed_scores, composite_inverse_scores = (
                improved_composite_smoothed_inverse_cdf_chunking(
                    rollout_scores=gen_scores,
                    value_norms=gen_value_norms,
                    num_blocks=num_blocks,
                    window=smoothing_window,
                    beta=composite_beta,
                )
            )
            composite_summary = _rollout_summary(composite_scores)
            composite_smoothed_summary = _rollout_summary(composite_smoothed_scores)
            composite_inverse_summary = _rollout_summary(composite_inverse_scores)
            block_composite_stats = _build_block_rollout_stats(composite_scores, blocks)
            schedule_source = "improved_composite_smoothed_inverse_cdf"
        elif use_composite_smoothed_inverse_cdf:
            if value_norm_scores is None:
                raise RuntimeError("composite_smoothed_inverse_cdf requires value norm scores")
            gen_value_norms = value_norm_scores.to(torch.float64)[prompt_len: prompt_len + gen_length]
            value_norm_summary = _rollout_summary(gen_value_norms)
            blocks, composite_scores, composite_smoothed_scores, composite_inverse_scores = (
                composite_smoothed_inverse_cdf_chunking(
                    rollout_scores=gen_scores,
                    value_norms=gen_value_norms,
                    num_blocks=num_blocks,
                    window=smoothing_window,
                )
            )
            composite_summary = _rollout_summary(composite_scores)
            composite_smoothed_summary = _rollout_summary(composite_smoothed_scores)
            composite_inverse_summary = _rollout_summary(composite_inverse_scores)
            block_composite_stats = _build_block_rollout_stats(composite_scores, blocks)
            schedule_source = "composite_smoothed_inverse_cdf"
        elif use_smoothed_inverse_cdf:
            blocks = smoothed_inverse_cdf_chunking(
                gen_scores=gen_scores,
                num_blocks=num_blocks,
                window=smoothing_window,
            )
            schedule_source = "smoothed_inverse_cdf"
        elif use_anchor_score:
            # Score-based anchor: place large block at argmax of rollout scores
            anchor_pos = int(torch.argmax(gen_scores).item())
            blocks = anchor_partition(
                gen_length=gen_length,
                num_blocks=num_blocks,
                anchor_size=anchor_size,
                anchor_pos=anchor_pos,
                min_block_size=anchor_min_block_size,
                pos_type=anchor_pos_type,
                all_right=anchor_all_right,
            )
            schedule_source = "anchor_score"
        elif use_cap_context:
            if context_aware_partition is None:
                raise ImportError(
                    "cap_context strategy requires cap_partition.py to be available"
                )
            cap_sizes = context_aware_partition(
                L=gen_length,
                K=num_blocks,
                P=prompt_len,
                alpha=cap_alpha,
                B_min=cap_b_min,
                max_iter=cap_max_iter,
            )
            blocks = _sizes_to_blocks(cap_sizes, gen_scores.numel())
            schedule_source = "cap_context"
        elif manual_block_sizes is not None:
            manual_sizes = [int(x) for x in manual_block_sizes]
            if any(v <= 0 for v in manual_sizes):
                raise ValueError("manual_block_sizes must contain only positive integers")
            if sum(manual_sizes) != gen_length:
                raise ValueError(
                    f"sum(manual_block_sizes)={sum(manual_sizes)} must equal gen_length={gen_length}"
                )
            blocks = _sizes_to_blocks(manual_sizes, gen_scores.numel())
            schedule_source = "manual_blocks"
        elif control_mode == "balanced_random":
            initial_inverse_blocks = hybrid_cdf_chunking(
                gen_scores=gen_scores,
                num_blocks=num_blocks,
                lam=lam,
                inverse=True,
            )
            blocks = balanced_random_chunking(
                gen_length=gen_length,
                num_blocks=num_blocks,
                min_size=control_min_size,
                max_size=control_max_size,
                scheduler_seed=scheduler_seed,
                sample_index=sample_index,
            )
            schedule_source = "balanced_random"
        elif control_mode == "inverse_permuted":
            initial_inverse_blocks = hybrid_cdf_chunking(
                gen_scores=gen_scores,
                num_blocks=num_blocks,
                lam=lam,
                inverse=True,
            )
            inverse_sizes = [e - s for s, e in initial_inverse_blocks]
            perm_order = list(range(len(inverse_sizes)))
            rng = random.Random(int(scheduler_seed) + int(sample_index))
            rng.shuffle(perm_order)
            permuted_sizes = [inverse_sizes[idx] for idx in perm_order]
            blocks = _sizes_to_blocks(permuted_sizes, gen_scores.numel())
            size_permutation = perm_order
            permuted_from_inverse = True
            schedule_source = "inverse_permuted"
        elif control_mode == "inverse_head_rescaled_tail":
            if int(first_block_size) <= 0:
                raise ValueError(
                    "control_mode='inverse_head_rescaled_tail' requires first_block_size > 0"
                )
            blocks, inverse_sizes, clamped_first = inverse_head_rescaled_tail_chunking(
                gen_scores=gen_scores,
                num_blocks=num_blocks,
                first_block_size=int(first_block_size),
                lam=lam,
            )
            initial_inverse_blocks = _sizes_to_blocks(inverse_sizes, gen_scores.numel())
            head_rescaled_from_inverse = True
            head_rescaled_clamped_first = int(clamped_first)
            schedule_source = "inverse_head_rescaled_tail"
        elif use_lowest_score_boundary:
            blocks = lowest_score_boundary_chunking(
                gen_scores=gen_scores,
                num_blocks=num_blocks,
            )
            schedule_source = "lowest_score_boundary"
        elif use_high_score_before_boundary or use_high_score_after_boundary:
            high_score_boundary_side = "before" if use_high_score_before_boundary else "after"
            high_score_top_k_used = (
                int(high_score_top_k) if high_score_top_k is not None else (int(num_blocks) - 1)
            )
            blocks = high_score_boundary_chunking(
                gen_scores=gen_scores,
                num_blocks=num_blocks,
                boundary_side=high_score_boundary_side,
                top_k=high_score_top_k_used,
            )
            schedule_source = f"high_score_boundary_{high_score_boundary_side}"
        else:
            blocks = hybrid_cdf_chunking(
                gen_scores=gen_scores,
                num_blocks=num_blocks,
                lam=lam,
                inverse=inverse,
            )
            schedule_source = "hybrid_cdf_inverse" if inverse else "hybrid_cdf"
            if inverse:
                initial_inverse_blocks = list(blocks)

        if initial_inverse_blocks is not None:
            initial_block_rollout_stats = _build_block_rollout_stats(
                gen_scores=gen_scores,
                blocks=initial_inverse_blocks,
            )
        block_rollout_stats = _build_block_rollout_stats(
            gen_scores=gen_scores,
            blocks=blocks,
        )
    elif manual_block_sizes is None:
        if use_cap_context:
            if context_aware_partition is None:
                raise ImportError(
                    "cap_context strategy requires cap_partition.py to be available"
                )
            cap_sizes = context_aware_partition(
                L=gen_length,
                K=num_blocks,
                P=prompt_len,
                alpha=cap_alpha,
                B_min=cap_b_min,
                max_iter=cap_max_iter,
            )
            blocks = _sizes_to_blocks(cap_sizes, gen_length)
            schedule_source = "cap_context"
        else:
            if verbose:
                print(f"  [WARNING] rollout computation returned None, falling back to uniform blocks")
            block_size = gen_length // num_blocks
            blocks = [
                (i * block_size, min((i + 1) * block_size, gen_length))
                for i in range(num_blocks)
            ]
            schedule_source = "uniform_fallback"

    if verbose:
        block_sizes = [e - s for s, e in blocks]
        print(
            f"  [{schedule_source}/{rollout_mode}/λ={lam}] "
            f"{len(blocks)} blocks, sizes={block_sizes}"
        )

    # ── Step 1+: 블록별 Confidence-based Parallel Decoding ──
    for block_idx, (block_start_rel, block_end_rel) in enumerate(blocks):
        block_start = prompt_len + block_start_rel
        block_end = prompt_len + block_end_rel

        block_mask = (x[:, block_start:block_end] == mask_id)
        if block_mask.sum() == 0:
            continue

        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        step_i = 0

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            nfe += 1
            mask_idx = (x == mask_id)
            mask_idx[:, block_end:] = False

            outputs = model(x)
            logits = outputs.logits

            logits_noisy = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            probs = F.softmax(logits.to(torch.float64), dim=-1)
            score = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_idx, x0, x)
            neg_inf = torch.tensor(
                torch.finfo(score.dtype).min, device=device, dtype=score.dtype
            )
            confidence = torch.where(mask_idx, score, neg_inf)

            if threshold is not None:
                transfer_index = select_transfer_index_threshold(
                    confidence, mask_idx, threshold
                )
            else:
                max_i = num_transfer.size(1) - 1
                si = min(step_i, max_i)
                per_step = num_transfer[:, si]
                transfer_index = select_transfer_index_topk(
                    confidence, mask_idx, per_step
                )

            x[transfer_index] = x0[transfer_index]
            step_i += 1

    info = {
        "rollout_mode": rollout_mode,
        "lam": lam,
        "inverse": inverse,
        "control_mode": control_mode,
        "control_min_size": control_min_size,
        "control_max_size": control_max_size,
        "scheduler_seed": scheduler_seed,
        "first_block_size": int(first_block_size),
        "manual_block_sizes": [int(x) for x in manual_block_sizes] if manual_block_sizes is not None else None,
        "high_score_top_k": (int(high_score_top_k) if high_score_top_k is not None else None),
        "high_score_top_k_used": high_score_top_k_used,
        "high_score_boundary_side": high_score_boundary_side,
        "prompt_len_chars": (int(prompt_len_chars) if prompt_len_chars is not None else None),
        "cap_alpha": float(cap_alpha),
        "cap_b_min": int(cap_b_min),
        "cap_max_iter": int(cap_max_iter),
        "composite_beta": float(composite_beta),
        "strategy": str(strategy),
        "schedule_source": schedule_source,
        "head_rescaled_clamped_first_block_size": head_rescaled_clamped_first,
        "sample_index": sample_index,
        "num_blocks_requested": num_blocks,
        "num_blocks_actual": len(blocks),
        "block_boundaries": blocks,
        "block_sizes": [e - s for s, e in blocks],
        "block_rollout_stats": block_rollout_stats,
        "rollout_summary": rollout_summary,
        "value_norm_summary": value_norm_summary,
        "composite_summary": composite_summary,
        "composite_smoothed_summary": composite_smoothed_summary,
        "composite_inverse_summary": composite_inverse_summary,
        "block_composite_stats": block_composite_stats,
        "initial_inverse_block_boundaries": initial_inverse_blocks,
        "initial_inverse_block_sizes": (
            [e - s for s, e in initial_inverse_blocks]
            if initial_inverse_blocks is not None
            else None
        ),
        "initial_inverse_block_rollout_stats": initial_block_rollout_stats,
        "size_permutation": size_permutation,
        "permuted_from_inverse": permuted_from_inverse,
        "head_rescaled_from_inverse": head_rescaled_from_inverse,
    }
    return x, nfe, info


# ═══════════════════════════════════════════════════════════════════════════
# Generation: Fixed-Block Baseline (비교용, 기존과 동일)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_fixed_block(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    block_length: int,
    steps_per_block: int,
    temperature: float = 0.0,
    threshold: Optional[float] = 0.9,
    gold_prefix_tokens: Optional[torch.Tensor] = None,
    save_top1_trace: bool = False,
    tokenizer: Any = None,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0, (
        f"gen_length({gen_length})는 block_length({block_length})의 배수여야 합니다."
    )
    num_blocks = gen_length // block_length
    nfe = 0
    gold_prefix_tokens_used = 0
    top1_trace: List[Dict[str, Any]] = []
    global_step = 0

    if gold_prefix_tokens is not None:
        if gold_prefix_tokens.dim() != 1:
            gold_prefix_tokens = gold_prefix_tokens.view(-1)
        gold_prefix_tokens = gold_prefix_tokens.to(device=device, dtype=torch.long)
        gold_prefix_tokens_used = min(int(gold_prefix_tokens.numel()), int(gen_length))
        if gold_prefix_tokens_used > 0:
            x[:, prompt_len: prompt_len + gold_prefix_tokens_used] = gold_prefix_tokens[
                :gold_prefix_tokens_used
            ].unsqueeze(0)

    decode_start_block = gold_prefix_tokens_used // block_length

    for num_block in range(decode_start_block, num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        block_mask = (x[:, block_start:block_end] == mask_id)
        if block_mask.sum() == 0:
            continue

        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        step_i = 0

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            nfe += 1
            mask_idx = (x == mask_id)
            mask_idx[:, block_end:] = False

            outputs = model(x)
            logits = outputs.logits

            logits_noisy = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            probs = F.softmax(logits.to(torch.float64), dim=-1)
            score = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_idx, x0, x)
            neg_inf = torch.tensor(
                torch.finfo(score.dtype).min, device=device, dtype=score.dtype
            )
            confidence = torch.where(mask_idx, score, neg_inf)

            if threshold is not None:
                transfer_index = select_transfer_index_threshold(
                    confidence, mask_idx, threshold
                )
            else:
                max_i = num_transfer.size(1) - 1
                si = min(step_i, max_i)
                per_step = num_transfer[:, si]
                transfer_index = select_transfer_index_topk(
                    confidence, mask_idx, per_step
                )

            if save_top1_trace:
                commit_abs = torch.nonzero(transfer_index[0], as_tuple=False).view(-1)
                commit_ids = x0[0, commit_abs].detach().cpu().tolist()
                commit_conf = confidence[0, commit_abs].detach().cpu().tolist()
                commit_abs_list = [int(pos) for pos in commit_abs.detach().cpu().tolist()]
                commit_rel_list = [int(pos - prompt_len) for pos in commit_abs_list]
                row = {
                    "global_step": int(global_step),
                    "block_idx": int(num_block),
                    "sub_step": int(step_i),
                    "block_start_rel": int(num_block * block_length),
                    "block_end_rel": int((num_block + 1) * block_length),
                    "block_start_abs": int(block_start),
                    "block_end_abs": int(block_end),
                    "remaining_before": int(remaining),
                    "top1_token_ids": [int(t) for t in x0[0].detach().cpu().tolist()],
                    "top1_generation_token_ids": [
                        int(t) for t in x0[0, prompt_len:].detach().cpu().tolist()
                    ],
                    "commit_positions_abs": commit_abs_list,
                    "commit_positions_rel": commit_rel_list,
                    "commit_positions_in_block": [
                        int(pos - block_start) for pos in commit_abs_list
                    ],
                    "commit_token_ids": [int(t) for t in commit_ids],
                    "commit_confidences": [float(c) for c in commit_conf],
                }
                if tokenizer is not None:
                    row["commit_tokens"] = tokenizer.convert_ids_to_tokens(
                        [int(t) for t in commit_ids]
                    )
                    row["commit_text"] = tokenizer.decode(
                        [int(t) for t in commit_ids],
                        skip_special_tokens=False,
                    )
                top1_trace.append(row)

            x[transfer_index] = x0[transfer_index]
            if save_top1_trace:
                top1_trace[-1]["remaining_after"] = int(
                    (x[:, block_start:block_end] == mask_id).sum().item()
                )
            step_i += 1
            global_step += 1

    info = {
        "block_length": block_length,
        "num_blocks": num_blocks,
        "gold_prefix_tokens_used": int(gold_prefix_tokens_used),
        "gold_prefix_truncated": bool(
            gold_prefix_tokens is not None
            and int(gold_prefix_tokens.numel()) > int(gen_length)
        ),
        "decode_start_block": int(decode_start_block),
    }
    if save_top1_trace:
        info["top1_trace"] = top1_trace
        info["top1_trace_schema"] = {
            "top1_token_ids": "argmax token id at every absolute sequence position for this sub-step",
            "top1_generation_token_ids": "argmax token id at every generation position only",
            "commit_positions_abs": "absolute sequence positions actually committed at this sub-step",
            "commit_positions_rel": "generation-relative positions actually committed at this sub-step",
            "commit_token_ids": "token ids actually written into x at commit_positions_abs",
        }
    return x, nfe, info


@torch.no_grad()
def generate_fixed_block_period_newline(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    block_length: int,
    steps_per_block: int,
    tokenizer: Any,
    newline_token_id: int,
    temperature: float = 0.0,
    threshold: Optional[float] = 0.9,
    gold_prefix_tokens: Optional[torch.Tensor] = None,
    save_top1_trace: bool = False,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """Fixed-block decoding with conservative committed period -> newline intervention."""
    if tokenizer is None:
        raise ValueError("tokenizer is required for period/newline intervention")

    device = model.device
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0, (
        f"gen_length({gen_length})는 block_length({block_length})의 배수여야 합니다."
    )
    num_blocks = gen_length // block_length
    nfe = 0
    gold_prefix_tokens_used = 0
    top1_trace: List[Dict[str, Any]] = []
    global_step = 0
    intervention_events: List[Dict[str, Any]] = []
    trigger_type_counts = {"previous_newline": 0, "same_step_newline": 0}

    if gold_prefix_tokens is not None:
        if gold_prefix_tokens.dim() != 1:
            gold_prefix_tokens = gold_prefix_tokens.view(-1)
        gold_prefix_tokens = gold_prefix_tokens.to(device=device, dtype=torch.long)
        gold_prefix_tokens_used = min(int(gold_prefix_tokens.numel()), int(gen_length))
        if gold_prefix_tokens_used > 0:
            x[:, prompt_len: prompt_len + gold_prefix_tokens_used] = gold_prefix_tokens[
                :gold_prefix_tokens_used
            ].unsqueeze(0)

    decode_start_block = gold_prefix_tokens_used // block_length

    for num_block in range(decode_start_block, num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        block_mask = (x[:, block_start:block_end] == mask_id)
        if block_mask.sum() == 0:
            continue

        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        step_i = 0

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            nfe += 1
            mask_idx = (x == mask_id)
            mask_idx[:, block_end:] = False

            logits = model(x).logits
            logits_noisy = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            probs = F.softmax(logits.to(torch.float64), dim=-1)
            score = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0 = torch.where(mask_idx, x0, x)
            neg_inf = torch.tensor(
                torch.finfo(score.dtype).min, device=device, dtype=score.dtype
            )
            confidence = torch.where(mask_idx, score, neg_inf)

            if threshold is not None:
                transfer_index = select_transfer_index_threshold(
                    confidence, mask_idx, threshold
                )
            else:
                max_i = num_transfer.size(1) - 1
                si = min(step_i, max_i)
                transfer_index = select_transfer_index_topk(
                    confidence, mask_idx, num_transfer[:, si]
                )

            commit_abs = torch.nonzero(transfer_index[0], as_tuple=False).view(-1)
            commit_abs_list = [int(pos) for pos in commit_abs.detach().cpu().tolist()]
            original_commit_ids = x0[0, commit_abs].detach().cpu().tolist()
            commit_conf = confidence[0, commit_abs].detach().cpu().tolist()
            commit_rel_list = [int(pos - prompt_len) for pos in commit_abs_list]

            step_events: List[Dict[str, Any]] = []
            for abs_pos in commit_abs_list:
                rel_pos = abs_pos - prompt_len
                if rel_pos < 0 or rel_pos + 1 >= gen_length:
                    continue
                original_token_id = int(x0[0, abs_pos].item())
                if tokenizer.decode([original_token_id], skip_special_tokens=False) != ".":
                    continue

                next_abs = abs_pos + 1
                previous_newline = int(x[0, next_abs].item()) == int(newline_token_id)
                same_step_newline = (
                    bool(transfer_index[0, next_abs].item())
                    and int(x0[0, next_abs].item()) == int(newline_token_id)
                )
                if not (previous_newline or same_step_newline):
                    continue

                trigger_type = "previous_newline" if previous_newline else "same_step_newline"
                x0[0, abs_pos] = int(newline_token_id)
                event = {
                    "global_step": int(global_step),
                    "block_idx": int(num_block),
                    "sub_step": int(step_i),
                    "abs_pos": int(abs_pos),
                    "rel_pos": int(rel_pos),
                    "next_abs_pos": int(next_abs),
                    "next_rel_pos": int(rel_pos + 1),
                    "original_token_id": int(original_token_id),
                    "replacement_token_id": int(newline_token_id),
                    "trigger_type": trigger_type,
                    "previous_newline": bool(previous_newline),
                    "same_step_newline": bool(same_step_newline),
                }
                step_events.append(event)
                intervention_events.append(event)
                trigger_type_counts[trigger_type] += 1

            if save_top1_trace:
                final_commit_ids = x0[0, commit_abs].detach().cpu().tolist()
                row = {
                    "global_step": int(global_step),
                    "block_idx": int(num_block),
                    "sub_step": int(step_i),
                    "block_start_rel": int(num_block * block_length),
                    "block_end_rel": int((num_block + 1) * block_length),
                    "block_start_abs": int(block_start),
                    "block_end_abs": int(block_end),
                    "remaining_before": int(remaining),
                    "top1_token_ids": [int(t) for t in x0[0].detach().cpu().tolist()],
                    "top1_generation_token_ids": [
                        int(t) for t in x0[0, prompt_len:].detach().cpu().tolist()
                    ],
                    "commit_positions_abs": commit_abs_list,
                    "commit_positions_rel": commit_rel_list,
                    "commit_positions_in_block": [
                        int(pos - block_start) for pos in commit_abs_list
                    ],
                    "commit_token_ids_original": [int(t) for t in original_commit_ids],
                    "commit_token_ids": [int(t) for t in final_commit_ids],
                    "commit_confidences": [float(c) for c in commit_conf],
                    "period_newline_events": step_events,
                    "commit_tokens_original": tokenizer.convert_ids_to_tokens(
                        [int(t) for t in original_commit_ids]
                    ),
                    "commit_tokens": tokenizer.convert_ids_to_tokens(
                        [int(t) for t in final_commit_ids]
                    ),
                    "commit_text_original": tokenizer.decode(
                        [int(t) for t in original_commit_ids],
                        skip_special_tokens=False,
                    ),
                    "commit_text": tokenizer.decode(
                        [int(t) for t in final_commit_ids],
                        skip_special_tokens=False,
                    ),
                }
                top1_trace.append(row)

            x[transfer_index] = x0[transfer_index]
            if save_top1_trace:
                top1_trace[-1]["remaining_after"] = int(
                    (x[:, block_start:block_end] == mask_id).sum().item()
                )
            step_i += 1
            global_step += 1

    info = {
        "block_length": block_length,
        "num_blocks": num_blocks,
        "gold_prefix_tokens_used": int(gold_prefix_tokens_used),
        "gold_prefix_truncated": bool(
            gold_prefix_tokens is not None
            and int(gold_prefix_tokens.numel()) > int(gen_length)
        ),
        "decode_start_block": int(decode_start_block),
        "intervention_name": "period_to_newline_next_newline",
        "newline_token_id": int(newline_token_id),
        "total_replacements": int(len(intervention_events)),
        "per_sample_intervention_count": int(len(intervention_events)),
        "affected_sample": bool(intervention_events),
        "trigger_type_counts": {
            key: int(value) for key, value in trigger_type_counts.items()
        },
        "intervention_events": intervention_events,
    }
    if save_top1_trace:
        info["top1_trace"] = top1_trace
        info["top1_trace_schema"] = {
            "commit_token_ids_original": "argmax token ids selected for commit before intervention",
            "commit_token_ids": "token ids committed after intervention replacement",
            "period_newline_events": "period-to-newline replacements fired at this sub-step",
        }

    return x, nfe, info


@torch.no_grad()
def generate_block_argmax1(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    block_length: int,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0, (
        f"gen_length({gen_length})는 block_length({block_length})의 배수여야 합니다."
    )
    num_blocks = gen_length // block_length
    nfe = 0

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            nfe += 1
            mask_idx = (x == mask_id)
            mask_idx[:, :block_start] = False
            mask_idx[:, block_end:] = False

            outputs = model(x)
            logits = outputs.logits

            logits_noisy = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            probs = F.softmax(logits.to(torch.float64), dim=-1)
            score = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_idx, x0, x)
            neg_inf = torch.tensor(
                torch.finfo(score.dtype).min, device=device, dtype=score.dtype
            )
            confidence = torch.where(mask_idx, score, neg_inf)
            per_step = torch.ones((confidence.size(0),), device=device, dtype=torch.long)
            transfer_index = select_transfer_index_topk(confidence, mask_idx, per_step)

            x[transfer_index] = x0[transfer_index]

    info = {
        "block_length": block_length,
        "num_blocks": num_blocks,
        "decode_policy": "block_local_confidence_argmax_one_token",
    }
    return x, nfe, info


# ═══════════════════════════════════════════════════════════════════════════
# Generation: Adaptive Block Scheduling
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_adaptive(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    steps_per_block: int = 32,
    tau: float = 0.9,
    min_block: int = 8,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, int, List[Dict[str, Any]]]:
    """
    Adaptive Block Scheduling: 매 블록 완료 후 남은 MASK 영역의
    confidence profile을 보고 다음 블록 크기를 동적으로 결정.

    블록 크기 결정 기준:
      safe_prefix = confidence >= tau 인 연속 prefix 길이
      next_block  = max(min_block, min(safe_prefix, remaining))

    기존 fixed/hybrid-cdf 방식과 달리 Step 0 rollout이 불필요하며,
    생성 도중 현재 상태에 맞게 블록 경계를 적응적으로 결정한다.
    """
    device = model.device
    prompt_len = prompt.shape[1]
    neg_inf = torch.finfo(torch.float32).min

    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    cursor = prompt_len  # 현재까지 확정된 위치 (절대 인덱스)
    nfe = 0
    block_log: List[Dict[str, Any]] = []

    while cursor < prompt_len + gen_length:
        remaining = prompt_len + gen_length - cursor

        # ── Confidence probe: 남은 MASK 전체에 대한 예측 ──
        # 미래 MASK는 그대로 두되, probe forward pass 카운트에 포함
        nfe += 1
        with torch.no_grad():
            outputs = model(x)
        logits_all = outputs.logits  # (1, full_len, vocab)

        # cursor 위치부터의 confidence (남은 영역)
        logits_rem = logits_all[:, cursor:cursor + remaining, :]  # (1, remaining, vocab)
        probs_rem = F.softmax(logits_rem.to(torch.float32), dim=-1)
        conf = probs_rem.max(dim=-1).values[0]  # (remaining,)

        # safe_prefix: confidence >= tau 인 연속 prefix 길이
        safe_prefix = 0
        for c_val in conf:
            if c_val.item() >= tau:
                safe_prefix += 1
            else:
                break

        # 블록 크기 결정
        next_block = max(min_block, min(safe_prefix if safe_prefix > 0 else min_block, remaining))
        next_block = min(next_block, remaining)  # 마지막 블록 clamp

        block_start = cursor
        block_end = cursor + next_block

        conf_at_boundary = conf[next_block - 1].item() if next_block <= remaining else None

        # ── Block denoising (generate_fixed_block Step 1+ 동일 로직) ──
        block_mask = (x[:, block_start:block_end] == mask_id)
        if block_mask.sum() > 0:
            num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
            step_i = 0

            while True:
                remaining_in_block = (x[:, block_start:block_end] == mask_id).sum().item()
                if remaining_in_block == 0:
                    break

                nfe += 1
                mask_idx = (x == mask_id)
                mask_idx[:, block_end:] = False  # 미래 블록 MASK 유지

                outputs = model(x)
                logits = outputs.logits

                logits_noisy = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_noisy, dim=-1)

                probs = F.softmax(logits.to(torch.float64), dim=-1)
                score = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

                x0 = torch.where(mask_idx, x0, x)
                confidence = torch.where(
                    mask_idx,
                    score,
                    torch.tensor(neg_inf, device=device, dtype=score.dtype),
                )

                max_i = num_transfer.size(1) - 1
                si = min(step_i, max_i)
                per_step = num_transfer[:, si]
                transfer_index = select_transfer_index_topk(confidence, mask_idx, per_step)

                x[transfer_index] = x0[transfer_index]
                step_i += 1

        block_log.append({
            "start": block_start - prompt_len,
            "end": block_end - prompt_len,
            "size": next_block,
            "safe_prefix": safe_prefix,
            "conf_at_boundary": conf_at_boundary,
            "nfe_in_block": step_i if block_mask.sum() > 0 else 0,
        })

        cursor = block_end

    return x, nfe, block_log
