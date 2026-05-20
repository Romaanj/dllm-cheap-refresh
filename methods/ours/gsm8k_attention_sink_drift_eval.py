"""
GSM8K fixed-block decoding with attention sink event and KV drift logging.

This experiment runs GSM8K 5-shot fixed-block decoding and logs sink events
under multiple sink definitions:

  - mean-ratio threshold: incoming_mass > tau * mean
  - z-score threshold: (incoming_mass - mean) / std > z
  - top-k incoming mass

It also records head disagreement summaries and optional head-debug dumps.

Each birth event is annotated with:
  - region (prefix / current_block / suffix)
  - layer_group (early / mid / deep)
  - birth_transfer_class (transfer_from_prefix / de_novo / prefix_internal_transfer
    / transfer_from_current / mixed / suffix_birth)
  - lifetime_steps and is_persistent (lifetime >= 2 OR still alive)
  - step-level co-occurrence flags (step_has_{prefix,current_block,suffix}_{birth,death})

This fork also measures prefix KV cache staleness drift. At each decoded state,
the FRESH world is the production forward pass used for logits and sink event
detection. The STALE world simulates a prefix KV cache last refreshed at the
current block boundary. Drift is observational only: decode choices and event
detection always use production-fresh tensors.

Indexing conventions:
  - global_step = the decoded state at which drift is measured.
  - event_source_global_step = the transition that produced this state.
  - birth_lag_k = whether a persistent current-block birth occurred at the
    transition that produced state t-k.

Tensor conventions:
  - Token-level K/V drift uses PRE-GQA-expansion tensors with num_kv_heads,
    matching what an actual KV cache stores.
  - Context drift and attention-weighted V drift use POST-GQA-expansion tensors
    with num_attention_heads, matching what attention computation consumes.
  - attn_weighted_v_l2_* uses FRESH production attention weights applied to
    stale-vs-fresh V drift. It measures the damage current-block queries would
    deliver if they used stale V at the weight distribution they actually
    computed.

Output files:
  - step_rows.jsonl: sink-only step/layer rows.
  - event_rows.jsonl: sink birth/death events.
  - drift_rows.jsonl: per (sample, global_step, layer) drift rows with event
    flags needed for slicing.
  - per_sample_records.jsonl: full per-sample trace.
  - summary.json and summary_drift.csv: sink summaries plus drift aggregates.
  - summary_{block_length}.json, drift_rows_{block_length}.jsonl, and
    summary_drift_{block_length}.csv: block-size aliases for scaling runs.
  - cache_scaling_summary.csv and cache_scaling_report.txt: cross-block-size
    report when matching B=16/32/64/128 outputs are available.
"""

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from gsm8k_hybrid_cdf_eval import add_gumbel_noise, select_transfer_index_threshold
from model.modeling_llada import LLaDAModelLM


RELATION_WORDS = {
    "more", "than", "less", "left", "remaining", "total", "half", "twice",
    "difference", "ratio", "each", "every", "per", "between",
    "plus", "minus", "times", "divided",
    "equals", "equal", "double", "triple", "quarter", "third",
    "added", "subtracted", "multiplied",
    "altogether", "combined", "together", "split",
    "gives", "gave", "gets", "got", "needs", "needed", "takes", "took",
    "costs", "cost", "pays", "paid", "earns", "earned", "saves", "saved",
    "spends", "spent", "bought", "sold", "made", "lost", "gained",
    "if", "then", "since", "because", "after", "before",
    "how", "many", "much",
}

ANSWER_PHRASES = {
    "therefore", "so", "thus", "hence", "answer", "result",
    "final", "conclude", "conclusion", "####"
}

OPERATORS = {"+", "-", "*", "/", "=", "%", "x", "÷", "^", "×"}


def parse_int_list(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_float_list(raw: str) -> List[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    m = safe_mean(values)
    assert m is not None
    var = sum((v - m) ** 2 for v in values) / len(values)
    return float(math.sqrt(max(var, 0.0)))


def serialize_metric_def(metric: str, param: float) -> str:
    if float(param).is_integer():
        param_str = str(int(param))
    else:
        param_str = str(param)
    return f"{metric}:{param_str}"


def load_gsm8k(seed: int, split: str) -> Any:
    return load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=seed)


def format_gsm8k_demo(sample: Dict[str, Any]) -> str:
    return f"Question: {sample['question']}\nAnswer: {sample['answer']}"


def build_gsm8k_fewshot_text(sample: Dict[str, Any], demo_ds: Any, num_shots: int) -> str:
    demos = []
    if num_shots > 0:
        for demo in demo_ds:
            demos.append(format_gsm8k_demo(demo))
            if len(demos) >= num_shots:
                break
    target = f"Question: {sample['question']}\nAnswer:"
    return "\n\n".join(demos + [target]) if demos else str(sample["question"])


def get_prompt_text(
    tokenizer: AutoTokenizer,
    sample: Dict[str, Any],
    demo_ds: Any,
    num_shots: int,
    no_chat_template: bool,
) -> str:
    text = build_gsm8k_fewshot_text(sample, demo_ds, num_shots)
    if no_chat_template:
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def make_initial_sequence(
    input_ids: torch.Tensor,
    gen_length: int,
    mask_id: int,
    device: torch.device,
) -> torch.Tensor:
    prompt_len = input_ids.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = input_ids.clone()
    return x


def classify_token_gsm8k(token_str: str, token_id: int, mask_id: int) -> str:
    if token_id == mask_id:
        return "mask"
    t = token_str.strip()
    t_clean = t.lstrip("ĠĊ▁ ")

    if not t_clean:
        return "other"
    if "####" in t_clean:
        return "answer_phrase"
    if re.match(r"^-?\d[\d,]*\.?\d*$", t_clean):
        return "number"
    if t_clean in OPERATORS:
        return "operator"
    if t_clean.lower() in RELATION_WORDS:
        return "relation_word"
    if t_clean.lower() in ANSWER_PHRASES:
        return "answer_phrase"
    if t_clean[0].isupper() and t_clean.isalpha() and len(t_clean) > 1:
        return "entity"
    if all(c in "()[]{}:;,.<>!?@#$%^&*-+=/'\"\\|~`\n" for c in t_clean):
        return "punctuation"
    return "other"


def token_repr(tokenizer: AutoTokenizer, token_id: int, mask_id: int) -> Dict[str, Any]:
    if token_id == mask_id:
        return {
            "token_id": int(token_id),
            "token_str": "<MASK>",
            "token_text": "<MASK>",
            "token_category": "mask",
        }
    token_str = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return {
        "token_id": int(token_id),
        "token_str": token_str,
        "token_text": token_text,
        "token_category": classify_token_gsm8k(token_str, int(token_id), mask_id),
    }


def normalize_distribution(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.float64)
    x = values.to(torch.float64).clamp_min(0.0)
    total = x.sum()
    if float(total.item()) <= eps:
        return torch.full_like(x, 1.0 / max(x.numel(), 1), dtype=torch.float64)
    return x / total


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    if p.numel() == 0 or q.numel() == 0:
        return 0.0
    p = normalize_distribution(p, eps=eps).clamp_min(eps)
    q = normalize_distribution(q, eps=eps).clamp_min(eps)
    return float(torch.sum(p * torch.log(p / q)).item())


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    if p.numel() == 0 or q.numel() == 0:
        return 0.0
    p = normalize_distribution(p, eps=eps)
    q = normalize_distribution(q, eps=eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def get_region(abs_pos: int, block_start: int, block_end: int) -> str:
    if abs_pos < block_start:
        return "prefix"
    if abs_pos < block_end:
        return "current_block"
    return "suffix"


# Layer grouping for LLaDA-8B (32 layers total). Roughly third-split.
# early: shallow layers dominated by positional/structural signal.
# mid:   transitional layers where semantic structure starts to matter.
# deep:  layers where semantic sink pattern (punctuation triggers) is cleanest.
def layer_group_name(layer_idx: int, num_layers: Optional[int] = None) -> str:
    # Piloted on LLaDA-8B (32 layers). Cutoffs below also work reasonably for
    # other sizes because they are fractional thirds when num_layers is given.
    if num_layers is not None and num_layers > 0:
        third = max(1, num_layers // 3)
        if layer_idx < third:
            return "early"
        if layer_idx < 2 * third:
            return "mid"
        return "deep"
    # Fallback when num_layers is unknown at logging time: use the pilot-derived
    # cutoffs that worked for LLaDA-8B (early: 0-8, mid: 9-18, deep: 19+).
    if layer_idx <= 8:
        return "early"
    if layer_idx <= 18:
        return "mid"
    return "deep"


def sink_sets_from_incoming(
    incoming: torch.Tensor,
    mean_ratio_taus: Sequence[float],
    z_thresholds: Sequence[float],
    topk_values: Sequence[int],
) -> Dict[str, Set[int]]:
    result: Dict[str, Set[int]] = {}
    if incoming.numel() == 0:
        for tau in mean_ratio_taus:
            result[serialize_metric_def("mean_ratio", tau)] = set()
        for z in z_thresholds:
            result[serialize_metric_def("zscore", z)] = set()
        for k in topk_values:
            result[serialize_metric_def("topk", k)] = set()
        return result

    inc = incoming.to(torch.float64)
    mean_v = inc.mean()
    std_v = inc.std(unbiased=False)
    for tau in mean_ratio_taus:
        key = serialize_metric_def("mean_ratio", tau)
        mask = inc > (mean_v * float(tau))
        result[key] = set(torch.nonzero(mask, as_tuple=False).view(-1).tolist())
    for z in z_thresholds:
        key = serialize_metric_def("zscore", z)
        if float(std_v.item()) <= 1e-12:
            result[key] = set()
        else:
            zscores = (inc - mean_v) / std_v
            mask = zscores > float(z)
            result[key] = set(torch.nonzero(mask, as_tuple=False).view(-1).tolist())
    for k in topk_values:
        key = serialize_metric_def("topk", k)
        k_eff = min(max(int(k), 0), int(inc.numel()))
        if k_eff <= 0:
            result[key] = set()
        else:
            idx = torch.argsort(inc, descending=True)[:k_eff]
            result[key] = set(int(v) for v in idx.tolist())
    return result


def summarize_head_overlap(
    head_incoming: torch.Tensor,
    avg_topk_set: Set[int],
    topk: int,
) -> Dict[str, Any]:
    num_heads = int(head_incoming.shape[0])
    head_sets: List[Set[int]] = []
    for head_idx in range(num_heads):
        values = head_incoming[head_idx]
        k_eff = min(max(int(topk), 0), int(values.numel()))
        if k_eff <= 0:
            head_sets.append(set())
            continue
        idx = torch.argsort(values, descending=True)[:k_eff]
        head_sets.append(set(int(v) for v in idx.tolist()))

    pairwise = []
    for a, b in combinations(head_sets, 2):
        pairwise.append(jaccard(a, b))
    union_set: Set[int] = set()
    intersection_set: Optional[Set[int]] = None
    freq = Counter()
    for s in head_sets:
        union_set |= s
        intersection_set = set(s) if intersection_set is None else (intersection_set & s)
        for item in s:
            freq[item] += 1
    majority_cut = max(1, math.ceil(num_heads / 2))
    majority_set = {idx for idx, count in freq.items() if count >= majority_cut}
    return {
        "num_heads": num_heads,
        "pairwise_jaccard_mean": safe_mean(pairwise),
        "pairwise_jaccard_std": safe_std(pairwise),
        "union_size": int(len(union_set)),
        "intersection_size": int(len(intersection_set or set())),
        "majority_size": int(len(majority_set)),
        "avg_topk_size": int(len(avg_topk_set)),
        "avg_vs_majority_jaccard": jaccard(avg_topk_set, majority_set),
        "head_sets": [sorted(s) for s in head_sets],
        "majority_set": sorted(majority_set),
    }


class StreamingSinkAnalyzer:
    def __init__(
        self,
        prompt_len: int,
        mean_ratio_taus: Sequence[float],
        z_thresholds: Sequence[float],
        topk_values: Sequence[int],
        head_overlap_topk: int,
        drift_oracle: Optional["ProductionKVDriftOracle"] = None,
    ) -> None:
        self.prompt_len = int(prompt_len)
        self.mean_ratio_taus = list(mean_ratio_taus)
        self.z_thresholds = list(z_thresholds)
        self.topk_values = list(topk_values)
        self.head_overlap_topk = int(head_overlap_topk)
        self.block_start = self.prompt_len
        self.block_end = self.prompt_len
        self.seq_len = 0
        self.avg_sink_sets: Dict[str, Dict[int, Set[int]]] = defaultdict(dict)
        self.prefix_dists: Dict[int, torch.Tensor] = {}
        self.layer_summaries: List[Dict[str, Any]] = []
        self.head_overlap_rows: List[Dict[str, Any]] = []
        self.head_debug_rows: List[Dict[str, Any]] = []
        self.full_head_incoming: List[torch.Tensor] = []
        self.drift_oracle = drift_oracle
        self._layer_idx = 0
        self._hooks: List[Any] = []

    def set_context(self, block_start: int, block_end: int) -> None:
        self.block_start = int(block_start)
        self.block_end = int(block_end)
        self.seq_len = 0
        self.avg_sink_sets = defaultdict(dict)
        self.prefix_dists = {}
        self.layer_summaries = []
        self.head_overlap_rows = []
        self.head_debug_rows = []
        self.full_head_incoming = []
        if self.drift_oracle is not None:
            self.drift_oracle.set_context(block_start=block_start, block_end=block_end)
        self._layer_idx = 0

    def _hook_fn(self, module, input, output):
        x, cache, attn_weights = output
        layer_idx = self._layer_idx
        self._layer_idx += 1
        if attn_weights is None:
            return output

        a = attn_weights[0].to(torch.float64)
        self.seq_len = int(a.shape[-1])
        prefix_end = int(self.block_start)
        avg_attn = a.mean(dim=0)
        incoming_avg = avg_attn.sum(dim=0)
        head_incoming = a.sum(dim=1).detach().cpu()
        self.full_head_incoming.append(head_incoming)
        self.prefix_dists[layer_idx] = normalize_distribution(incoming_avg[:prefix_end]).detach().cpu()

        sink_map = sink_sets_from_incoming(
            incoming=incoming_avg,
            mean_ratio_taus=self.mean_ratio_taus,
            z_thresholds=self.z_thresholds,
            topk_values=self.topk_values,
        )
        for metric_key, sink_set in sink_map.items():
            self.avg_sink_sets[metric_key][layer_idx] = sink_set

        layer_summary = {
            "layer": int(layer_idx),
            "incoming_mean": float(incoming_avg.mean().item()) if incoming_avg.numel() else 0.0,
            "incoming_std": float(incoming_avg.std(unbiased=False).item()) if incoming_avg.numel() else 0.0,
            "incoming_min": float(incoming_avg.min().item()) if incoming_avg.numel() else 0.0,
            "incoming_max": float(incoming_avg.max().item()) if incoming_avg.numel() else 0.0,
            "prefix_len": int(prefix_end),
            "seq_len": int(self.seq_len),
            "sink_counts": {metric_key: int(len(sink_set)) for metric_key, sink_set in sink_map.items()},
        }
        self.layer_summaries.append(layer_summary)

        avg_topk_key = serialize_metric_def("topk", self.head_overlap_topk)
        avg_topk_set = sink_map.get(avg_topk_key, set())
        overlap = summarize_head_overlap(head_incoming, avg_topk_set, self.head_overlap_topk)
        self.head_overlap_rows.append({
            "layer": int(layer_idx),
            "topk": int(self.head_overlap_topk),
            "pairwise_jaccard_mean": overlap["pairwise_jaccard_mean"],
            "pairwise_jaccard_std": overlap["pairwise_jaccard_std"],
            "union_size": overlap["union_size"],
            "intersection_size": overlap["intersection_size"],
            "majority_size": overlap["majority_size"],
            "avg_topk_size": overlap["avg_topk_size"],
            "avg_vs_majority_jaccard": overlap["avg_vs_majority_jaccard"],
        })
        self.head_debug_rows.append({
            "layer": int(layer_idx),
            "topk": int(self.head_overlap_topk),
            "avg_topk_positions": sorted(avg_topk_set),
            "majority_positions": overlap["majority_set"],
            "head_topk_positions": overlap["head_sets"],
        })

        if self.drift_oracle is not None:
            self.drift_oracle.compute_layer(layer_idx, attn_weights, sink_set=sink_map.get("zscore:3", set()))

        return (x, cache, None)

    def register(self, blocks: Sequence[Any]) -> None:
        for block in blocks:
            self._hooks.append(block.register_forward_hook(self._hook_fn))

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "avg_sink_sets": self.avg_sink_sets,
            "prefix_dists": self.prefix_dists,
            "layer_summaries": self.layer_summaries,
            "head_overlap_rows": self.head_overlap_rows,
            "head_debug_rows": self.head_debug_rows,
            "full_head_incoming": self.full_head_incoming,
            "drift_rows": self.drift_oracle.drift_rows if self.drift_oracle is not None else [],
        }


def _stats(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    vals = values.to(torch.float64)
    return {
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        "max": float(vals.max().item()),
        "min": float(vals.min().item()),
    }


def _nan_stats() -> Dict[str, float]:
    nan = float("nan")
    return {"mean": nan, "std": nan, "max": nan, "min": nan}


def _expand_kv_to_q_heads(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    num_kv_heads = int(x.size(1))
    if num_kv_heads == num_q_heads:
        return x
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"Cannot GQA-expand {num_kv_heads} kv heads to {num_q_heads} q heads")
    return x.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)


class ProductionKVDriftOracle:
    """Capture production attention tensors and compute prefix stale-cache drift.

    Hooks observe tensors produced by the real forward pass. They do not rewrite
    outputs and do not rerun Q/K/V projections.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        no_drift_logging: bool = False,
        assert_boundary_zero: bool = True,
    ) -> None:
        self.eps = float(eps)
        self.no_drift_logging = bool(no_drift_logging)
        self.assert_boundary_zero = bool(assert_boundary_zero)
        self.block_start = 0
        self.block_end = 0
        self.sub_step = 0
        self.layer_tmp: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.stale_prefix: Dict[int, Dict[str, torch.Tensor]] = {}
        self.drift_rows: List[Dict[str, Any]] = []
        self._hooks: List[Any] = []

    def reset_block(self) -> None:
        self.stale_prefix = {}

    def set_step(self, sub_step: int) -> None:
        self.sub_step = int(sub_step)
        self.drift_rows = []
        self.layer_tmp = defaultdict(dict)

    def set_context(self, block_start: int, block_end: int) -> None:
        self.block_start = int(block_start)
        self.block_end = int(block_end)

    def _capture_llama_v_proj(self, layer_idx: int, output: torch.Tensor, block: Any) -> None:
        if self.no_drift_logging:
            return
        B, T, _ = output.shape
        head_dim = int(block.config.d_model // block.config.n_heads)
        heads = int(block.config.effective_n_kv_heads)
        self.layer_tmp[layer_idx]["v_pre"] = output.detach().view(B, T, heads, head_dim).transpose(1, 2)

    def _capture_sequential_att_proj(self, layer_idx: int, output: torch.Tensor, block: Any) -> None:
        if self.no_drift_logging:
            return
        B, T, _ = output.shape
        head_dim = int(block.config.d_model // block.config.n_heads)
        _, _, v = output.detach().split(block.fused_dims, dim=-1)
        self.layer_tmp[layer_idx]["v_pre"] = v.view(B, T, block.config.effective_n_kv_heads, head_dim).transpose(1, 2)

    def _capture_rope(self, layer_idx: int, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        if self.no_drift_logging:
            return
        q, k = output
        self.layer_tmp[layer_idx]["q_post"] = q.detach()
        self.layer_tmp[layer_idx]["k_post"] = k.detach()

    def _capture_context(self, layer_idx: int, input: Tuple[torch.Tensor, ...]) -> None:
        if self.no_drift_logging or not input:
            return
        self.layer_tmp[layer_idx]["fresh_context"] = input[0].detach()

    def register(self, blocks: Sequence[Any]) -> None:
        if self.no_drift_logging:
            return
        for layer_idx, block in enumerate(blocks):
            if hasattr(block, "q_proj") and hasattr(block, "k_proj") and hasattr(block, "v_proj"):
                self._hooks.append(block.v_proj.register_forward_hook(
                    lambda module, inp, out, li=layer_idx, b=block: self._capture_llama_v_proj(li, out, b)
                ))
            elif hasattr(block, "att_proj") and hasattr(block, "fused_dims"):
                self._hooks.append(block.att_proj.register_forward_hook(
                    lambda module, inp, out, li=layer_idx, b=block: self._capture_sequential_att_proj(li, out, b)
                ))
            else:
                raise TypeError(f"Unsupported block type for drift capture: {type(block)}")
            if hasattr(block, "rotary_emb"):
                self._hooks.append(block.rotary_emb.register_forward_hook(
                    lambda module, inp, out, li=layer_idx: self._capture_rope(li, out)
                ))
            else:
                raise TypeError(f"Block has no rotary_emb for post-RoPE K capture: {type(block)}")
            self._hooks.append(block.attn_out.register_forward_pre_hook(
                lambda module, inp, li=layer_idx: self._capture_context(li, inp)
            ))

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def compute_layer(self, layer_idx: int, attn_weights: torch.Tensor, sink_set: Set[int]) -> None:
        if self.no_drift_logging:
            return
        tmp = self.layer_tmp.pop(layer_idx, {})
        required = ("q_post", "k_post", "v_pre", "fresh_context")
        if not all(k in tmp for k in required):
            raise RuntimeError(f"Missing drift capture tensors for layer {layer_idx}: {sorted(tmp)}")

        q = tmp["q_post"]
        k_fresh = tmp["k_post"]
        v_fresh = tmp["v_pre"]
        fresh_context = tmp["fresh_context"]
        prefix_len = int(self.block_start)
        seq_len = int(k_fresh.size(-2))
        query_end = min(int(self.block_end), seq_len)
        query_start = min(int(self.block_start), query_end)

        stale = self.stale_prefix.get(layer_idx)
        if stale is None:
            stale = {
                "k": k_fresh[:, :, :prefix_len, :].detach().clone(),
                "v": v_fresh[:, :, :prefix_len, :].detach().clone(),
            }
            self.stale_prefix[layer_idx] = stale

        k_stale = stale["k"].to(device=k_fresh.device, dtype=k_fresh.dtype)
        v_stale = stale["v"].to(device=v_fresh.device, dtype=v_fresh.dtype)
        k_prefix = k_fresh[:, :, :prefix_len, :]
        v_prefix = v_fresh[:, :, :prefix_len, :]

        if k_stale.shape != k_prefix.shape or v_stale.shape != v_prefix.shape:
            raise RuntimeError(
                f"Stale/fresh prefix shape mismatch at layer {layer_idx}: "
                f"k {tuple(k_stale.shape)} vs {tuple(k_prefix.shape)}, "
                f"v {tuple(v_stale.shape)} vs {tuple(v_prefix.shape)}"
            )

        num_q_heads = int(q.size(1))
        k_stale_exp = _expand_kv_to_q_heads(k_stale, num_q_heads)
        v_stale_exp = _expand_kv_to_q_heads(v_stale, num_q_heads)
        k_fresh_exp = _expand_kv_to_q_heads(k_fresh, num_q_heads)
        v_fresh_exp = _expand_kv_to_q_heads(v_fresh, num_q_heads)
        v_prefix_fresh_exp = v_fresh_exp[:, :, :prefix_len, :]
        v_prefix_stale_exp = v_stale_exp

        q_cur = q[:, :, query_start:query_end, :]
        fresh_ctx_cur = fresh_context[:, query_start:query_end, :]
        attn_cur = attn_weights.detach()[:, :, query_start:query_end, :]
        attn_prefix = attn_cur[:, :, :, :prefix_len]
        eps_t = torch.tensor(self.eps, device=k_fresh.device, dtype=torch.float32)
        prefix_device = k_fresh.device
        all_prefix_idx = torch.arange(prefix_len, device=prefix_device, dtype=torch.long)
        sink_positions = sorted(int(p) for p in sink_set if 0 <= int(p) < prefix_len)
        sink_idx = torch.tensor(sink_positions, device=prefix_device, dtype=torch.long)
        is_sink = torch.zeros(prefix_len, device=prefix_device, dtype=torch.bool)
        if sink_idx.numel() > 0:
            is_sink[sink_idx] = True
        non_sink_idx = all_prefix_idx[~is_sink]
        if int(sink_idx.numel()) + int(non_sink_idx.numel()) != prefix_len:
            raise AssertionError(
                f"sink/non-sink prefix partition mismatch at layer {layer_idx}: "
                f"{int(sink_idx.numel())}+{int(non_sink_idx.numel())}!={prefix_len}"
            )

        # FRESH production attention weights over stale-vs-fresh V drift.
        v_token_diff_exp = torch.linalg.vector_norm(
            (v_prefix_stale_exp - v_prefix_fresh_exp).float(), ord=2, dim=-1
        )

        def context_rel_and_cos(candidate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            diff = torch.linalg.vector_norm((candidate - fresh_ctx_cur).float(), ord=2, dim=-1)
            denom = torch.linalg.vector_norm(fresh_ctx_cur.float(), ord=2, dim=-1).clamp_min(eps_t)
            rel = diff / denom
            cos = F.cosine_similarity(candidate.float(), fresh_ctx_cur.float(), dim=-1, eps=self.eps)
            return rel, cos

        def token_metric_stats(selected_idx: torch.Tensor) -> Dict[str, Dict[str, float]]:
            if selected_idx.numel() == 0:
                return {"k_rel": _nan_stats(), "v_rel": _nan_stats(), "k_cos": _nan_stats(), "v_cos": _nan_stats()}
            k_sel_stale = k_stale.index_select(2, selected_idx)
            k_sel_fresh = k_prefix.index_select(2, selected_idx)
            v_sel_stale = v_stale.index_select(2, selected_idx)
            v_sel_fresh = v_prefix.index_select(2, selected_idx)
            k_diff = torch.linalg.vector_norm((k_sel_stale - k_sel_fresh).float(), ord=2, dim=-1)
            v_diff = torch.linalg.vector_norm((v_sel_stale - v_sel_fresh).float(), ord=2, dim=-1)
            k_norm = torch.linalg.vector_norm(k_sel_fresh.float(), ord=2, dim=-1).clamp_min(eps_t)
            v_norm = torch.linalg.vector_norm(v_sel_fresh.float(), ord=2, dim=-1).clamp_min(eps_t)
            return {
                "k_rel": _stats((k_diff / k_norm).reshape(-1)),
                "v_rel": _stats((v_diff / v_norm).reshape(-1)),
                "k_cos": _stats(F.cosine_similarity(k_sel_stale.float(), k_sel_fresh.float(), dim=-1, eps=self.eps).reshape(-1)),
                "v_cos": _stats(F.cosine_similarity(v_sel_stale.float(), v_sel_fresh.float(), dim=-1, eps=self.eps).reshape(-1)),
            }

        def make_class_contexts(selected_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Build class-only stale contexts.

            V-only:
              V_class_stale[i] = V_stale[i] if i in class else V_fresh[i],
              for prefix positions only. V outside prefix is always fresh.

            Full:
              K_class_stale[i] = K_stale[i] if i in class else K_fresh[i],
              V_class_stale[i] = V_stale[i] if i in class else V_fresh[i],
              for prefix positions only. K/V outside prefix are always fresh.
            """
            v_mixed = v_fresh_exp.clone()
            k_mixed = k_fresh_exp.clone()
            if selected_idx.numel() > 0:
                v_mixed[:, :, selected_idx, :] = v_stale_exp.index_select(2, selected_idx)
                k_mixed[:, :, selected_idx, :] = k_stale_exp.index_select(2, selected_idx)
            v_only_heads = torch.matmul(attn_cur.to(v_mixed.dtype), v_mixed)
            v_only_ctx = v_only_heads.transpose(1, 2).contiguous().view(q.size(0), query_end - query_start, -1)
            scores = torch.matmul(q_cur, k_mixed.transpose(-2, -1)) * (1.0 / math.sqrt(q_cur.size(-1)))
            class_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_cur.dtype)
            full_heads = torch.matmul(class_weights, v_mixed)
            full_ctx = full_heads.transpose(1, 2).contiguous().view(q.size(0), query_end - query_start, -1)
            return v_only_ctx, full_ctx, class_weights

        def class_stats(selected_idx: torch.Tensor) -> Tuple[Dict[str, Dict[str, float]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            token_stats = token_metric_stats(selected_idx)
            if selected_idx.numel() == 0:
                nan = _nan_stats()
                token_stats.update({
                    "awv": nan,
                    "v_only_rel": nan,
                    "v_only_cos": nan,
                    "full_rel": nan,
                    "full_cos": nan,
                })
                empty = torch.empty(0, device=prefix_device)
                return token_stats, empty, empty, empty, empty
            v_only_ctx, full_ctx, class_weights = make_class_contexts(selected_idx)
            v_only_rel, v_only_cos = context_rel_and_cos(v_only_ctx)
            full_rel, full_cos = context_rel_and_cos(full_ctx)
            awv = (
                attn_prefix.index_select(-1, selected_idx).float()
                * v_token_diff_exp.index_select(-1, selected_idx).unsqueeze(2)
            ).sum(dim=-1)
            token_stats.update({
                "awv": _stats(awv.reshape(-1)),
                "v_only_rel": _stats(v_only_rel.reshape(-1)),
                "v_only_cos": _stats(v_only_cos.reshape(-1)),
                "full_rel": _stats(full_rel.reshape(-1)),
                "full_cos": _stats(full_cos.reshape(-1)),
            })
            return token_stats, awv, full_rel, class_weights, full_ctx

        full_stats, awv_full, full_rel, stale_weights, stale_full_ctx = class_stats(all_prefix_idx)
        sink_stats, awv_sink, full_rel_sink, stale_weights_sink, _ = class_stats(sink_idx)
        non_sink_stats, awv_non_sink, full_rel_non_sink, _, _ = class_stats(non_sink_idx)

        sink_mask_exp = is_sink.to(attn_prefix.dtype).view(1, 1, 1, prefix_len)
        fresh_sink_mass = (attn_prefix * sink_mask_exp).sum(dim=-1).float()
        stale_sink_mass = (stale_weights[:, :, :, :prefix_len] * sink_mask_exp.to(stale_weights.dtype)).sum(dim=-1).float()
        total_prefix_attn = attn_prefix.sum(dim=-1).float()
        stale_total_prefix_attn = stale_weights[:, :, :, :prefix_len].sum(dim=-1).float()
        sink_attn_mass_diff = torch.abs(fresh_sink_mass - stale_sink_mass) / fresh_sink_mass.clamp_min(eps_t)
        fraction_sink_attn_t = fresh_sink_mass / total_prefix_attn.clamp_min(eps_t)
        stale_sink_fraction_t = stale_sink_mass / stale_total_prefix_attn.clamp_min(eps_t)
        fresh_sink_fraction_t = fresh_sink_mass / total_prefix_attn.clamp_min(eps_t)
        sink_fraction_diff_t = stale_sink_fraction_t - fresh_sink_fraction_t

        fresh_ctx_norm = torch.linalg.vector_norm(fresh_ctx_cur.float(), ord=2, dim=-1).clamp_min(eps_t)
        stale_ctx_norm = torch.linalg.vector_norm(stale_full_ctx.float(), ord=2, dim=-1)
        norm_ratio_t = stale_ctx_norm / fresh_ctx_norm

        numerator = awv_sink.sum() if awv_sink.numel() else torch.tensor(0.0, device=prefix_device)
        denominator = awv_full.sum() if awv_full.numel() else torch.tensor(0.0, device=prefix_device)
        attn_weighted_v_sink_share = torch.clamp(numerator / denominator.clamp_min(eps_t), 0.0, 1.0)

        fraction_sink_attn = float(fraction_sink_attn_t.mean().item()) if fraction_sink_attn_t.numel() else float("nan")
        sink_full_mean = sink_stats["full_rel"]["mean"]
        non_sink_full_mean = non_sink_stats["full_rel"]["mean"]
        sink_term = fraction_sink_attn * sink_full_mean if math.isfinite(sink_full_mean) else float("nan")
        non_sink_term = (1.0 - fraction_sink_attn) * non_sink_full_mean if math.isfinite(non_sink_full_mean) else float("nan")

        row = {
            "layer": int(layer_idx),
            "prefix_len": int(prefix_len),
            "current_block_len": int(max(query_end - query_start, 0)),
            "seq_len": int(seq_len),
            "k_rel_l2_mean": full_stats["k_rel"]["mean"],
            "k_rel_l2_std": full_stats["k_rel"]["std"],
            "k_rel_l2_max": full_stats["k_rel"]["max"],
            "v_rel_l2_mean": full_stats["v_rel"]["mean"],
            "v_rel_l2_std": full_stats["v_rel"]["std"],
            "v_rel_l2_max": full_stats["v_rel"]["max"],
            "k_cos_mean": full_stats["k_cos"]["mean"],
            "k_cos_std": full_stats["k_cos"]["std"],
            "k_cos_min": full_stats["k_cos"]["min"],
            "v_cos_mean": full_stats["v_cos"]["mean"],
            "v_cos_std": full_stats["v_cos"]["std"],
            "v_cos_min": full_stats["v_cos"]["min"],
            "attn_weighted_v_l2_mean": full_stats["awv"]["mean"],
            "attn_weighted_v_l2_std": full_stats["awv"]["std"],
            "attn_weighted_v_l2_max": full_stats["awv"]["max"],
            "v_only_context_rel_l2_mean": full_stats["v_only_rel"]["mean"],
            "v_only_context_rel_l2_std": full_stats["v_only_rel"]["std"],
            "v_only_context_rel_l2_max": full_stats["v_only_rel"]["max"],
            "v_only_context_cos_mean": full_stats["v_only_cos"]["mean"],
            "v_only_context_cos_std": full_stats["v_only_cos"]["std"],
            "v_only_context_cos_min": full_stats["v_only_cos"]["min"],
            "full_context_rel_l2_mean": full_stats["full_rel"]["mean"],
            "full_context_rel_l2_std": full_stats["full_rel"]["std"],
            "full_context_rel_l2_max": full_stats["full_rel"]["max"],
            "full_context_cos_mean": full_stats["full_cos"]["mean"],
            "full_context_cos_std": full_stats["full_cos"]["std"],
            "full_context_cos_min": full_stats["full_cos"]["min"],
            "sink_attn_mass_fresh_mean": float(fresh_sink_mass.mean().item()) if fresh_sink_mass.numel() else float("nan"),
            "sink_attn_mass_stale_mean": float(stale_sink_mass.mean().item()) if stale_sink_mass.numel() else float("nan"),
            "sink_attn_mass_diff_mean": float(sink_attn_mass_diff.mean().item()) if sink_attn_mass_diff.numel() else float("nan"),
            "norm_ratio_mean": float(norm_ratio_t.mean().item()) if norm_ratio_t.numel() else float("nan"),
            "sink_fraction_diff_mean": float(sink_fraction_diff_t.mean().item()) if sink_fraction_diff_t.numel() else float("nan"),
            "attn_weighted_v_sink_share": float(attn_weighted_v_sink_share.item()),
            "fraction_sink_attn": fraction_sink_attn,
            "sink_term_contribution": float(sink_term),
            "non_sink_term_contribution": float(non_sink_term),
            "num_sink_prefix_tokens": int(sink_idx.numel()),
            "num_non_sink_prefix_tokens": int(non_sink_idx.numel()),
        }
        def add_class_fields(suffix: str, stats: Dict[str, Dict[str, float]]) -> None:
            row.update({
                f"k_rel_l2_mean_{suffix}": stats["k_rel"]["mean"],
                f"k_rel_l2_std_{suffix}": stats["k_rel"]["std"],
                f"k_rel_l2_max_{suffix}": stats["k_rel"]["max"],
                f"v_rel_l2_mean_{suffix}": stats["v_rel"]["mean"],
                f"v_rel_l2_std_{suffix}": stats["v_rel"]["std"],
                f"v_rel_l2_max_{suffix}": stats["v_rel"]["max"],
                f"k_cos_mean_{suffix}": stats["k_cos"]["mean"],
                f"k_cos_std_{suffix}": stats["k_cos"]["std"],
                f"k_cos_min_{suffix}": stats["k_cos"]["min"],
                f"v_cos_mean_{suffix}": stats["v_cos"]["mean"],
                f"v_cos_std_{suffix}": stats["v_cos"]["std"],
                f"v_cos_min_{suffix}": stats["v_cos"]["min"],
                f"attn_weighted_v_l2_mean_{suffix}": stats["awv"]["mean"],
                f"attn_weighted_v_l2_std_{suffix}": stats["awv"]["std"],
                f"attn_weighted_v_l2_max_{suffix}": stats["awv"]["max"],
                f"v_only_context_rel_l2_mean_{suffix}": stats["v_only_rel"]["mean"],
                f"v_only_context_rel_l2_std_{suffix}": stats["v_only_rel"]["std"],
                f"v_only_context_rel_l2_max_{suffix}": stats["v_only_rel"]["max"],
                f"v_only_context_cos_mean_{suffix}": stats["v_only_cos"]["mean"],
                f"v_only_context_cos_std_{suffix}": stats["v_only_cos"]["std"],
                f"v_only_context_cos_min_{suffix}": stats["v_only_cos"]["min"],
                f"full_context_rel_l2_mean_{suffix}": stats["full_rel"]["mean"],
                f"full_context_rel_l2_std_{suffix}": stats["full_rel"]["std"],
                f"full_context_rel_l2_max_{suffix}": stats["full_rel"]["max"],
                f"full_context_cos_mean_{suffix}": stats["full_cos"]["mean"],
                f"full_context_cos_std_{suffix}": stats["full_cos"]["std"],
                f"full_context_cos_min_{suffix}": stats["full_cos"]["min"],
            })

        add_class_fields("sink", sink_stats)
        add_class_fields("non_sink", non_sink_stats)
        if self.assert_boundary_zero and self.sub_step == 0:
            max_boundary = max(
                abs(float(row["k_rel_l2_max"])),
                abs(float(row["v_rel_l2_max"])),
                abs(float(row["v_only_context_rel_l2_max"])),
                abs(float(row["full_context_rel_l2_max"])),
                *[
                    abs(float(row[key]))
                    for key in (
                        "k_rel_l2_max_sink",
                        "v_rel_l2_max_sink",
                        "v_only_context_rel_l2_max_sink",
                        "full_context_rel_l2_max_sink",
                        "k_rel_l2_max_non_sink",
                        "v_rel_l2_max_non_sink",
                        "v_only_context_rel_l2_max_non_sink",
                        "full_context_rel_l2_max_non_sink",
                    )
                    if math.isfinite(float(row[key]))
                ],
            )
            if max_boundary > max(self.eps * 100.0, 1e-5):
                raise AssertionError(
                    f"Non-zero boundary drift at layer {layer_idx}: {max_boundary}"
                )
            norm_ratio_boundary = abs(float(row["norm_ratio_mean"]) - 1.0)
            if norm_ratio_boundary > 1e-4:
                raise AssertionError(
                    f"Boundary norm_ratio_mean drift at layer {layer_idx}: {norm_ratio_boundary}"
                )
            sink_fraction_boundary = abs(float(row["sink_fraction_diff_mean"]))
            if sink_fraction_boundary > 1e-5:
                raise AssertionError(
                    f"Boundary sink_fraction_diff_mean drift at layer {layer_idx}: {sink_fraction_boundary}"
                )
        awv_additive_err = abs(
            float(row["attn_weighted_v_l2_mean"])
            - (
                (0.0 if not math.isfinite(float(row["attn_weighted_v_l2_mean_sink"])) else float(row["attn_weighted_v_l2_mean_sink"]))
                + (0.0 if not math.isfinite(float(row["attn_weighted_v_l2_mean_non_sink"])) else float(row["attn_weighted_v_l2_mean_non_sink"]))
            )
        )
        if awv_additive_err > 1e-4:
            raise AssertionError(
                f"attn_weighted_v class contribution mismatch at layer {layer_idx}: {awv_additive_err}"
            )
        self.drift_rows.append(row)


@torch.no_grad()
def forward_with_sink_snapshot(
    model,
    x: torch.Tensor,
    prompt_len: int,
    block_start: int,
    block_end: int,
    mean_ratio_taus: Sequence[float],
    z_thresholds: Sequence[float],
    topk_values: Sequence[int],
    head_overlap_topk: int,
    drift_oracle: Optional[ProductionKVDriftOracle] = None,
    sub_step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    core_model = model.model if hasattr(model, "model") else model
    blocks_list = core_model.transformer.blocks
    analyzer = StreamingSinkAnalyzer(
        prompt_len=prompt_len,
        mean_ratio_taus=mean_ratio_taus,
        z_thresholds=z_thresholds,
        topk_values=topk_values,
        head_overlap_topk=head_overlap_topk,
        drift_oracle=drift_oracle,
    )
    analyzer.set_context(block_start=block_start, block_end=block_end)
    if drift_oracle is not None:
        drift_oracle.set_step(sub_step=sub_step)
        drift_oracle.register(blocks_list)
    analyzer.register(blocks_list)
    try:
        outputs = model(x, output_attentions=True)
    finally:
        analyzer.remove()
        if drift_oracle is not None:
            drift_oracle.remove()
    logits = outputs.logits
    snapshot = analyzer.snapshot()
    del outputs
    return logits, snapshot


def build_event_row(
    sample_id: int,
    global_step: int,
    block_idx: int,
    sub_step: int,
    layer_idx: int,
    metric_name: str,
    metric_param: float,
    metric_key: str,
    event_type: str,
    token_abs_pos: int,
    prompt_len: int,
    block_start: int,
    block_end: int,
    prev_ids: Sequence[int],
    curr_ids: Sequence[int],
    newly_unmasked_abs: Sequence[int],
    tokenizer: AutoTokenizer,
    mask_id: int,
) -> Dict[str, Any]:
    prev_repr = token_repr(tokenizer, int(prev_ids[token_abs_pos]), mask_id)
    curr_repr = token_repr(tokenizer, int(curr_ids[token_abs_pos]), mask_id)
    token_rel_pos = token_abs_pos - prompt_len
    region = get_region(token_abs_pos, block_start, block_end)
    is_new = token_abs_pos in set(newly_unmasked_abs)
    is_masked_now = int(curr_ids[token_abs_pos]) == mask_id
    was_unmasked_before = int(prev_ids[token_abs_pos]) != mask_id
    return {
        "sample_id": int(sample_id),
        "global_step": int(global_step),
        "block_idx": int(block_idx),
        "sub_step": int(sub_step),
        "layer": int(layer_idx),
        "layer_group": layer_group_name(int(layer_idx)),
        "sink_metric": metric_name,
        "sink_param": float(metric_param),
        "sink_metric_key": metric_key,
        "event_type": event_type,
        "token_abs_pos": int(token_abs_pos),
        "token_rel_pos": int(token_rel_pos),
        "region": region,
        "current_block_start_rel": int(block_start - prompt_len),
        "current_block_end_rel": int(block_end - prompt_len),
        "is_masked_now": bool(is_masked_now),
        "was_newly_unmasked_this_step": bool(is_new),
        "was_already_unmasked_before_step": bool(was_unmasked_before and not is_new),
        "birth_matches_newly_unmasked": bool(event_type == "birth" and is_new),
        "num_newly_unmasked": int(len(newly_unmasked_abs)),
        "newly_unmasked_abs_positions": [int(v) for v in newly_unmasked_abs],
        "newly_unmasked_rel_positions": [int(v - prompt_len) for v in newly_unmasked_abs],
        "prev_token_id": int(prev_repr["token_id"]),
        "prev_token_str": prev_repr["token_str"],
        "prev_token_text": prev_repr["token_text"],
        "prev_token_category": prev_repr["token_category"],
        "token_id": int(curr_repr["token_id"]),
        "token_str": curr_repr["token_str"],
        "token_text": curr_repr["token_text"],
        "token_category": curr_repr["token_category"],
    }


def finalize_transition(
    sample_id: int,
    pending: Dict[str, Any],
    current_snapshot: Dict[str, Any],
    tokenizer: AutoTokenizer,
    mask_id: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    prev_snapshot = pending["prev_snapshot"]
    prompt_len = int(pending["prompt_len"])
    block_start = int(pending["block_start"])
    block_end = int(pending["block_end"])
    prev_ids = pending["prev_token_ids"]
    curr_ids = pending["curr_token_ids"]
    newly_unmasked_abs = pending["newly_unmasked_abs"]
    newly_unmasked_set = set(newly_unmasked_abs)

    step_events: List[Dict[str, Any]] = []
    metric_summary_rows: List[Dict[str, Any]] = []
    all_metric_keys = sorted(set(prev_snapshot["avg_sink_sets"]) | set(current_snapshot["avg_sink_sets"]))

    new_token_categories = []
    new_token_texts = []
    for abs_pos in newly_unmasked_abs:
        info = token_repr(tokenizer, curr_ids[abs_pos], mask_id)
        new_token_categories.append(info["token_category"])
        new_token_texts.append(info["token_text"])
    punct_ratio = (
        sum(cat == "punctuation" for cat in new_token_categories) / len(new_token_categories)
        if new_token_categories else 0.0
    )

    step_record = {
        "sample_id": int(sample_id),
        "global_step": int(pending["global_step"]),
        "block_idx": int(pending["block_idx"]),
        "sub_step": int(pending["sub_step"]),
        "current_block_start_rel": int(block_start - prompt_len),
        "current_block_end_rel": int(block_end - prompt_len),
        "current_block_start_abs": int(block_start),
        "current_block_end_abs": int(block_end),
        "remaining_before": int(pending["remaining_before"]),
        "remaining_after": int(pending["remaining_after"]),
        "newly_unmasked_abs_positions": [int(v) for v in newly_unmasked_abs],
        "newly_unmasked_rel_positions": [int(v - prompt_len) for v in newly_unmasked_abs],
        "newly_unmasked_token_ids": [int(curr_ids[v]) for v in newly_unmasked_abs],
        "newly_unmasked_tokens": new_token_texts,
        "newly_unmasked_categories": new_token_categories,
        "newly_unmasked_punctuation_ratio": float(punct_ratio),
        "layer_metrics": [],
    }

    for metric_key in all_metric_keys:
        metric_name, metric_param_raw = metric_key.split(":", 1)
        metric_param = float(metric_param_raw)
        prev_layers = prev_snapshot["avg_sink_sets"].get(metric_key, {})
        curr_layers = current_snapshot["avg_sink_sets"].get(metric_key, {})
        num_layers = max(
            [0] + list(prev_layers.keys()) + list(curr_layers.keys())
        ) + 1

        for layer_idx in range(num_layers):
            prev_set = prev_layers.get(layer_idx, set())
            curr_set = curr_layers.get(layer_idx, set())
            births = sorted(curr_set - prev_set)
            deaths = sorted(prev_set - curr_set)
            persistent = sorted(curr_set & prev_set)
            prefix_births = [p for p in births if get_region(p, block_start, block_end) == "prefix"]
            prefix_deaths = [p for p in deaths if get_region(p, block_start, block_end) == "prefix"]
            current_births = [p for p in births if get_region(p, block_start, block_end) == "current_block"]
            suffix_births = [p for p in births if get_region(p, block_start, block_end) == "suffix"]

            prev_prefix = prev_snapshot["prefix_dists"].get(layer_idx, torch.empty(0, dtype=torch.float64))
            curr_prefix = current_snapshot["prefix_dists"].get(layer_idx, torch.empty(0, dtype=torch.float64))
            kl_fwd = kl_divergence(curr_prefix, prev_prefix)
            kl_rev = kl_divergence(prev_prefix, curr_prefix)
            jsd = js_divergence(prev_prefix, curr_prefix)

            metric_summary = {
                "sample_id": int(sample_id),
                "global_step": int(pending["global_step"]),
                "block_idx": int(pending["block_idx"]),
                "sub_step": int(pending["sub_step"]),
                "layer": int(layer_idx),
                "layer_group": layer_group_name(int(layer_idx)),
                "sink_metric": metric_name,
                "sink_param": float(metric_param),
                "sink_metric_key": metric_key,
                "prev_sink_count": int(len(prev_set)),
                "curr_sink_count": int(len(curr_set)),
                "birth_count": int(len(births)),
                "death_count": int(len(deaths)),
                "persistent_count": int(len(persistent)),
                "birth_count_prefix": int(len(prefix_births)),
                "birth_count_current_block": int(len(current_births)),
                "birth_count_suffix": int(len(suffix_births)),
                "death_count_prefix": int(len(prefix_deaths)),
                "death_count_current_block": int(len([p for p in deaths if get_region(p, block_start, block_end) == "current_block"])),
                "death_count_suffix": int(len([p for p in deaths if get_region(p, block_start, block_end) == "suffix"])),
                "has_any_birth": bool(len(births) > 0),
                "has_current_block_birth": bool(len(current_births) > 0),
                "has_prefix_birth": bool(len(prefix_births) > 0),
                "has_suffix_birth": bool(len(suffix_births) > 0),
                "has_prefix_death": bool(len(prefix_deaths) > 0),
                "has_current_block_death": bool(len([p for p in deaths if get_region(p, block_start, block_end) == "current_block"]) > 0),
                "has_prefix_transfer": bool(len(prefix_births) > 0 and len(prefix_deaths) > 0),
                "has_cross_region_transfer_cur_from_prefix": bool(len(current_births) > 0 and len(prefix_deaths) > 0),
                "has_cross_region_transfer_prefix_from_cur": bool(len(prefix_births) > 0 and len([p for p in deaths if get_region(p, block_start, block_end) == "current_block"]) > 0),
                "kl_curr_vs_prev_prefix": float(kl_fwd),
                "kl_prev_vs_curr_prefix": float(kl_rev),
                "jsd_prefix": float(jsd),
                "newly_unmasked_punctuation_ratio": float(punct_ratio),
                "newly_unmasked_has_punctuation": bool(punct_ratio > 0.0),
                "newly_unmasked_count": int(len(newly_unmasked_abs)),
                "birth_matches_newly_unmasked_count": int(sum(pos in newly_unmasked_set for pos in births)),
            }
            metric_summary_rows.append(metric_summary)
            step_record["layer_metrics"].append(metric_summary)

            # Precompute step/layer-level co-occurrence flags used by H1/H2/H3 analysis.
            # These live on the metric_summary row, but we also stamp them onto
            # each event row so that event_rows can be analyzed standalone.
            step_has_prefix_birth = bool(len(prefix_births) > 0)
            step_has_current_birth = bool(len(current_births) > 0)
            step_has_suffix_birth = bool(len(suffix_births) > 0)
            step_has_prefix_death = bool(len(prefix_deaths) > 0)
            current_deaths_list = [p for p in deaths if get_region(p, block_start, block_end) == "current_block"]
            suffix_deaths_list = [p for p in deaths if get_region(p, block_start, block_end) == "suffix"]
            step_has_current_death = bool(len(current_deaths_list) > 0)
            step_has_suffix_death = bool(len(suffix_deaths_list) > 0)

            for token_abs_pos in births:
                event_row = build_event_row(
                    sample_id=sample_id,
                    global_step=pending["global_step"],
                    block_idx=pending["block_idx"],
                    sub_step=pending["sub_step"],
                    layer_idx=layer_idx,
                    metric_name=metric_name,
                    metric_param=metric_param,
                    metric_key=metric_key,
                    event_type="birth",
                    token_abs_pos=token_abs_pos,
                    prompt_len=prompt_len,
                    block_start=block_start,
                    block_end=block_end,
                    prev_ids=prev_ids,
                    curr_ids=curr_ids,
                    newly_unmasked_abs=newly_unmasked_abs,
                    tokenizer=tokenizer,
                    mask_id=mask_id,
                )
                event_row["newly_unmasked_punctuation_ratio"] = float(punct_ratio)
                event_row["newly_unmasked_tokens"] = list(new_token_texts)
                event_row["newly_unmasked_has_punctuation"] = bool(punct_ratio > 0.0)
                event_row["step_has_prefix_birth"] = step_has_prefix_birth
                event_row["step_has_current_block_birth"] = step_has_current_birth
                event_row["step_has_suffix_birth"] = step_has_suffix_birth
                event_row["step_has_prefix_death"] = step_has_prefix_death
                event_row["step_has_current_block_death"] = step_has_current_death
                event_row["step_has_suffix_death"] = step_has_suffix_death
                # The classification below is what enables the "current birth came
                # from prefix death" vs "de novo current birth" split directly at
                # aggregation time without re-indexing events by (sample, layer, step).
                if event_row["region"] == "current_block":
                    if step_has_prefix_death and not step_has_current_death:
                        event_row["birth_transfer_class"] = "transfer_from_prefix"
                    elif not step_has_prefix_death and not step_has_current_death:
                        event_row["birth_transfer_class"] = "de_novo"
                    else:
                        event_row["birth_transfer_class"] = "mixed"
                elif event_row["region"] == "prefix":
                    if step_has_prefix_death and not step_has_current_death:
                        event_row["birth_transfer_class"] = "prefix_internal_transfer"
                    elif step_has_current_death and not step_has_prefix_death:
                        event_row["birth_transfer_class"] = "transfer_from_current"
                    elif not step_has_prefix_death and not step_has_current_death:
                        event_row["birth_transfer_class"] = "de_novo"
                    else:
                        event_row["birth_transfer_class"] = "mixed"
                else:
                    event_row["birth_transfer_class"] = "suffix_birth"
                step_events.append(event_row)

            for token_abs_pos in deaths:
                event_row = build_event_row(
                    sample_id=sample_id,
                    global_step=pending["global_step"],
                    block_idx=pending["block_idx"],
                    sub_step=pending["sub_step"],
                    layer_idx=layer_idx,
                    metric_name=metric_name,
                    metric_param=metric_param,
                    metric_key=metric_key,
                    event_type="death",
                    token_abs_pos=token_abs_pos,
                    prompt_len=prompt_len,
                    block_start=block_start,
                    block_end=block_end,
                    prev_ids=prev_ids,
                    curr_ids=curr_ids,
                    newly_unmasked_abs=newly_unmasked_abs,
                    tokenizer=tokenizer,
                    mask_id=mask_id,
                )
                event_row["newly_unmasked_punctuation_ratio"] = float(punct_ratio)
                event_row["newly_unmasked_tokens"] = list(new_token_texts)
                event_row["newly_unmasked_has_punctuation"] = bool(punct_ratio > 0.0)
                event_row["step_has_prefix_birth"] = step_has_prefix_birth
                event_row["step_has_current_block_birth"] = step_has_current_birth
                event_row["step_has_suffix_birth"] = step_has_suffix_birth
                event_row["step_has_prefix_death"] = step_has_prefix_death
                event_row["step_has_current_block_death"] = step_has_current_death
                event_row["step_has_suffix_death"] = step_has_suffix_death
                step_events.append(event_row)

    return step_record, step_events, metric_summary_rows


def annotate_birth_lifetimes(
    event_rows: Sequence[Dict[str, Any]],
    max_global_step: Optional[int] = None,
) -> None:
    """Stamp `lifetime_steps` and `is_persistent` on each birth event in-place.

    Lifetime is the number of global_steps from this birth event to the next
    matching death event at the same (sample_id, sink_metric_key, layer,
    token_abs_pos). A birth whose next death never occurs within the event
    list is treated as "still alive" and assigned lifetime = None with
    `is_persistent = True`.

    `is_persistent` is True when lifetime is None OR lifetime >= 2. This
    is the pilot-derived filter that excludes 1-step flickers.
    """
    trajectories: Dict[Tuple[int, str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        key = (
            int(row["sample_id"]),
            str(row["sink_metric_key"]),
            int(row["layer"]),
            int(row["token_abs_pos"]),
        )
        trajectories[key].append(row)

    for traj in trajectories.values():
        traj.sort(key=lambda r: (int(r["global_step"]), 0 if r["event_type"] == "death" else 1))
        last_birth: Optional[Dict[str, Any]] = None
        for row in traj:
            if row["event_type"] == "birth":
                # An earlier unmatched birth becomes "still alive at end of
                # this segment" -- we assume lifetime=None and persistent=True.
                # This would only happen in malformed trajectories (two
                # consecutive births without a death), so we guard it for
                # safety. It should not occur in practice.
                if last_birth is not None:
                    last_birth["lifetime_steps"] = None
                    last_birth["is_persistent"] = True
                last_birth = row
            elif row["event_type"] == "death" and last_birth is not None:
                lifetime = int(row["global_step"]) - int(last_birth["global_step"])
                last_birth["lifetime_steps"] = int(lifetime)
                last_birth["is_persistent"] = bool(lifetime >= 2)
                last_birth = None
        if last_birth is not None:
            last_birth["lifetime_steps"] = None
            last_birth["is_persistent"] = True

    for row in event_rows:
        if row["event_type"] == "birth":
            if "lifetime_steps" not in row:
                row["lifetime_steps"] = None
                row["is_persistent"] = True
        else:
            row.setdefault("lifetime_steps", None)
            row.setdefault("is_persistent", False)


def _kl_jsd_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": int(len(rows)),
        "kl_curr_vs_prev_mean": safe_mean([r["kl_curr_vs_prev_prefix"] for r in rows]),
        "kl_prev_vs_curr_mean": safe_mean([r["kl_prev_vs_curr_prefix"] for r in rows]),
        "jsd_mean": safe_mean([r["jsd_prefix"] for r in rows]),
        "kl_curr_vs_prev_std": safe_std([r["kl_curr_vs_prev_prefix"] for r in rows]),
        "jsd_std": safe_std([r["jsd_prefix"] for r in rows]),
    }


def _region_counts(evs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    c = Counter(r["region"] for r in evs)
    total = sum(c.values())
    return {
        "counts": dict(c),
        "ratios": {k: (float(v) / total) if total > 0 else 0.0 for k, v in c.items()},
        "total": int(total),
    }


def _category_counts(evs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(r["token_category"] for r in evs)
    return dict(c)


def _lifetime_histogram(evs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {"flicker_1": 0, "2": 0, "3": 0, "4_10": 0, "11_plus": 0, "still_alive": 0}
    for r in evs:
        lt = r.get("lifetime_steps")
        if lt is None:
            buckets["still_alive"] += 1
        elif lt == 1:
            buckets["flicker_1"] += 1
        elif lt == 2:
            buckets["2"] += 1
        elif lt == 3:
            buckets["3"] += 1
        elif lt <= 10:
            buckets["4_10"] += 1
        else:
            buckets["11_plus"] += 1
    total = sum(buckets.values())
    return {
        "counts": buckets,
        "ratios": {k: (float(v) / total) if total > 0 else 0.0 for k, v in buckets.items()},
        "total": int(total),
    }


def _per_sample_region_ratios(
    evs: Sequence[Dict[str, Any]],
    regions: Sequence[str] = ("prefix", "current_block", "suffix"),
) -> Dict[str, Any]:
    """Per-sample normalized region ratios to surface cross-sample variance.

    Rather than pooling all events and computing a single ratio, we compute
    each sample's ratio separately and then report the mean and std across
    samples. This protects against a single long sample dominating the
    aggregate, which was a concern raised earlier in scope discussion.
    """
    per_sample: Dict[int, Counter] = defaultdict(Counter)
    for r in evs:
        per_sample[int(r["sample_id"])][r["region"]] += 1

    ratios_by_region: Dict[str, List[float]] = {reg: [] for reg in regions}
    for sid, ctr in per_sample.items():
        total = sum(ctr.values())
        if total <= 0:
            continue
        for reg in regions:
            ratios_by_region[reg].append(float(ctr.get(reg, 0)) / total)

    return {
        "num_samples": int(len(per_sample)),
        "mean": {reg: safe_mean(vals) for reg, vals in ratios_by_region.items()},
        "std": {reg: safe_std(vals) for reg, vals in ratios_by_region.items()},
    }


def _h3_summary(current_birth_evs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute H3 aggregates over a given set of current-block birth events.

    Splits by `is_masked_now` so that "birth on an already unmasked real
    token" (which is what we care about for the punctuation trigger
    hypothesis) is separable from "birth on a MASK token" (dLLM temporary
    sink phenomenon).
    """
    by_mask_state = {
        "all": list(current_birth_evs),
        "on_unmasked_token": [r for r in current_birth_evs if not r.get("is_masked_now", False)],
        "on_mask_token": [r for r in current_birth_evs if r.get("is_masked_now", False)],
    }
    out: Dict[str, Any] = {}
    for label, evs in by_mask_state.items():
        n = len(evs)
        match_rate = safe_mean([1.0 if r["birth_matches_newly_unmasked"] else 0.0 for r in evs]) if evs else None
        punct_ratio_newly = safe_mean([r["newly_unmasked_punctuation_ratio"] for r in evs]) if evs else None
        cat_counts = _category_counts(evs)
        punct_count = cat_counts.get("punctuation", 0)
        out[label] = {
            "count": int(n),
            "birth_matches_newly_unmasked_rate": match_rate,
            "newly_unmasked_punctuation_ratio_mean": punct_ratio_newly,
            "birth_on_punctuation_token_ratio": (float(punct_count) / n) if n > 0 else None,
            "token_category_counts": cat_counts,
        }
    return out


def summarize_metric_results(
    metric_summary_rows: Sequence[Dict[str, Any]],
    event_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_metric: Dict[str, Dict[str, Any]] = {}
    metrics = sorted({row["sink_metric_key"] for row in metric_summary_rows})
    for metric_key in metrics:
        rows = [row for row in metric_summary_rows if row["sink_metric_key"] == metric_key]
        births = [row for row in event_rows if row["sink_metric_key"] == metric_key and row["event_type"] == "birth"]
        deaths = [row for row in event_rows if row["sink_metric_key"] == metric_key and row["event_type"] == "death"]
        region_counter = Counter(row["region"] for row in births)
        total_births = sum(region_counter.values())
        current_birth_rows = [row for row in rows if row["has_current_block_birth"]]
        any_birth_rows = [row for row in rows if row["has_any_birth"]]
        no_birth_rows = [row for row in rows if not row["has_any_birth"]]
        prefix_transfer_rows = [row for row in rows if row["has_prefix_transfer"]]
        prefix_birth_rows = [row for row in rows if row["has_prefix_birth"]]

        h3_birth_events = [
            row for row in births
            if row["region"] == "current_block"
        ]
        h3_category_counter = Counter(row["token_category"] for row in h3_birth_events)
        birth_match_rate = safe_mean([
            1.0 if row["birth_matches_newly_unmasked"] else 0.0 for row in h3_birth_events
        ])

        # ------------------------------------------------------------------
        # Extended aggregates: persistent filter, layer groups, transfer
        # classes, 2x2 (punct x cb_birth) splits, block-idx break-down,
        # lifetime histogram. Each section below is self-contained so that
        # tables for the paper can be built directly from summary.json.
        # ------------------------------------------------------------------

        persistent_births = [r for r in births if r.get("is_persistent", True)]
        flicker_births = [r for r in births if not r.get("is_persistent", True)]
        persistent_current_births = [r for r in persistent_births if r["region"] == "current_block"]

        lifetime_by_region: Dict[str, Dict[str, Any]] = {}
        for region in ("prefix", "current_block", "suffix"):
            lifetime_by_region[region] = _lifetime_histogram(
                [r for r in births if r["region"] == region]
            )

        layer_groups = sorted({r["layer_group"] for r in rows})
        layer_group_summary: Dict[str, Any] = {}
        for lg in layer_groups:
            lg_rows = [r for r in rows if r["layer_group"] == lg]
            lg_births_all = [r for r in births if r["layer_group"] == lg]
            lg_births_persist = [r for r in persistent_births if r["layer_group"] == lg]
            lg_region = _region_counts(lg_births_all)
            lg_region_persist = _region_counts(lg_births_persist)
            lg_cb_births_persist = [
                r for r in lg_births_persist if r["region"] == "current_block"
            ]
            lg_any_birth_rows = [r for r in lg_rows if r["has_any_birth"]]
            lg_no_birth_rows = [r for r in lg_rows if not r["has_any_birth"]]
            lg_cb_birth_rows = [r for r in lg_rows if r["has_current_block_birth"]]
            lg_pfx_birth_rows = [r for r in lg_rows if r["has_prefix_birth"]]

            layer_group_summary[lg] = {
                "num_layer_steps": int(len(lg_rows)),
                "H1_raw": lg_region,
                "H1_persistent": lg_region_persist,
                "H3_all": _h3_summary([r for r in lg_births_all if r["region"] == "current_block"]),
                "H3_persistent": _h3_summary(lg_cb_births_persist),
                "H4_any_birth": _kl_jsd_stats(lg_any_birth_rows),
                "H4_no_birth": _kl_jsd_stats(lg_no_birth_rows),
                "H4_current_block_birth": _kl_jsd_stats(lg_cb_birth_rows),
                "H4_prefix_birth": _kl_jsd_stats(lg_pfx_birth_rows),
            }

        # Transfer-class breakdown (birth_transfer_class field on births)
        transfer_class_counter = Counter(
            r.get("birth_transfer_class", "unknown") for r in births
        )
        transfer_class_counter_persistent = Counter(
            r.get("birth_transfer_class", "unknown") for r in persistent_births
        )
        # For each current-block transfer class, compute H3 separately so the
        # "transfer_from_prefix" vs "de_novo current" comparison can be read
        # directly from summary.json.
        transfer_class_h3: Dict[str, Any] = {}
        for cls in ("transfer_from_prefix", "de_novo", "mixed"):
            cls_evs = [
                r for r in persistent_births
                if r["region"] == "current_block" and r.get("birth_transfer_class") == cls
            ]
            transfer_class_h3[cls] = _h3_summary(cls_evs)

        # 2x2 split: (punctuation unmask yes/no) x (current-block birth yes/no)
        # over step-layer rows. This identifies the independent contribution
        # of each trigger to prefix attention KL divergence.
        split_2x2: Dict[str, Any] = {}
        for punct in (True, False):
            for cb in (True, False):
                key = f"punct_unmask={punct}__cb_birth={cb}"
                filt = [
                    r for r in rows
                    if bool(r["newly_unmasked_has_punctuation"]) == punct
                    and bool(r["has_current_block_birth"]) == cb
                ]
                split_2x2[key] = _kl_jsd_stats(filt)

        # Block-idx breakdown (H1 region ratios, H3 cb-birth analysis)
        block_idxs = sorted({r["block_idx"] for r in rows})
        block_summary: Dict[str, Any] = {}
        for bidx in block_idxs:
            b_rows = [r for r in rows if r["block_idx"] == bidx]
            b_births = [r for r in births if r["block_idx"] == bidx]
            b_births_persist = [r for r in persistent_births if r["block_idx"] == bidx]
            b_cb_births_persist = [r for r in b_births_persist if r["region"] == "current_block"]
            b_any_rows = [r for r in b_rows if r["has_any_birth"]]
            b_no_rows = [r for r in b_rows if not r["has_any_birth"]]
            block_summary[str(int(bidx))] = {
                "num_layer_steps": int(len(b_rows)),
                "H1_raw": _region_counts(b_births),
                "H1_persistent": _region_counts(b_births_persist),
                "H3_persistent": _h3_summary(b_cb_births_persist),
                "H4_any_birth": _kl_jsd_stats(b_any_rows),
                "H4_no_birth": _kl_jsd_stats(b_no_rows),
            }

        # Cross-region co-occurrence rates on step-layer rows
        cross_region_rates = {
            "cur_birth_with_prefix_death_rate_over_cur_birth_rows": (
                safe_mean([
                    1.0 if r["has_cross_region_transfer_cur_from_prefix"] else 0.0
                    for r in current_birth_rows
                ])
            ),
            "prefix_birth_with_current_death_rate_over_prefix_birth_rows": (
                safe_mean([
                    1.0 if r["has_cross_region_transfer_prefix_from_cur"] else 0.0
                    for r in prefix_birth_rows
                ])
            ),
            "prefix_internal_transfer_rate_over_prefix_birth_rows": (
                safe_mean([
                    1.0 if r["has_prefix_transfer"] else 0.0
                    for r in prefix_birth_rows
                ])
            ),
        }

        by_metric[metric_key] = {
            "num_layer_steps": int(len(rows)),
            "num_birth_events": int(total_births),
            "num_death_events": int(len(deaths)),
            "H1_birth_region_counts": dict(region_counter),
            "H1_birth_region_ratios": {
                k: (float(v) / total_births if total_births > 0 else 0.0)
                for k, v in region_counter.items()
            },
            "H1_per_sample_region_ratios": _per_sample_region_ratios(births),
            "H2_prefix_transfer_count": int(len(prefix_transfer_rows)),
            "H2_prefix_transfer_rate": (
                float(len(prefix_transfer_rows) / len(rows)) if rows else 0.0
            ),
            "H2_prefix_birth_rate": (
                float(len(prefix_birth_rows) / len(rows)) if rows else 0.0
            ),
            "H3_current_birth_step_count": int(len(current_birth_rows)),
            "H3_current_birth_newly_unmasked_punctuation_ratio_mean": safe_mean(
                [row["newly_unmasked_punctuation_ratio"] for row in current_birth_rows]
            ),
            "H3_current_birth_match_rate": birth_match_rate,
            "H3_current_birth_token_category_counts": dict(h3_category_counter),
            "H4_any_birth": _kl_jsd_stats(any_birth_rows),
            "H4_no_birth": _kl_jsd_stats(no_birth_rows),
            "H4_current_block_birth": _kl_jsd_stats(current_birth_rows),
            "H4_prefix_birth": _kl_jsd_stats(prefix_birth_rows),
            # ---- extended keys ----
            "extended": {
                "persistent_filter": {
                    "num_births_raw": int(len(births)),
                    "num_births_persistent": int(len(persistent_births)),
                    "num_births_flicker": int(len(flicker_births)),
                    "persistent_ratio": (
                        float(len(persistent_births)) / len(births) if births else 0.0
                    ),
                    "H1_persistent": _region_counts(persistent_births),
                    "H1_flicker": _region_counts(flicker_births),
                    "H1_persistent_per_sample": _per_sample_region_ratios(persistent_births),
                    "H3_current_persistent": _h3_summary(persistent_current_births),
                    "lifetime_histogram_by_region": lifetime_by_region,
                },
                "transfer_class": {
                    "counts_raw": dict(transfer_class_counter),
                    "counts_persistent": dict(transfer_class_counter_persistent),
                    "H3_current_by_class_persistent": transfer_class_h3,
                },
                "cross_region_rates": cross_region_rates,
                "layer_group": layer_group_summary,
                "split_2x2_punct_x_cb_birth": split_2x2,
                "by_block_idx": block_summary,
            },
        }
    return by_metric


def summarize_head_overlap_rows(head_overlap_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_rows": int(len(head_overlap_rows)),
        "pairwise_jaccard_mean": safe_mean(
            [row["pairwise_jaccard_mean"] for row in head_overlap_rows if row["pairwise_jaccard_mean"] is not None]
        ),
        "pairwise_jaccard_std": safe_std(
            [row["pairwise_jaccard_mean"] for row in head_overlap_rows if row["pairwise_jaccard_mean"] is not None]
        ),
        "avg_vs_majority_jaccard_mean": safe_mean(
            [row["avg_vs_majority_jaccard"] for row in head_overlap_rows]
        ),
        "union_size_mean": safe_mean([float(row["union_size"]) for row in head_overlap_rows]),
        "intersection_size_mean": safe_mean([float(row["intersection_size"]) for row in head_overlap_rows]),
    }


def summarize_robustness(metric_summary_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[int, int, int], Dict[str, bool]] = defaultdict(dict)
    for row in metric_summary_rows:
        key = (int(row["sample_id"]), int(row["global_step"]), int(row["layer"]))
        grouped[key][row["sink_metric_key"]] = bool(row["has_current_block_birth"])

    metric_keys = sorted({row["sink_metric_key"] for row in metric_summary_rows})
    pairwise = []
    all_agree = 0
    any_birth = 0
    for status in grouped.values():
        vals = [status.get(metric_key, False) for metric_key in metric_keys]
        if any(vals):
            any_birth += 1
        if vals and all(v == vals[0] for v in vals):
            all_agree += 1
        for i in range(len(metric_keys)):
            for j in range(i + 1, len(metric_keys)):
                a = status.get(metric_keys[i], False)
                b = status.get(metric_keys[j], False)
                pairwise.append({
                    "metric_a": metric_keys[i],
                    "metric_b": metric_keys[j],
                    "agree": bool(a == b),
                    "both_true": bool(a and b),
                })

    pairwise_summary: Dict[str, Dict[str, Any]] = {}
    grouped_pairwise: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in pairwise:
        grouped_pairwise[(row["metric_a"], row["metric_b"])].append(row)
    for (metric_a, metric_b), rows in grouped_pairwise.items():
        pairwise_summary[f"{metric_a}__vs__{metric_b}"] = {
            "agree_rate": safe_mean([1.0 if row["agree"] else 0.0 for row in rows]),
            "both_true_rate": safe_mean([1.0 if row["both_true"] else 0.0 for row in rows]),
            "count": int(len(rows)),
        }

    return {
        "num_step_layer_groups": int(len(grouped)),
        "any_birth_group_rate": (float(any_birth / len(grouped)) if grouped else 0.0),
        "all_metrics_agree_rate": (float(all_agree / len(grouped)) if grouped else 0.0),
        "pairwise": pairwise_summary,
    }


DRIFT_METRIC_COLUMNS = [
    "k_rel_l2_mean",
    "v_rel_l2_mean",
    "attn_weighted_v_l2_mean",
    "v_only_context_rel_l2_mean",
    "full_context_rel_l2_mean",
    "norm_ratio_mean",
    "sink_fraction_diff_mean",
    "k_cos_mean",
    "v_cos_mean",
    "v_only_context_cos_mean",
    "full_context_cos_mean",
]


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(pct)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _metric_distribution(rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    vals = [
        float(r[metric]) for r in rows
        if metric in r and r[metric] is not None and math.isfinite(float(r[metric]))
    ]
    return {
        "count": int(len(vals)),
        "total_n": int(len(rows)),
        "effective_n": int(len(vals)),
        "mean": safe_mean(vals),
        "std": safe_std(vals),
        "p50": _percentile(vals, 0.50),
        "p90": _percentile(vals, 0.90),
        "p95": _percentile(vals, 0.95),
        "p99": _percentile(vals, 0.99),
    }


def _pearsonr_from_rows(rows: Sequence[Dict[str, Any]], x_key: str, y_key: str) -> Dict[str, Any]:
    pairs = [
        (float(r[x_key]), float(r[y_key]))
        for r in rows
        if x_key in r and y_key in r
        and r[x_key] is not None and r[y_key] is not None
        and math.isfinite(float(r[x_key])) and math.isfinite(float(r[y_key]))
    ]
    if len(pairs) < 2:
        return {"r": None, "effective_n": int(len(pairs))}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = safe_mean(xs)
    my = safe_mean(ys)
    assert mx is not None and my is not None
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return {"r": None, "effective_n": int(len(pairs))}
    return {"r": float(num / (den_x * den_y)), "effective_n": int(len(pairs))}


def _saturation_group_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sink_dist = _metric_distribution(rows, "full_context_rel_l2_mean_sink")
    non_sink_dist = _metric_distribution(rows, "full_context_rel_l2_mean_non_sink")
    ratio_vals = []
    for row in rows:
        a = row.get("full_context_rel_l2_mean_sink")
        b = row.get("full_context_rel_l2_mean_non_sink")
        if a is None or b is None:
            continue
        a_f = float(a)
        b_f = float(b)
        if math.isfinite(a_f) and math.isfinite(b_f) and abs(b_f) > 1e-12:
            ratio_vals.append(a_f / b_f)
    return {
        "sink_drift": sink_dist,
        "non_sink_drift": non_sink_dist,
        "sink_to_non_sink_drift_ratio": {
            "count": int(len(ratio_vals)),
            "total_n": int(len(rows)),
            "effective_n": int(len(ratio_vals)),
            "mean": safe_mean(ratio_vals),
            "std": safe_std(ratio_vals),
            "p50": _percentile(ratio_vals, 0.50),
            "p90": _percentile(ratio_vals, 0.90),
            "p95": _percentile(ratio_vals, 0.95),
            "p99": _percentile(ratio_vals, 0.99),
        },
        "fraction_sink_attn": _metric_distribution(rows, "fraction_sink_attn"),
        "sink_attn_mass_diff_mean": _metric_distribution(rows, "sink_attn_mass_diff_mean"),
        "norm_ratio_mean": _metric_distribution(rows, "norm_ratio_mean"),
        "sink_fraction_diff_mean": _metric_distribution(rows, "sink_fraction_diff_mean"),
        "sink_term_contribution": _metric_distribution(rows, "sink_term_contribution"),
        "non_sink_term_contribution": _metric_distribution(rows, "non_sink_term_contribution"),
        "attn_weighted_v_sink_share": _metric_distribution(rows, "attn_weighted_v_sink_share"),
    }


def summarize_saturation_decomposition(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_lg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_sub: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_lg[str(row.get("layer_group", "unknown"))].append(row)
        by_sub[str(int(row.get("sub_step_within_block", -1)))].append(row)
    return {
        "by_layer_group": {
            key: _saturation_group_summary(group_rows)
            for key, group_rows in sorted(by_lg.items())
        },
        "by_sub_step_within_block": {
            key: _saturation_group_summary(group_rows)
            for key, group_rows in sorted(by_sub.items(), key=lambda kv: int(kv[0]))
        },
        "correlation": {
            key: _pearsonr_from_rows(group_rows, "fraction_sink_attn", "full_context_rel_l2_mean")
            for key, group_rows in sorted(by_lg.items())
        },
    }


def _summarize_groups(
    rows: Sequence[Dict[str, Any]],
    group_fn,
    metrics: Sequence[str] = DRIFT_METRIC_COLUMNS,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(group_fn(row))].append(row)
    return {
        key: {
            metric: _metric_distribution(group_rows, metric)
            for metric in metrics
        }
        for key, group_rows in sorted(grouped.items())
    }


def drift_event_class(row: Dict[str, Any]) -> str:
    has_cb = bool(row.get("has_current_block_birth", False))
    has_pfx = bool(row.get("has_prefix_birth", False))
    has_suffix = bool(row.get("has_suffix_birth", False))
    has_prefix_transfer = bool(row.get("has_prefix_transfer", False))
    if has_cb and has_pfx:
        return "both"
    if has_cb:
        return "cb_birth_only"
    if has_prefix_transfer:
        return "prefix_internal_transfer"
    if has_pfx:
        return "pfx_birth_only"
    if has_suffix:
        return "suffix_only"
    return "no_event"


def annotate_drift_rows_with_events(
    drift_rows: Sequence[Dict[str, Any]],
    metric_summary_rows: Sequence[Dict[str, Any]],
    event_rows: Sequence[Dict[str, Any]],
    default_metric_key: str,
) -> None:
    summary_by_key: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for row in metric_summary_rows:
        summary_by_key[(int(row["global_step"]), int(row["layer"]), str(row["sink_metric_key"]))] = row

    persistent_birth_steps: Set[Tuple[int, int, str]] = set()
    persistent_cb_birth_steps: Set[Tuple[int, int, str]] = set()
    persistent_birth_class: Dict[Tuple[int, int, str], str] = {}
    for ev in event_rows:
        if (
            ev.get("event_type") == "birth"
            and bool(ev.get("is_persistent", True))
        ):
            key = (int(ev["global_step"]), int(ev["layer"]), str(ev["sink_metric_key"]))
            persistent_birth_steps.add(key)
            persistent_birth_class.setdefault(key, str(ev.get("birth_transfer_class", "")))
            if ev.get("region") == "current_block":
                persistent_cb_birth_steps.add(key)

    for row in drift_rows:
        event_source = row.get("event_source_global_step")
        layer = int(row["layer"])
        metric_key = str(row.get("sink_metric_key", default_metric_key))
        metric_row = None
        if event_source is not None:
            metric_row = summary_by_key.get((int(event_source), layer, metric_key))

        if metric_row is None:
            defaults = {
                "has_any_birth": False,
                "has_current_block_birth": False,
                "has_prefix_birth": False,
                "has_suffix_birth": False,
                "has_prefix_transfer": False,
                "newly_unmasked_punctuation_ratio": 0.0,
                "newly_unmasked_has_punctuation": False,
                "newly_unmasked_count": 0,
                "birth_transfer_class": "",
            }
            row.update(defaults)
        else:
            row.update({
                "has_any_birth": bool(metric_row.get("has_any_birth", False)),
                "has_current_block_birth": bool(metric_row.get("has_current_block_birth", False)),
                "has_prefix_birth": bool(metric_row.get("has_prefix_birth", False)),
                "has_suffix_birth": bool(metric_row.get("has_suffix_birth", False)),
                "has_prefix_transfer": bool(metric_row.get("has_prefix_transfer", False)),
                "newly_unmasked_punctuation_ratio": float(metric_row.get("newly_unmasked_punctuation_ratio", 0.0)),
                "newly_unmasked_has_punctuation": bool(metric_row.get("newly_unmasked_has_punctuation", False)),
                "newly_unmasked_count": int(metric_row.get("newly_unmasked_count", 0)),
                "birth_transfer_class": "",
            })

        key = (int(event_source), layer, metric_key) if event_source is not None else None
        row["has_persistent_birth"] = bool(key in persistent_birth_steps if key is not None else False)
        row["has_persistent_current_block_birth"] = bool(key in persistent_cb_birth_steps if key is not None else False)
        if key is not None and key in persistent_birth_class:
            row["birth_transfer_class"] = persistent_birth_class[key]
        row["event_class"] = drift_event_class(row)
        for lag in (0, 1, 2):
            lag_key = (int(event_source) - lag, layer, metric_key) if event_source is not None else None
            row[f"birth_lag_{lag}"] = bool(lag_key in persistent_cb_birth_steps if lag_key is not None else False)


def build_event_aligned_drift_rows(
    drift_rows: Sequence[Dict[str, Any]],
    event_rows: Sequence[Dict[str, Any]],
    default_metric_key: str,
    window: Sequence[int] = tuple(range(-2, 6)),
) -> List[Dict[str, Any]]:
    drift_by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for row in drift_rows:
        drift_by_key[(int(row["sample_id"]), int(row["global_step"]), int(row["layer"]))] = row

    aligned: List[Dict[str, Any]] = []
    for ev in event_rows:
        if not (
            ev.get("event_type") == "birth"
            and ev.get("sink_metric_key") == default_metric_key
            and ev.get("region") == "current_block"
            and bool(ev.get("is_persistent", True))
        ):
            continue
        birth_state_step = int(ev["global_step"]) + 1
        for rel in window:
            row = drift_by_key.get((int(ev["sample_id"]), birth_state_step + int(rel), int(ev["layer"])))
            if row is None:
                continue
            aligned.append({
                "sample_id": int(ev["sample_id"]),
                "birth_event_global_step": int(ev["global_step"]),
                "relative_sub_step": int(rel),
                "layer": int(ev["layer"]),
                "layer_group": ev.get("layer_group", layer_group_name(int(ev["layer"]))),
                "birth_transfer_class": ev.get("birth_transfer_class", ""),
                **{metric: row.get(metric) for metric in DRIFT_METRIC_COLUMNS},
            })
    return aligned


def summarize_drift_results(
    drift_rows: Sequence[Dict[str, Any]],
    event_aligned_rows: Sequence[Dict[str, Any]],
    default_metric_key: str,
) -> Dict[str, Any]:
    rows = [r for r in drift_rows if r.get("sink_metric_key") == default_metric_key]
    boundary_rows = [r for r in rows if int(r.get("sub_step_within_block", -1)) == 0]
    boundary_max = 0.0
    boundary_norm_ratio_max_deviation = 0.0
    boundary_sink_fraction_max_abs = 0.0
    for r in boundary_rows:
        boundary_max = max(
            boundary_max,
            abs(float(r.get("k_rel_l2_max", 0.0))),
            abs(float(r.get("v_rel_l2_max", 0.0))),
            abs(float(r.get("v_only_context_rel_l2_max", 0.0))),
            abs(float(r.get("full_context_rel_l2_max", 0.0))),
        )
        if "norm_ratio_mean" in r and r["norm_ratio_mean"] is not None:
            boundary_norm_ratio_max_deviation = max(
                boundary_norm_ratio_max_deviation,
                abs(float(r["norm_ratio_mean"]) - 1.0),
            )
        if "sink_fraction_diff_mean" in r and r["sink_fraction_diff_mean"] is not None:
            boundary_sink_fraction_max_abs = max(
                boundary_sink_fraction_max_abs,
                abs(float(r["sink_fraction_diff_mean"])),
            )

    split_2x2: Dict[str, Any] = {}
    for birth in (False, True):
        for punct in (False, True):
            key = f"persistent_cb_birth={birth}__punct_unmask={punct}"
            filt = [
                r for r in rows
                if bool(r.get("has_persistent_current_block_birth", False)) == birth
                and bool(r.get("newly_unmasked_has_punctuation", False)) == punct
            ]
            split_2x2[key] = {
                metric: _metric_distribution(filt, metric)
                for metric in DRIFT_METRIC_COLUMNS
            }

    lag_analysis = {
        f"birth_lag_{lag}": {
            "true": {
                metric: _metric_distribution([r for r in rows if bool(r.get(f"birth_lag_{lag}", False))], metric)
                for metric in DRIFT_METRIC_COLUMNS
            },
            "false": {
                metric: _metric_distribution([r for r in rows if not bool(r.get(f"birth_lag_{lag}", False))], metric)
                for metric in DRIFT_METRIC_COLUMNS
            },
        }
        for lag in (0, 1, 2)
    }

    return {
        "default_sink_metric_key": default_metric_key,
        "num_drift_rows": int(len(rows)),
        "by_event_class": _summarize_groups(rows, lambda r: r.get("event_class", "unknown")),
        "by_layer_group": _summarize_groups(rows, lambda r: r.get("layer_group", "unknown")),
        "by_block_idx": _summarize_groups(rows, lambda r: int(r.get("block_idx", -1))),
        "by_sub_step_within_block": _summarize_groups(rows, lambda r: int(r.get("sub_step_within_block", -1))),
        "sink_birth_x_punct_unmask_2x2": split_2x2,
        "lag_analysis": lag_analysis,
        "event_aligned_drift_profile": _summarize_groups(
            event_aligned_rows,
            lambda r: int(r.get("relative_sub_step", 0)),
        ),
        "saturation_decomposition": summarize_saturation_decomposition(rows),
        "boundary_zero_drift": {
            "count": int(len(boundary_rows)),
            "max_abs_drift": float(boundary_max),
            "norm_ratio_max_deviation": float(boundary_norm_ratio_max_deviation),
            "sink_fraction_diff_max_abs": float(boundary_sink_fraction_max_abs),
            "passed": bool(
                boundary_max <= 1e-5
                and boundary_norm_ratio_max_deviation <= 1e-4
                and boundary_sink_fraction_max_abs <= 1e-5
            ),
        },
    }


def round_drift_row(row: Dict[str, Any], digits: int = 6) -> Dict[str, Any]:
    rounded: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = round(value, digits) if math.isfinite(value) else value
        else:
            rounded[key] = value
    return rounded


def write_summary_drift_csv(path: str, drift_summary: Dict[str, Any]) -> None:
    fieldnames = ["section", "group", "metric", "count", "total_n", "effective_n", "mean", "std", "p50", "p90", "p95", "p99"]
    rows: List[Dict[str, Any]] = []
    for section in ("by_event_class", "by_layer_group", "by_block_idx", "by_sub_step_within_block", "event_aligned_drift_profile"):
        for group, metric_map in drift_summary.get(section, {}).items():
            for metric, stats in metric_map.items():
                rows.append({"section": section, "group": group, "metric": metric, **stats})
    for section, split_map in drift_summary.get("lag_analysis", {}).items():
        for group, metric_map in split_map.items():
            for metric, stats in metric_map.items():
                rows.append({"section": section, "group": group, "metric": metric, **stats})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl_if_exists(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _discover_cache_scaling_runs(current_out_dir: str) -> Dict[int, str]:
    candidates = {
        16: [
            "outputs/gsm8k_attention_sink_drift_16_v2",
            "outputs/gsm8k_attention_sink_drift_16",
        ],
        32: ["outputs/gsm8k_attention_sink_drift_32"],
        64: [
            "outputs/gsm8k_attention_sink_drift_64_v2",
            "outputs/gsm8k_attention_sink_drift_64",
        ],
        128: [
            current_out_dir,
            "outputs/gsm8k_attention_sink_drift_128",
        ],
    }
    runs: Dict[int, str] = {}
    for block_length, paths in candidates.items():
        valid_candidates: List[Tuple[int, int, str]] = []
        for path in paths:
            summary = _read_json_if_exists(os.path.join(path, "summary.json"))
            if summary is None:
                continue
            cfg_block = int(summary.get("config", {}).get("block_length", block_length))
            if cfg_block == block_length and os.path.exists(os.path.join(path, "drift_rows.jsonl")):
                failures = int(summary.get("num_failures", 0))
                completed = int(summary.get("num_samples_completed", 0))
                valid_candidates.append((failures, -completed, path))
        if valid_candidates:
            _, _, chosen_path = sorted(valid_candidates)[0]
            runs[block_length] = chosen_path
    return runs


def _finite_row_value(row: Dict[str, Any], metric: str) -> Optional[float]:
    if metric not in row or row[metric] is None:
        return None
    value = float(row[metric])
    return value if math.isfinite(value) else None


def _scaling_metric_stats(rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    vals = []
    for row in rows:
        value = _finite_row_value(row, metric)
        if value is not None:
            vals.append(value)
    if metric == "sink_fraction_diff_mean":
        peak = max((abs(v) for v in vals), default=None)
    elif metric == "norm_ratio_mean":
        peak = max(vals) if vals else None
        peak_abs_deviation = max((abs(v - 1.0) for v in vals), default=None)
        return {
            "count": int(len(vals)),
            "mean": safe_mean(vals),
            "peak": peak,
            "peak_abs_deviation": peak_abs_deviation,
        }
    else:
        peak = max(vals) if vals else None
    return {
        "count": int(len(vals)),
        "mean": safe_mean(vals),
        "peak": peak,
    }


def _format_optional_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    value_f = float(value)
    if not math.isfinite(value_f):
        return ""
    return f"{value_f:.{digits}g}"


def write_cache_scaling_report(
    out_dir: str,
    default_metric_key: str,
) -> None:
    runs = _discover_cache_scaling_runs(out_dir)
    if not runs:
        return

    metrics = [
        "k_rel_l2_mean",
        "v_rel_l2_mean",
        "full_context_rel_l2_mean",
        "norm_ratio_mean",
        "sink_fraction_diff_mean",
    ]
    grouped: Dict[Tuple[str, str, str], Dict[int, Dict[str, Dict[str, Any]]]] = defaultdict(dict)
    summaries: Dict[int, Dict[str, Any]] = {}
    for block_length, path in sorted(runs.items()):
        summary = _read_json_if_exists(os.path.join(path, "summary.json"))
        if summary is not None:
            summaries[block_length] = summary
        by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in _iter_jsonl_if_exists(os.path.join(path, "drift_rows.jsonl")):
            sink_metric_key = str(row.get("sink_metric_key", default_metric_key))
            if sink_metric_key != default_metric_key:
                continue
            key = (
                str(row.get("layer_group", "unknown")),
                str(row.get("event_class", "unknown")),
                sink_metric_key,
            )
            by_key[key].append(row)
        for key, rows in by_key.items():
            grouped[key][block_length] = {
                metric: _scaling_metric_stats(rows, metric)
                for metric in metrics
            }

    fieldnames = ["layer_group", "event_class", "sink_metric_key"]
    for block_length in (16, 32, 64, 128):
        fieldnames.append(f"B{block_length}_count")
        for metric in metrics:
            fieldnames.extend([
                f"B{block_length}_{metric}_mean",
                f"B{block_length}_{metric}_peak",
            ])
            if metric == "norm_ratio_mean":
                fieldnames.append(f"B{block_length}_{metric}_peak_abs_deviation")

    csv_path = os.path.join(out_dir, "cache_scaling_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped):
            layer_group, event_class, sink_metric_key = key
            out_row: Dict[str, Any] = {
                "layer_group": layer_group,
                "event_class": event_class,
                "sink_metric_key": sink_metric_key,
            }
            for block_length in (16, 32, 64, 128):
                metric_map = grouped[key].get(block_length, {})
                first_metric = metric_map.get(metrics[0], {})
                out_row[f"B{block_length}_count"] = first_metric.get("count", "")
                for metric in metrics:
                    stats = metric_map.get(metric, {})
                    out_row[f"B{block_length}_{metric}_mean"] = stats.get("mean", "")
                    out_row[f"B{block_length}_{metric}_peak"] = stats.get("peak", "")
                    if metric == "norm_ratio_mean":
                        out_row[f"B{block_length}_{metric}_peak_abs_deviation"] = stats.get("peak_abs_deviation", "")
            writer.writerow(out_row)

    report_lines = [
        "Cache Scaling Summary",
        f"default_sink_metric_key: {default_metric_key}",
        f"runs: {', '.join(f'B={b}:{p}' for b, p in sorted(runs.items()))}",
        "",
    ]
    for block_length in (16, 32, 64, 128):
        summary = summaries.get(block_length)
        if summary is None:
            report_lines.append(f"B={block_length}: missing")
            continue
        cfg = summary.get("config", {})
        report_lines.append(
            "B={}: samples_completed={} failures={} elapsed_seconds={} sample_ids={}".format(
                block_length,
                summary.get("num_samples_completed"),
                summary.get("num_failures"),
                _format_optional_float(summary.get("elapsed_seconds")),
                cfg.get("sample_ids"),
            )
        )
    report_lines.append("")

    all_rows_by_b: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for key, by_b in grouped.items():
        for block_length, metric_map in by_b.items():
            synthetic = {
                metric: metric_map.get(metric, {}).get("mean")
                for metric in metrics
            }
            all_rows_by_b[block_length].append(synthetic)

    for metric in metrics:
        report_lines.append(metric)
        for block_length in (16, 32, 64, 128):
            vals = [
                float(row[metric])
                for row in all_rows_by_b.get(block_length, [])
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            mean_value = safe_mean(vals)
            if metric == "sink_fraction_diff_mean":
                peak_value = max((abs(v) for v in vals), default=None)
            else:
                peak_value = max(vals) if vals else None
            report_lines.append(
                f"  B={block_length}: mean={_format_optional_float(mean_value)} peak={_format_optional_float(peak_value)}"
            )
        report_lines.append("")

    b64_full = [
        float(row["full_context_rel_l2_mean"])
        for row in all_rows_by_b.get(64, [])
        if row.get("full_context_rel_l2_mean") is not None and math.isfinite(float(row["full_context_rel_l2_mean"]))
    ]
    b128_full = [
        float(row["full_context_rel_l2_mean"])
        for row in all_rows_by_b.get(128, [])
        if row.get("full_context_rel_l2_mean") is not None and math.isfinite(float(row["full_context_rel_l2_mean"]))
    ]
    b64_mean = safe_mean(b64_full)
    b128_mean = safe_mean(b128_full)
    if b64_mean is not None and b128_mean is not None and b64_mean > 0.0:
        ratio = b128_mean / b64_mean
        report_lines.append(
            f"full_context_rel_l2_mean B128/B64 ratio: {_format_optional_float(ratio)}"
        )
        if ratio > 1.5:
            report_lines.append("FLAG: B=128 full_context_rel_l2_mean exceeds 1.5x B=64.")
    else:
        report_lines.append("full_context_rel_l2_mean B128/B64 ratio: unavailable")

    with open(os.path.join(out_dir, "cache_scaling_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_table(path: str, summary_by_metric: Dict[str, Dict[str, Any]]) -> None:
    fieldnames = [
        "sink_metric_key",
        "num_layer_steps",
        "num_birth_events",
        "num_births_persistent",
        "persistent_ratio",
        "birth_ratio_current_block",
        "birth_ratio_prefix",
        "birth_ratio_suffix",
        "birth_ratio_current_block_persistent",
        "birth_ratio_prefix_persistent",
        "birth_ratio_suffix_persistent",
        "prefix_transfer_rate",
        "cur_birth_with_prefix_death_rate",
        "prefix_birth_with_current_death_rate",
        "current_birth_match_rate",
        "current_birth_punct_ratio_mean",
        "current_birth_on_punct_token_ratio_persistent",
        "kl_any_birth_mean",
        "kl_no_birth_mean",
        "jsd_any_birth_mean",
        "jsd_no_birth_mean",
        "kl_cb_birth_mean",
        "kl_pfx_birth_mean",
        "kl_2x2_punct0_cb0",
        "kl_2x2_punct1_cb0",
        "kl_2x2_punct0_cb1",
        "kl_2x2_punct1_cb1",
    ]

    def _safe_get(d, *path, default=None):
        cur = d
        for k in path:
            if cur is None or k not in cur:
                return default
            cur = cur[k]
        return cur if cur is not None else default

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric_key, summary in sorted(summary_by_metric.items()):
            h1 = summary["H1_birth_region_ratios"]
            h4_any = summary["H4_any_birth"]
            h4_none = summary["H4_no_birth"]
            h4_cb = summary.get("H4_current_block_birth", {})
            h4_pfx = summary.get("H4_prefix_birth", {})
            ext = summary.get("extended", {})
            pf = ext.get("persistent_filter", {})
            pf_h1 = pf.get("H1_persistent", {}).get("ratios", {})
            pf_h3_on_unmasked = _safe_get(pf, "H3_current_persistent", "on_unmasked_token", default={})
            cr = ext.get("cross_region_rates", {})
            s2x2 = ext.get("split_2x2_punct_x_cb_birth", {})
            writer.writerow({
                "sink_metric_key": metric_key,
                "num_layer_steps": summary["num_layer_steps"],
                "num_birth_events": summary["num_birth_events"],
                "num_births_persistent": pf.get("num_births_persistent"),
                "persistent_ratio": pf.get("persistent_ratio"),
                "birth_ratio_current_block": h1.get("current_block", 0.0),
                "birth_ratio_prefix": h1.get("prefix", 0.0),
                "birth_ratio_suffix": h1.get("suffix", 0.0),
                "birth_ratio_current_block_persistent": pf_h1.get("current_block", 0.0),
                "birth_ratio_prefix_persistent": pf_h1.get("prefix", 0.0),
                "birth_ratio_suffix_persistent": pf_h1.get("suffix", 0.0),
                "prefix_transfer_rate": summary["H2_prefix_transfer_rate"],
                "cur_birth_with_prefix_death_rate": cr.get("cur_birth_with_prefix_death_rate_over_cur_birth_rows"),
                "prefix_birth_with_current_death_rate": cr.get("prefix_birth_with_current_death_rate_over_prefix_birth_rows"),
                "current_birth_match_rate": summary["H3_current_birth_match_rate"],
                "current_birth_punct_ratio_mean": summary["H3_current_birth_newly_unmasked_punctuation_ratio_mean"],
                "current_birth_on_punct_token_ratio_persistent": pf_h3_on_unmasked.get("birth_on_punctuation_token_ratio"),
                "kl_any_birth_mean": h4_any["kl_curr_vs_prev_mean"],
                "kl_no_birth_mean": h4_none["kl_curr_vs_prev_mean"],
                "jsd_any_birth_mean": h4_any["jsd_mean"],
                "jsd_no_birth_mean": h4_none["jsd_mean"],
                "kl_cb_birth_mean": h4_cb.get("kl_curr_vs_prev_mean"),
                "kl_pfx_birth_mean": h4_pfx.get("kl_curr_vs_prev_mean"),
                "kl_2x2_punct0_cb0": _safe_get(s2x2, "punct_unmask=False__cb_birth=False", "kl_curr_vs_prev_mean"),
                "kl_2x2_punct1_cb0": _safe_get(s2x2, "punct_unmask=True__cb_birth=False", "kl_curr_vs_prev_mean"),
                "kl_2x2_punct0_cb1": _safe_get(s2x2, "punct_unmask=False__cb_birth=True", "kl_curr_vs_prev_mean"),
                "kl_2x2_punct1_cb1": _safe_get(s2x2, "punct_unmask=True__cb_birth=True", "kl_curr_vs_prev_mean"),
            })


def save_full_head_attn_dump(
    out_path: str,
    sample_id: int,
    step_payloads: Sequence[Dict[str, Any]],
) -> None:
    payload = {"sample_id": int(sample_id), "steps": list(step_payloads)}
    torch.save(payload, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-ids", type=str, default="")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--limit", type=int, default=0, help="Optional cap on selected sample ids after filtering.")
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--steps-per-block", type=int, default=32)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--mean-ratio-taus", type=str, default="3,4,5")
    p.add_argument("--z-thresholds", type=str, default="3")
    p.add_argument("--topk-values", type=str, default="3")
    p.add_argument("--head-overlap-topk", type=int, default=3)
    p.add_argument("--head-debug-samples", type=int, default=5)
    p.add_argument("--save-head-debug", dest="save_head_debug", action="store_true")
    p.add_argument("--no-save-head-debug", dest="save_head_debug", action="store_false")
    p.add_argument("--save-full-head-attn", action="store_true")
    p.add_argument("--no-drift-logging", action="store_true")
    p.add_argument("--drift-sink-metric-key", type=str, default="zscore:3")
    p.add_argument("--drift-eps", type=float, default=1e-8)
    p.add_argument("--assert-boundary-zero-drift", dest="assert_boundary_zero_drift", action="store_true")
    p.add_argument("--no-assert-boundary-zero-drift", dest="assert_boundary_zero_drift", action="store_false")
    p.add_argument("--write-cache-scaling-report", dest="write_cache_scaling_report", action="store_true")
    p.add_argument("--no-write-cache-scaling-report", dest="write_cache_scaling_report", action="store_false")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--out-dir", type=str, default="outputs/gsm8k_attention_sink_drift")
    p.set_defaults(save_head_debug=True, assert_boundary_zero_drift=True, write_cache_scaling_report=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_start_time = time.time()

    if args.gen_length <= 0:
        raise ValueError("--gen-length must be positive")
    if args.block_length <= 0:
        raise ValueError("--block-length must be positive")
    if args.steps_per_block <= 0:
        raise ValueError("--steps-per-block must be positive")
    if args.gen_length % args.block_length != 0:
        raise ValueError("--gen-length must be divisible by --block-length")
    if args.num_fewshot < 0:
        raise ValueError("--num-fewshot must be >= 0")

    mean_ratio_taus = parse_float_list(args.mean_ratio_taus)
    z_thresholds = parse_float_list(args.z_thresholds)
    topk_values = parse_int_list(args.topk_values)
    if not mean_ratio_taus:
        raise ValueError("--mean-ratio-taus must not be empty")
    if not z_thresholds:
        raise ValueError("--z-thresholds must not be empty")
    if not topk_values:
        raise ValueError("--topk-values must not be empty")
    drift_metric_keys = (
        [serialize_metric_def("mean_ratio", tau) for tau in mean_ratio_taus]
        + [serialize_metric_def("zscore", z) for z in z_thresholds]
        + [serialize_metric_def("topk", k) for k in topk_values]
    )
    if args.drift_sink_metric_key not in drift_metric_keys:
        raise ValueError(
            f"--drift-sink-metric-key {args.drift_sink_metric_key!r} is not among configured sink metrics: "
            f"{drift_metric_keys}"
        )

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}")
    model = (
        LLaDAModelLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch_dtype,
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading GSM8K {args.split} split...")
    ds = load_gsm8k(args.seed, args.split)
    demo_ds = load_gsm8k(args.seed, "train") if args.num_fewshot > 0 else None

    if args.sample_ids:
        sample_ids = sorted(set(int(x.strip()) for x in args.sample_ids.split(",") if x.strip()))
    else:
        sample_ids = list(range(args.start_id, args.start_id + args.num_samples))
    valid_ids = [i for i in sample_ids if 0 <= i < len(ds)]
    if args.limit and args.limit > 0:
        valid_ids = valid_ids[: int(args.limit)]
    if not valid_ids:
        raise ValueError("No valid sample ids")

    os.makedirs(args.out_dir, exist_ok=True)

    all_event_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    all_metric_summary_rows: List[Dict[str, Any]] = []
    all_head_overlap_rows: List[Dict[str, Any]] = []
    all_sample_records: List[Dict[str, Any]] = []
    all_head_debug_rows: List[Dict[str, Any]] = []
    all_drift_rows: List[Dict[str, Any]] = []
    all_event_aligned_drift_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    debug_sample_ids = set(valid_ids[: max(int(args.head_debug_samples), 0)])

    for sample_id in tqdm(valid_ids, desc="Attention sink eval"):
        try:
            sample = ds[int(sample_id)]
            prompt_text = get_prompt_text(
                tokenizer=tokenizer,
                sample=sample,
                demo_ds=demo_ds,
                num_shots=args.num_fewshot,
                no_chat_template=args.no_chat_template,
            )
            input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)
            prompt_len = int(input_ids.shape[1])
            x = make_initial_sequence(input_ids, args.gen_length, args.mask_id, model.device)
            num_blocks = args.gen_length // args.block_length

            sample_event_rows: List[Dict[str, Any]] = []
            sample_step_rows: List[Dict[str, Any]] = []
            sample_metric_summary_rows: List[Dict[str, Any]] = []
            sample_head_overlap_rows: List[Dict[str, Any]] = []
            sample_head_debug_rows: List[Dict[str, Any]] = []
            sample_drift_rows: List[Dict[str, Any]] = []
            sample_event_aligned_drift_rows: List[Dict[str, Any]] = []
            full_head_attn_steps: List[Dict[str, Any]] = []
            drift_oracle = ProductionKVDriftOracle(
                eps=args.drift_eps,
                no_drift_logging=args.no_drift_logging,
                assert_boundary_zero=args.assert_boundary_zero_drift,
            )

            pending: Optional[Dict[str, Any]] = None
            current_global_step = 0
            total_nfe = 0

            def append_snapshot_side_outputs(step_meta: Dict[str, Any], snapshot_obj: Dict[str, Any]) -> None:
                for overlap_row in snapshot_obj["head_overlap_rows"]:
                    sample_head_overlap_rows.append({
                        "sample_id": int(sample_id),
                        "global_step": int(step_meta["global_step"]),
                        "block_idx": int(step_meta["block_idx"]),
                        "sub_step": int(step_meta["sub_step"]),
                        **overlap_row,
                    })
                if int(sample_id) in debug_sample_ids and args.save_head_debug:
                    for debug_row in snapshot_obj["head_debug_rows"]:
                        sample_head_debug_rows.append({
                            "sample_id": int(sample_id),
                            "global_step": int(step_meta["global_step"]),
                            "block_idx": int(step_meta["block_idx"]),
                            "sub_step": int(step_meta["sub_step"]),
                            **debug_row,
                        })
                if args.save_full_head_attn and int(sample_id) in debug_sample_ids:
                    full_head_attn_steps.append({
                        "global_step": int(step_meta["global_step"]),
                        "block_idx": int(step_meta["block_idx"]),
                        "sub_step": int(step_meta["sub_step"]),
                        "head_incoming": snapshot_obj["full_head_incoming"],
                    })

            def append_snapshot_drift_rows(
                snapshot_obj: Dict[str, Any],
                state_global_step: int,
                event_source_global_step: Optional[int],
                block_idx_value: int,
                sub_step_value: int,
                block_start_value: int,
                block_end_value: int,
            ) -> None:
                if args.no_drift_logging:
                    return
                for drift_row in snapshot_obj.get("drift_rows", []):
                    base_row = {
                        "sample_id": int(sample_id),
                        "global_step": int(state_global_step),
                        "event_source_global_step": (
                            int(event_source_global_step)
                            if event_source_global_step is not None else None
                        ),
                        "block_idx": int(block_idx_value),
                        "sub_step_within_block": int(sub_step_value),
                        "current_block_start_abs": int(block_start_value),
                        "current_block_end_abs": int(block_end_value),
                        "current_block_start_rel": int(block_start_value - prompt_len),
                        "current_block_end_rel": int(block_end_value - prompt_len),
                        "layer_group": layer_group_name(int(drift_row["layer"])),
                        **drift_row,
                    }
                    for metric_key in drift_metric_keys:
                        sample_drift_rows.append({
                            **base_row,
                            "sink_metric_key": str(metric_key),
                        })

            for block_idx in range(num_blocks):
                block_start = prompt_len + block_idx * args.block_length
                block_end = prompt_len + (block_idx + 1) * args.block_length
                sub_step = 0
                block_drift_initialized = False
                while True:
                    if pending is not None and (
                        int(pending["block_start"]) != block_start or int(pending["block_end"]) != block_end
                    ):
                        _, boundary_snapshot = forward_with_sink_snapshot(
                            model=model,
                            x=x,
                            prompt_len=prompt_len,
                            block_start=int(pending["block_start"]),
                            block_end=int(pending["block_end"]),
                            mean_ratio_taus=mean_ratio_taus,
                            z_thresholds=z_thresholds,
                            topk_values=topk_values,
                            head_overlap_topk=args.head_overlap_topk,
                            drift_oracle=None,
                        )
                        step_row, events, metric_rows = finalize_transition(
                            sample_id=sample_id,
                            pending=pending,
                            current_snapshot=boundary_snapshot,
                            tokenizer=tokenizer,
                            mask_id=args.mask_id,
                        )
                        sample_step_rows.append(step_row)
                        sample_event_rows.extend(events)
                        sample_metric_summary_rows.extend(metric_rows)
                        append_snapshot_side_outputs(pending, boundary_snapshot)
                        pending = None
                        if x.device.type == "cuda":
                            torch.cuda.empty_cache()

                    if not block_drift_initialized:
                        drift_oracle.reset_block()
                        block_drift_initialized = True

                    remaining = int((x[:, block_start:block_end] == args.mask_id).sum().item())
                    if remaining == 0:
                        break

                    mask_idx = (x == args.mask_id)
                    mask_idx[:, block_end:] = False

                    logits, snapshot = forward_with_sink_snapshot(
                        model=model,
                        x=x,
                        prompt_len=prompt_len,
                        block_start=block_start,
                        block_end=block_end,
                        mean_ratio_taus=mean_ratio_taus,
                        z_thresholds=z_thresholds,
                        topk_values=topk_values,
                        head_overlap_topk=args.head_overlap_topk,
                        drift_oracle=drift_oracle,
                        sub_step=sub_step,
                    )
                    total_nfe += 1
                    append_snapshot_drift_rows(
                        snapshot_obj=snapshot,
                        state_global_step=current_global_step,
                        event_source_global_step=(current_global_step - 1 if current_global_step > 0 else None),
                        block_idx_value=block_idx,
                        sub_step_value=sub_step,
                        block_start_value=block_start,
                        block_end_value=block_end,
                    )

                    if pending is not None:
                        step_row, events, metric_rows = finalize_transition(
                            sample_id=sample_id,
                            pending=pending,
                            current_snapshot=snapshot,
                            tokenizer=tokenizer,
                            mask_id=args.mask_id,
                        )
                        sample_step_rows.append(step_row)
                        sample_event_rows.extend(events)
                        sample_metric_summary_rows.extend(metric_rows)
                        append_snapshot_side_outputs(pending, snapshot)

                    logits_noisy = add_gumbel_noise(logits, temperature=args.temperature)
                    x0 = torch.argmax(logits_noisy, dim=-1)
                    probs = F.softmax(logits.to(torch.float64), dim=-1)
                    score = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    x0 = torch.where(mask_idx, x0, x)
                    neg_inf = torch.tensor(torch.finfo(score.dtype).min, device=model.device, dtype=score.dtype)
                    confidence = torch.where(mask_idx, score, neg_inf)
                    transfer_index = select_transfer_index_threshold(confidence, mask_idx, args.threshold)
                    commit_abs = torch.nonzero(transfer_index[0], as_tuple=False).view(-1)
                    newly_unmasked_abs = [int(v) for v in commit_abs.detach().cpu().tolist()]

                    prev_token_ids = [int(v) for v in x[0].detach().cpu().tolist()]
                    x[transfer_index] = x0[transfer_index]
                    curr_token_ids = [int(v) for v in x[0].detach().cpu().tolist()]
                    remaining_after = int((x[:, block_start:block_end] == args.mask_id).sum().item())

                    pending = {
                        "sample_id": int(sample_id),
                        "global_step": int(current_global_step),
                        "block_idx": int(block_idx),
                        "sub_step": int(sub_step),
                        "prompt_len": int(prompt_len),
                        "block_start": int(block_start),
                        "block_end": int(block_end),
                        "remaining_before": int(remaining),
                        "remaining_after": int(remaining_after),
                        "newly_unmasked_abs": list(newly_unmasked_abs),
                        "prev_snapshot": snapshot,
                        "prev_token_ids": prev_token_ids,
                        "curr_token_ids": curr_token_ids,
                    }
                    current_global_step += 1
                    sub_step += 1

                    if x.device.type == "cuda":
                        torch.cuda.empty_cache()

            if pending is not None:
                _, final_snapshot = forward_with_sink_snapshot(
                    model=model,
                    x=x,
                    prompt_len=prompt_len,
                    block_start=pending["block_start"],
                    block_end=pending["block_end"],
                    mean_ratio_taus=mean_ratio_taus,
                    z_thresholds=z_thresholds,
                    topk_values=topk_values,
                    head_overlap_topk=args.head_overlap_topk,
                    drift_oracle=None,
                )
                step_row, events, metric_rows = finalize_transition(
                    sample_id=sample_id,
                    pending=pending,
                    current_snapshot=final_snapshot,
                    tokenizer=tokenizer,
                    mask_id=args.mask_id,
                )
                sample_step_rows.append(step_row)
                sample_event_rows.extend(events)
                sample_metric_summary_rows.extend(metric_rows)
                append_snapshot_side_outputs(pending, final_snapshot)
                if x.device.type == "cuda":
                    torch.cuda.empty_cache()

            # Stamp lifetime_steps and is_persistent on each birth event within
            # this sample. Trajectories are tracked per (sample, metric, layer,
            # token_abs_pos) so different samples cannot interfere.
            annotate_birth_lifetimes(sample_event_rows)
            if not args.no_drift_logging:
                annotate_drift_rows_with_events(
                    drift_rows=sample_drift_rows,
                    metric_summary_rows=sample_metric_summary_rows,
                    event_rows=sample_event_rows,
                    default_metric_key=args.drift_sink_metric_key,
                )
                sample_event_aligned_drift_rows = build_event_aligned_drift_rows(
                    drift_rows=sample_drift_rows,
                    event_rows=sample_event_rows,
                    default_metric_key=args.drift_sink_metric_key,
                )

            sample_summary = summarize_metric_results(sample_metric_summary_rows, sample_event_rows)
            sample_head_overlap_summary = summarize_head_overlap_rows(sample_head_overlap_rows)
            sample_record = {
                "sample_id": int(sample_id),
                "question": sample["question"],
                "answer": sample["answer"],
                "prompt_len": int(prompt_len),
                "num_shots": int(args.num_fewshot),
                "gen_length": int(args.gen_length),
                "block_length": int(args.block_length),
                "steps_per_block": int(args.steps_per_block),
                "threshold": float(args.threshold),
                "nfe": int(total_nfe),
                "num_steps": int(len(sample_step_rows)),
                "num_events": int(len(sample_event_rows)),
                "metric_summary": sample_summary,
                "head_overlap_summary": sample_head_overlap_summary,
                "steps": sample_step_rows,
                "events": sample_event_rows,
                "drift_rows": [round_drift_row(r) for r in sample_drift_rows],
            }

            all_event_rows.extend(sample_event_rows)
            all_step_rows.extend(sample_step_rows)
            all_metric_summary_rows.extend(sample_metric_summary_rows)
            all_head_overlap_rows.extend(sample_head_overlap_rows)
            all_sample_records.append(sample_record)
            all_head_debug_rows.extend(sample_head_debug_rows)
            all_drift_rows.extend(sample_drift_rows)
            all_event_aligned_drift_rows.extend(sample_event_aligned_drift_rows)

            if args.save_full_head_attn and int(sample_id) in debug_sample_ids and full_head_attn_steps:
                dump_path = os.path.join(args.out_dir, f"full_head_attn_sample_{sample_id}.pt")
                save_full_head_attn_dump(dump_path, int(sample_id), full_head_attn_steps)

        except Exception as exc:
            import traceback
            failures.append({
                "sample_id": int(sample_id),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"[ERROR] sample_id={sample_id}: {exc}")

    summary_by_metric = summarize_metric_results(all_metric_summary_rows, all_event_rows)
    robustness = summarize_robustness(all_metric_summary_rows)
    head_overlap_summary = summarize_head_overlap_rows(all_head_overlap_rows)
    drift_summary = (
        summarize_drift_results(
            drift_rows=all_drift_rows,
            event_aligned_rows=all_event_aligned_drift_rows,
            default_metric_key=args.drift_sink_metric_key,
        )
        if not args.no_drift_logging else {}
    )

    summary = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "split": args.split,
            "seed": int(args.seed),
            "sample_ids": [int(v) for v in valid_ids],
            "num_fewshot": int(args.num_fewshot),
            "gen_length": int(args.gen_length),
            "block_length": int(args.block_length),
            "steps_per_block": int(args.steps_per_block),
            "threshold": float(args.threshold),
            "mean_ratio_taus": [float(v) for v in mean_ratio_taus],
            "z_thresholds": [float(v) for v in z_thresholds],
            "topk_values": [int(v) for v in topk_values],
            "head_overlap_topk": int(args.head_overlap_topk),
            "head_debug_samples": int(args.head_debug_samples),
            "save_head_debug": bool(args.save_head_debug),
            "save_full_head_attn": bool(args.save_full_head_attn),
            "no_drift_logging": bool(args.no_drift_logging),
            "drift_sink_metric_key": str(args.drift_sink_metric_key),
            "drift_metric_keys": list(drift_metric_keys),
            "drift_eps": float(args.drift_eps),
            "assert_boundary_zero_drift": bool(args.assert_boundary_zero_drift),
            "write_cache_scaling_report": bool(args.write_cache_scaling_report),
        },
        "num_samples_requested": int(len(sample_ids)),
        "num_samples_completed": int(len(all_sample_records)),
        "num_failures": int(len(failures)),
        "elapsed_seconds": float(time.time() - run_start_time),
        "metric_summary": summary_by_metric,
        "robustness": robustness,
        "head_overlap_summary": head_overlap_summary,
        "drift_summary": drift_summary,
        "failures": failures,
    }

    write_jsonl(os.path.join(args.out_dir, "event_rows.jsonl"), all_event_rows)
    write_jsonl(os.path.join(args.out_dir, "step_rows.jsonl"), all_step_rows)
    write_jsonl(os.path.join(args.out_dir, "drift_rows.jsonl"), (round_drift_row(r) for r in all_drift_rows))
    write_jsonl(os.path.join(args.out_dir, "event_aligned_drift_rows.jsonl"), (round_drift_row(r) for r in all_event_aligned_drift_rows))
    write_jsonl(os.path.join(args.out_dir, "per_sample_records.jsonl"), all_sample_records)
    write_jsonl(os.path.join(args.out_dir, "head_overlap_rows.jsonl"), all_head_overlap_rows)
    write_jsonl(os.path.join(args.out_dir, "head_debug.jsonl"), all_head_debug_rows)
    write_jsonl(os.path.join(args.out_dir, "failures.jsonl"), failures)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_table(os.path.join(args.out_dir, "summary_tables.csv"), summary_by_metric)
    if not args.no_drift_logging:
        write_summary_drift_csv(os.path.join(args.out_dir, "summary_drift.csv"), drift_summary)
    block_suffix = str(int(args.block_length))
    with open(os.path.join(args.out_dir, f"summary_{block_suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_jsonl(os.path.join(args.out_dir, f"drift_rows_{block_suffix}.jsonl"), (round_drift_row(r) for r in all_drift_rows))
    if not args.no_drift_logging:
        write_summary_drift_csv(os.path.join(args.out_dir, f"summary_drift_{block_suffix}.csv"), drift_summary)
    if args.write_cache_scaling_report and not args.no_drift_logging:
        write_cache_scaling_report(
            out_dir=args.out_dir,
            default_metric_key=args.drift_sink_metric_key,
        )

    print(f"Saved outputs to {args.out_dir}")
    print(f"Completed samples: {len(all_sample_records)} / {len(valid_ids)}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']:.1f}")
    if failures:
        print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
