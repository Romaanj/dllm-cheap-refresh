# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM
import json
import time
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        use_cache=False,
        threshold=None,
        factor=None,
        save_dir=None,
        show_speed=False,
        dual_cache=False,
        ragged=False,
        K_ratio=0.30,
        K_bands=None,
        K_blocks=None,
        v_weighted_topk=False,
        dpad_intersect=False,
        right_edge_floor=False,
        near_future_floor=False,
        r_curr_floor=False,
        recency_window=0,
        r_future_stride=0,
        head_policy_path=None,
        head_policy_recency_W=32,
        knockout_mask_path=None,
        sparse_attention_policy_path=None,
        sparse_per_head_config_path=None,
        scope_incremental_path=None,
        drift_refresh_threshold=None,
        use_cuda_graph=True,
        cheap_e3=False,
        cheap_K=384,
        cheap_k_schedule="stepped",
        cheap_warmup=2,
        cheap_window=4,
        cheap_use_kv_cache=True,
        cheap_include_u_t=True,
        cheap_window_only=False,
        cheap_zar_mode=False,
        cheap_zar_lookahead=8,
        cheap_zar_prompt_tail=32,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        # LLaDA-1.5 is conversational/instruct-tuned but doesn't have 'instruct' in its name
        # (huggingface tag: conversational). Detect explicitly.
        mp_lower = model_path.lower()
        self.is_instruct = ('instruct' in mp_lower) or ('llada-1.5' in mp_lower)
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache
        self.ragged = ragged
        # K_bands="0.10,0.20,0.45" → 3-band layer-aware K. Edges hardcoded for
        # LLaDA-8B 32 layers: [0,7) shallow, [7,16) mid, [16,32) deep.
        if K_bands is not None:
            # accept '/', ':' or '|' separator (lm-eval --model_args uses ',' itself)
            s = str(K_bands).replace("/", " ").replace(":", " ").replace("|", " ")
            band_ratios = [float(x) for x in s.split()]
            assert len(band_ratios) == 3, f"K_bands needs 3 values, got {band_ratios}"
            band_edges = [0, 7, 16, 32]
            per_layer_K = []
            for li in range(32):
                for bi in range(3):
                    if band_edges[bi] <= li < band_edges[bi+1]:
                        per_layer_K.append(band_ratios[bi])
                        break
            self.K_ratio = per_layer_K
            print(f"[eval_llada] K_bands enabled: shallow={band_ratios[0]} mid={band_ratios[1]} deep={band_ratios[2]} (avg={sum(per_layer_K)/32:.3f})")
        else:
            self.K_ratio = K_ratio
        # K_blocks="0.475/0.425/0.375/0.325/0.275/0.225/0.175/0.125" → per-block scalar K.
        # Number of entries must equal n_blocks (gen_length / block_length).
        # Overrides per-block K_ratio (still uses K_bands or scalar K_ratio if K_blocks is None).
        if K_blocks is not None:
            s = str(K_blocks).replace("/", " ").replace(":", " ").replace("|", " ")
            block_schedule = [float(x) for x in s.split()]
            self.K_blocks = block_schedule
            print(f"[eval_llada] K_blocks enabled: schedule={block_schedule} avg={sum(block_schedule)/len(block_schedule):.3f}")
        else:
            self.K_blocks = None
        self.dpad_intersect = bool(dpad_intersect)
        self.right_edge_floor = bool(right_edge_floor)
        self.near_future_floor = bool(near_future_floor)
        self.r_curr_floor = bool(r_curr_floor)
        self.recency_window = int(recency_window)
        self.r_future_stride = int(r_future_stride)
        self.head_policy_recency_W = int(head_policy_recency_W)
        # head_policy_path: path to (n_layers, n_heads) int8 npy of cluster IDs
        if head_policy_path is not None:
            import numpy as _np
            self.head_policy = _np.load(head_policy_path).astype(_np.int8)
            print(f"[eval_llada] head_policy loaded: {head_policy_path}  shape={self.head_policy.shape}")
            print(f"[eval_llada]   recency_W for bidir_asym_L heads = {self.head_policy_recency_W}")
        else:
            self.head_policy = None
        self.v_weighted_topk = bool(v_weighted_topk)
        if self.v_weighted_topk:
            print(f"[eval_llada] v_weighted_topk=True (rank by attn × ||V||)")
        # Knockout: zero specified (layer, head) attention head outputs via o_proj forward pre-hook
        if knockout_mask_path is not None:
            import numpy as _np
            ko_mask = _np.load(knockout_mask_path).astype(bool)
            assert ko_mask.shape == (32, 32), f"knockout mask must be (32, 32), got {ko_mask.shape}"
            n_ko = int(ko_mask.sum())
            print(f"[eval_llada] knockout_mask loaded: {knockout_mask_path} ({n_ko}/{ko_mask.size} heads will be zeroed)")
            self._install_knockout_hooks(ko_mask)
        if sparse_attention_policy_path is not None:
            from sparse_attention_prototype import install_sparse_attention
            print(f"[eval_llada] installing sparse attention from {sparse_attention_policy_path}")
            if sparse_per_head_config_path is not None:
                print(f"  + per-head W config: {sparse_per_head_config_path}")
            install_sparse_attention(self.model, self.tokenizer, sparse_attention_policy_path,
                                     mask_id=self.mask_id,
                                     per_head_config_path=sparse_per_head_config_path)
        if scope_incremental_path is not None:
            from scope_incremental_attention import install_scope_incremental
            print(f"[eval_llada] installing scope-incremental from {scope_incremental_path}")
            self._scope_cache = install_scope_incremental(
                self.model, self.tokenizer, scope_incremental_path,
                mask_id=self.mask_id,
            )
        if drift_refresh_threshold is not None:
            from drift_refresh_attention import install_drift_refresh
            print(f"[eval_llada] installing drift-refresh with threshold={drift_refresh_threshold}")
            self._drift_cache = install_drift_refresh(self.model, threshold=float(drift_refresh_threshold))
        if self.r_curr_floor:
            print(f"[eval_llada] r_curr_floor=True (add current block mask positions to floor)")
        if self.recency_window > 0:
            print(f"[eval_llada] recency_window={self.recency_window} (last-W unmask floor)")
        if self.r_future_stride > 0:
            print(f"[eval_llada] r_future_stride={self.r_future_stride} (future anchor-only, drop rest)")
        self.use_cuda_graph = use_cuda_graph

        # Cheap E3 mode (Phase 14a) — selective layer-position refresh.
        self.cheap_e3 = bool(cheap_e3) if not isinstance(cheap_e3, str) else cheap_e3.lower() in ("true", "1", "yes")
        self.cheap_K = int(cheap_K)
        self.cheap_k_schedule = str(cheap_k_schedule)
        self.cheap_warmup = int(cheap_warmup)
        # cheap_window: int OR comma/semicolon-separated list of 32 ints (per-layer).
        if isinstance(cheap_window, str) and (',' in cheap_window or ';' in cheap_window):
            sep = ',' if ',' in cheap_window else ';'
            self.cheap_window = [int(x) for x in cheap_window.split(sep)]
        else:
            self.cheap_window = int(cheap_window)
        self.cheap_use_kv_cache = bool(cheap_use_kv_cache) if not isinstance(cheap_use_kv_cache, str) else cheap_use_kv_cache.lower() in ("true", "1", "yes")
        self.cheap_include_u_t = bool(cheap_include_u_t) if not isinstance(cheap_include_u_t, str) else cheap_include_u_t.lower() in ("true", "1", "yes")
        self.cheap_window_only = bool(cheap_window_only) if not isinstance(cheap_window_only, str) else cheap_window_only.lower() in ("true", "1", "yes")
        self.cheap_zar_mode = bool(cheap_zar_mode) if not isinstance(cheap_zar_mode, str) else cheap_zar_mode.lower() in ("true", "1", "yes")
        self.cheap_zar_lookahead = int(cheap_zar_lookahead)
        self.cheap_zar_prompt_tail = int(cheap_zar_prompt_tail)
        self._cheap_capture = None
        if self.cheap_e3:
            from phase14a_cheap_e2e import AllLayerCapture
            n_layers_ = int(self.model.config.n_layers)
            self._cheap_capture = AllLayerCapture(n_layers_)
            self._cheap_capture.install(self.model)
            print(f"[eval_llada] cheap_e3 mode ENABLED: K={self.cheap_K} schedule={self.cheap_k_schedule} "
                  f"warmup={self.cheap_warmup} window={self.cheap_window} kv_cache={self.cheap_use_kv_cache} "
                  f"include_u_t={self.cheap_include_u_t} window_only={self.cheap_window_only} "
                  f"zar={self.cheap_zar_mode} zar_la={self.cheap_zar_lookahead} zar_pt={self.cheap_zar_prompt_tail} "
                  f"threshold={self.threshold}")

        if self.ragged:
            # Patch attention + pre-compute punct/special id sets for our impl
            import string as _string
            import numpy as _np
            from experiments.fastgen_head_probe.end2end_ragged_dual import (
                patch_attention as _patch_attention,
                build_punct_id_set as _bps,
                build_special_id_set as _bss,
            )
            _patch_attention(self.model)
            self.punct_arr = _np.fromiter(_bps(self.tokenizer), dtype=_np.int64)
            self.special_arr = _np.fromiter(_bss(self.tokenizer), dtype=_np.int64)
            assert self.batch_size == 1, "ragged path currently supports batch_size=1 only"

    def _install_knockout_hooks(self, ko_mask):
        """For each (layer, head) marked True in ko_mask, register a forward
        pre-hook on that layer's o_proj that zeroes the corresponding
        head-dim slice of the input.

        Input to o_proj is (B, T, n_heads * head_dim) — flattened heads.
        For head h: channels [h*head_dim, (h+1)*head_dim) → set to 0.
        """
        import torch as _torch
        n_layers, n_heads = ko_mask.shape
        head_dim = self.model.config.d_model // n_heads
        blocks = list(self.model.model.transformer.blocks)
        n_hooks = 0
        for li, block in enumerate(blocks):
            heads_to_ko = [hi for hi in range(n_heads) if ko_mask[li, hi]]
            if not heads_to_ko:
                continue
            slices = [(hi * head_dim, (hi + 1) * head_dim) for hi in heads_to_ko]
            def make_hook(slices_):
                def pre_hook(module, args):
                    inp = args[0]
                    inp = inp.clone()
                    for s_lo, s_hi in slices_:
                        inp[..., s_lo:s_hi] = 0
                    return (inp,) + args[1:]
                return pre_hook
            block.attn_out.register_forward_pre_hook(make_hook(slices))
            n_hooks += 1
        print(f"[eval_llada] knockout hooks installed on {n_hooks}/{n_layers} layers")

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        
        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])
        
        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                # Apply chat template uniformly for instruct models (incl. HumanEval),
                # matching Fast-dLLM upstream + Elastic-Cache convention.
                if self.is_instruct:
                    m = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(
                        m, add_generation_prompt=True, tokenize=False
                    )
                    input_ids = self.tokenizer(user_input)["input_ids"]
                else:
                    user_input = question
                    input_ids = self.tokenizer(user_input)["input_ids"]
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))
            
            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([torch.full((1, max_len - len(input_ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device), torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)
            
            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros((batched_input_ids.shape[0], 1, max_len+self.gen_length, max_len+self.gen_length), device=self.device, dtype=torch.bool)
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i]:, pad_len[i]:] = True


            stop_tokens = req.args[1]['until']
            input_ids = batched_input_ids
            if self.cheap_e3:
                from cheap_e3_inference_llada import generate_cheap_e3
                generated_answer, nfe = generate_cheap_e3(
                    self.model, self._cheap_capture, input_ids,
                    K=self.cheap_K, k_schedule=self.cheap_k_schedule,
                    warmup=self.cheap_warmup, threshold=self.threshold or 0.0,
                    window=self.cheap_window,
                    gen_length=self.gen_length, block_length=self.block_length, steps=self.steps,
                    use_kv_cache=self.cheap_use_kv_cache,
                    include_u_t=self.cheap_include_u_t,
                    window_only=self.cheap_window_only,
                    zar_mode=self.cheap_zar_mode,
                    zar_lookahead=self.cheap_zar_lookahead,
                    zar_prompt_tail=self.cheap_zar_prompt_tail,
                    mask_id=self.mask_id,
                )
            elif self.ragged:
                # Our ragged dual-cache + CUDA Graph
                from experiments.fastgen_head_probe.end2end_ragged_dual import (
                    generate_with_ragged_dual_cache,
                )
                res = generate_with_ragged_dual_cache(
                    self.model, self.tokenizer, prompt_text=None, input_ids=input_ids,
                    gen=self.gen_length, block=self.block_length, steps=self.gen_length,
                    use_ragged=True, K_ratio=self.K_ratio, use_cuda_graph=self.use_cuda_graph,
                    K_block_schedule=self.K_blocks,
                    punct_arr=self.punct_arr, special_arr=self.special_arr,
                    dpad_intersect=self.dpad_intersect,
                    right_edge_floor=self.right_edge_floor,
                    near_future_floor=self.near_future_floor,
                    r_curr_floor=self.r_curr_floor,
                    recency_window=self.recency_window,
                    r_future_stride=self.r_future_stride,
                    head_policy=self.head_policy,
                    head_policy_recency_W=self.head_policy_recency_W,
                    v_weighted_topk=self.v_weighted_topk,
                )
                generated_answer = res["x"] if "x" in res else None
                if generated_answer is None:
                    # Reconstruct from gen_text — fallback (slower)
                    gen_ids = self.tokenizer(res["gen_text"], add_special_tokens=False)["input_ids"]
                    gen_t = torch.tensor(gen_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                    generated_answer = torch.cat([input_ids, gen_t], dim=1)
                # nfe estimate: 1 warm pass per block + (spb-1) block-internal per block
                nb = self.gen_length // self.block_length
                nfe = nb * (self.gen_length // nb)
            elif self.use_cache:
                if self.dual_cache:
                    generated_answer, nfe = generate_with_dual_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length,
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
                else:
                    generated_answer, nfe = generate_with_prefix_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length,
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
            else:
                generated_answer, nfe = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length,
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)

            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                batched_generated_answer = [self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True) for i in range(len(generated_answer_ids))]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    generated_answer_i = self.tokenizer.decode(generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False)
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        num_nfe += nfe
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                # Incrementally save newly generated answers
                with open(save_path, 'a', encoding='utf-8') as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')

            for i in range(len(batched_generated_answer)):
                print('=' * 20)
                # print('question: ', question)
                print('answer: ', batched_generated_answer[i])
                print('nfe: ', nfe)
                print('avg nfe: ', num_nfe / len(output))
                print('=' * 20, end='\n\n')
            # self.accelerator.wait_for_everyone()
        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")
            
        return output


if __name__ == "__main__":
    cli_evaluate()
    
