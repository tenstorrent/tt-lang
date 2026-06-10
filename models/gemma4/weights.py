# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HF checkpoint -> per-card chain weight dicts (host staging, pre-decode).

Tensors load lazily from the safetensors shards so a single layer fits in
host RAM. Sharding (TP=4): q/k/v columns by head, o/down rows, experts
E/4 per card, router and global K replicated.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .host import MLP_PAD

PRE = "model.language_model"


class Checkpoint:
    def __init__(self, snapshot):
        snapshot = Path(snapshot)
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        self.shards = {n: snapshot / f for n, f in index["weight_map"].items()}
        self._open = {}

    def get(self, name):
        path = self.shards[name]
        if path not in self._open:
            self._open[path] = safe_open(path, framework="pt")
        return self._open[path].get_tensor(name).float()


def norm(ckpt, layer, name):
    return 1 + ckpt.get(f"{PRE}.layers.{layer}.{name}.weight")


def layer_weights(ckpt, cfg, layer, card):
    """Per-card weight dict for SlidingChain/GlobalChain + FFNChain."""
    L = f"{PRE}.layers.{layer}"
    H = cfg.hidden
    is_global = cfg.layer_type(layer) == "global"
    D = cfg.global_head_dim if is_global else cfg.head_dim
    qh = 4

    w = {
        "g_in": norm(ckpt, layer, "input_layernorm"),
        "g_postattn": norm(ckpt, layer, "post_attention_layernorm"),
        "g_preffw": norm(ckpt, layer, "pre_feedforward_layernorm"),
        "g_postffw1": norm(ckpt, layer, "post_feedforward_layernorm_1"),
        "g_preffw2": norm(ckpt, layer, "pre_feedforward_layernorm_2"),
        "g_postffw2": norm(ckpt, layer, "post_feedforward_layernorm_2"),
        "g_postffw": norm(ckpt, layer, "post_feedforward_layernorm"),
        "q_norm": 1 + ckpt.get(f"{L}.self_attn.q_norm.weight"),
        "k_norm": 1 + ckpt.get(f"{L}.self_attn.k_norm.weight"),
        "layer_scalar": ckpt.get(f"{L}.layer_scalar").item(),
        "router_w": ckpt.get(f"{L}.router.proj.weight"),
        "router_scale": ckpt.get(f"{L}.router.scale"),
        "per_expert": ckpt.get(f"{L}.router.per_expert_scale"),
    }

    wq = ckpt.get(f"{L}.self_attn.q_proj.weight")
    wk = ckpt.get(f"{L}.self_attn.k_proj.weight")
    wo = ckpt.get(f"{L}.self_attn.o_proj.weight")
    qs = card * qh
    if is_global:
        # Q shards 4/card; each card stages only the KV head its Q group reads.
        kv = qs // (cfg.q_heads // cfg.global_kv_heads)
        w["w_q"] = wq[qs * D:(qs + qh) * D].T.contiguous()
        w["w_k"] = wk[kv * D:(kv + 1) * D].T.contiguous()
        w["w_o"] = wo[:, qs * D:(qs + qh) * D].T.contiguous()
    else:
        w["v_norm"] = torch.ones(D)
        wv = ckpt.get(f"{L}.self_attn.v_proj.weight")
        kvh, ks = 2, card * 2
        w["w_qkv"] = torch.cat([
            wq[qs * D:(qs + qh) * D],
            wk[ks * D:(ks + kvh) * D],
            wv[ks * D:(ks + kvh) * D]]).T.contiguous()
        w["w_o"] = wo[:, qs * D:(qs + qh) * D].T.contiguous()

    P, inter = MLP_PAD, cfg.mlp_inter // 4
    s = card * inter
    gate = ckpt.get(f"{L}.mlp.gate_proj.weight")[s:s + inter]
    up = ckpt.get(f"{L}.mlp.up_proj.weight")[s:s + inter]
    down = ckpt.get(f"{L}.mlp.down_proj.weight")[:, s:s + inter]
    w["w_gate"] = torch.nn.functional.pad(gate.T, (0, P - inter)).contiguous()
    w["w_up"] = torch.nn.functional.pad(up.T, (0, P - inter)).contiguous()
    w["w_down"] = torch.nn.functional.pad(down.T, (0, 0, 0, P - inter)).contiguous()

    E = cfg.experts // 4
    es = card * E
    gu = ckpt.get(f"{L}.experts.gate_up_proj")[es:es + E]      # [E, 2I, H]
    dn = ckpt.get(f"{L}.experts.down_proj")[es:es + E]          # [E, H, I]
    w["w_gu"] = gu.transpose(1, 2).contiguous()                 # [E, H, 2I]
    w["w_dn"] = dn.transpose(1, 2).contiguous()                 # [E, I, H]
    return w


def embed_weights(ckpt):
    return ckpt.get(f"{PRE}.embed_tokens.weight"), 1 + ckpt.get(f"{PRE}.norm.weight")
