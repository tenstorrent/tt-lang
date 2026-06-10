# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 26B-A4B text config and TP=4 shard plan (decode, B=1)."""

from dataclasses import dataclass

TILE = 32


@dataclass(frozen=True)
class Gemma4Config:
    hidden: int = 2816
    layers: int = 30
    q_heads: int = 16
    kv_heads: int = 8           # sliding layers
    global_kv_heads: int = 2    # full-attention layers
    head_dim: int = 256
    global_head_dim: int = 512
    sliding_window: int = 1024
    mlp_inter: int = 2112
    experts: int = 128
    top_k: int = 8
    moe_inter: int = 704
    vocab: int = 262144
    eps: float = 1e-6
    softcap: float = 30.0
    rope_theta: float = 10000.0
    global_rope_theta: float = 1000000.0
    global_rot_frac: float = 0.25

    def layer_type(self, idx):
        return "global" if idx % 6 == 5 else "sliding"


@dataclass(frozen=True)
class ShardPlan:
    """Per-card slices at TP=4. Heads col-shard; O/down row-shard; experts
    32/card; global K replicated (2 heads, not 4-splittable); router full."""
    tp: int = 4

    def q_heads(self, cfg):
        return cfg.q_heads // self.tp          # 4

    def kv_heads(self, cfg):
        return cfg.kv_heads // self.tp         # 2 (sliding)

    def experts(self, cfg):
        return cfg.experts // self.tp          # 32

    def mlp_inter(self, cfg):
        return cfg.mlp_inter // self.tp        # 528

    def moe_inter(self, cfg):
        return cfg.moe_inter                   # 704, expert-local
