# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-layer PCC vs full-model torch on TP=4 (real weights, BOS, pos 0).

Usage: python -m gemma4.parity --layers 6
"""

import argparse
import glob

import torch
import ttnn

from .config import Gemma4Config
from .decode_chain import DecodeChain, GlobalChain, SlidingChain, StepState, TILE
from .host import from_dev, to_dev
from .weights import Checkpoint, embed_weights, layer_weights

SNAP = "/root/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/*"
PRE = "model.language_model"
TP = 4


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


def full_ref_weights(ckpt, cfg, L):
    """Full-model dict in the tp_chain TorchLayer key scheme."""
    n = lambda k: ckpt.get(f"{PRE}.layers.{L}.{k}.weight")
    g = lambda k: ckpt.get(f"{PRE}.layers.{L}.{k}")
    w = {
        "g_in": n("input_layernorm"),
        "g_postattn": n("post_attention_layernorm"),
        "g_preffw": n("pre_feedforward_layernorm"),
        "g_postffw1": n("post_feedforward_layernorm_1"),
        "g_preffw2": n("pre_feedforward_layernorm_2"),
        "g_postffw2": n("post_feedforward_layernorm_2"),
        "g_postffw": n("post_feedforward_layernorm"),
        "q_norm": n("self_attn.q_norm"),
        "k_norm": n("self_attn.k_norm"),
        "v_norm": torch.ones(cfg.head_dim),
        "layer_scalar": g("layer_scalar").item(),
        "router_w": g("router.proj.weight"),
        "router_scale": g("router.scale"),
        "per_expert": g("router.per_expert_scale"),
        "wq_f": g("self_attn.q_proj.weight"),
        "wk_f": g("self_attn.k_proj.weight"),
        "wo_f": g("self_attn.o_proj.weight"),
        "w_gate_f": g("mlp.gate_proj.weight").T.contiguous(),
        "w_up_f": g("mlp.up_proj.weight").T.contiguous(),
        "w_down_f": g("mlp.down_proj.weight").T.contiguous(),
        "w_gu_f": g("experts.gate_up_proj").transpose(1, 2).contiguous(),
        "w_dn_f": g("experts.down_proj").transpose(1, 2).contiguous(),
    }
    if cfg.layer_type(L) == "sliding":
        w["wv_f"] = g("self_attn.v_proj.weight")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=1024)
    args = ap.parse_args()

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "test" / "python" / "models"))
    from test_gemma_tp_chain import TorchLayer

    cfg = Gemma4Config()
    ckpt = Checkpoint(glob.glob(SNAP)[0])
    embed, _ = embed_weights(ckpt)
    H = cfg.hidden
    tok, pos = 2, 0

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    try:
        st = StepState(mesh, cfg, args.ctx)
        st.prime(mesh, pos)
        st.step()
        # Build all chains before staging activations: staging activations
        # first shifts DRAM layout and corrupts the expert indexed-GEMV path
        # (alloc-order compiler bug; bring-up generation uses chain-first
        # order so parity must match).
        layers = []
        for L in range(args.layers):
            cards = [layer_weights(ckpt, cfg, L, c) for c in range(TP)]
            layers.append(SlidingChain(cards, mesh, cfg, st)
                          if cfg.layer_type(L) == "sliding"
                          else GlobalChain(cards, mesh, cfg, st, args.ctx))
        x = embed[tok].float() * H ** 0.5
        x_t = torch.zeros(TILE, H, dtype=torch.bfloat16)
        x_t[0] = x.to(torch.bfloat16)
        x_d = to_dev(x_t, mesh)
        # Feed each layer the ref input so per-layer PCC is decoupled from
        # bf16 drift accumulating across layers.
        for L, layer in enumerate(layers):
            kind = cfg.layer_type(L)
            ref = TorchLayer(full_ref_weights(ckpt, cfg, L), cfg, kind)
            want = ref.step(x, pos)
            out_d = layer.step(x_d)
            pccs = []
            for c in range(TP):
                got = from_dev(out_d, card=c)[0]
                pccs.append(round(torch.corrcoef(
                    torch.stack([got, want]))[0, 1].item(), 5))
            g0 = from_dev(out_d, card=0)[0].double()
            w0 = want.double()
            rel_l2 = ((g0 - w0).norm() / w0.norm()).item()
            magratio = (g0.norm() / w0.norm()).item()
            print(f"layer {L} ({kind}) pcc {pccs} "
                  f"relL2={rel_l2:.5f} magratio={magratio:.5f}", flush=True)
            x = want
            x_t[0] = x.to(torch.bfloat16)
            x_d = to_dev(x_t, mesh)
    finally:
        ttnn.close_device(mesh)


if __name__ == "__main__":
    main()
