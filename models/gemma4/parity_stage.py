# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stage-wise PCC inside layer 0 on TP=4 (real weights, BOS, pos 0)."""

import glob

import torch
import ttnn

from .config import Gemma4Config
from .decode_chain import SlidingChain, StepState, TILE, all_reduce
from .host import from_dev, to_dev
from .parity import full_ref_weights, rms
from .weights import Checkpoint, embed_weights, layer_weights

SNAP = "/root/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/*"
TP = 4


def pcc(a, b):
    return round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 5)


def main():
    cfg = Gemma4Config()
    ckpt = Checkpoint(glob.glob(SNAP)[0])
    embed, _ = embed_weights(ckpt)
    H, D, eps = cfg.hidden, cfg.head_dim, cfg.eps
    tok, pos = 2, 0

    w = full_ref_weights(ckpt, cfg, 0)
    x = embed[tok].float() * H ** 0.5
    xn = rms(x, w["g_in"], eps)
    heads = [xn @ w["wq_f"][h * D:(h + 1) * D].T for h in range(16)]
    q = [rms(h, w["q_norm"], eps) for h in heads]
    k = [rms(xn @ w["wk_f"][g * D:(g + 1) * D].T, w["k_norm"], eps) for g in range(8)]
    v = [rms(xn @ w["wv_f"][g * D:(g + 1) * D].T, 1.0, eps) for g in range(8)]
    # pos 0: softmax over one position = 1 -> attn out = v[g]
    attn_ref = w["wo_f"] @ torch.cat([v[h // 2] for h in range(16)])
    hres = x + rms(attn_ref, w["g_postattn"], eps)
    hn = rms(hres, w["g_preffw"], eps)
    dense_ref = (torch.nn.functional.gelu(hn @ w["w_gate_f"], approximate="tanh")
                 * (hn @ w["w_up_f"])) @ w["w_down_f"]
    rnorm = hres / torch.sqrt(hres.pow(2).mean() + eps) * w["router_scale"] * H ** -0.5
    probs = torch.softmax(w["router_w"] @ rnorm, dim=-1)
    wts, idx = torch.topk(probs, cfg.top_k)
    wts = wts / wts.sum() * w["per_expert"][idx]
    hn2 = rms(hres, w["g_preffw2"], eps)
    I = cfg.moe_inter
    exp_ref = torch.zeros(H)
    for t in range(cfg.top_k):
        gu = hn2 @ w["w_gu_f"][idx[t]]
        exp_ref += (torch.nn.functional.gelu(gu[:I], approximate="tanh")
                    * gu[I:] * wts[t]) @ w["w_dn_f"][idx[t]]
    print("ref idx", idx.tolist(), flush=True)
    print("ref wts", [round(v.item(), 4) for v in wts], flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    try:
        st = StepState(mesh, cfg, 1024)
        st.prime(mesh, pos)
        st.step()
        cards = [layer_weights(ckpt, cfg, 0, c) for c in range(TP)]
        layer = SlidingChain(cards, mesh, cfg, st)
        x_t = torch.zeros(TILE, H, dtype=torch.bfloat16)
        x_t[0] = x.to(torch.bfloat16)
        layer.step(to_dev(x_t, mesh))

        xn_b = xn.to(torch.bfloat16).float()
        heads_dev = from_dev(layer.heads, card=0)
        for r, name, want_v in ((0, "q0", q[0]), (4, "k0", k[0]), (6, "v0", v[0])):
            print(name, "pcc", pcc(heads_dev[r * TILE], want_v), flush=True)
        o_dev = from_dev(layer.o_row, card=0)[0]
        print("o_row pcc", pcc(o_dev, torch.cat([v[0], v[0], v[1], v[1]])), flush=True)
        attn_parts = [from_dev(layer.attn, card=c)[0] for c in range(TP)]
        attn_dev = sum(attn_parts)
        print("attn pcc", pcc(attn_dev, attn_ref), flush=True)
        h_dev = from_dev(layer.h, card=0)[0]
        print("h pcc", pcc(h_dev, hres), flush=True)
        f = layer.ffn
        dense_dev = from_dev(f.dense, card=0)[0]
        print("dense pcc", pcc(dense_dev, dense_ref), flush=True)
        exp_dev = from_dev(f.dn, card=0)[0]
        print("exp pcc", pcc(exp_dev, exp_ref), flush=True)
        idx_dev = from_dev(f.idx, card=0)[0]
        wts_dev = [from_dev(f.wts, card=c)[0] for c in range(TP)]
        print("dev idx", [int(idx_dev[t * TILE]) for t in range(8)], flush=True)
        print("dev wts", [[round(wc[t * TILE].item(), 4) for t in range(8)]
                          for wc in wts_dev], flush=True)
        out = from_dev(f.out, card=0)[0]
        want = (hres + rms(rms(dense_ref, w["g_postffw1"], eps)
                           + rms(exp_ref, w["g_postffw2"], eps),
                           w["g_postffw"], eps)) * w["layer_scalar"]
        print("layer pcc", pcc(out, want), flush=True)
    finally:
        ttnn.close_device(mesh)


if __name__ == "__main__":
    main()
