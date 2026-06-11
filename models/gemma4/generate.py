# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Greedy generation on 4xp150: prompt prefill via decode steps, B=1.

Usage: python -m gemma4.generate --prompt "..." --tokens 32 [--layers 30]
"""

import argparse
import glob
import time

import torch
import ttnn

from .config import Gemma4Config
from .decode_chain import DecodeChain, GlobalChain, SlidingChain, StepState
from .weights import Checkpoint, embed_weights, layer_weights

SNAP = "/root/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/*"
TP = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--ctx", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    snap = glob.glob(SNAP)[0]
    tokenizer = AutoTokenizer.from_pretrained(snap)
    ids = tokenizer(args.prompt)["input_ids"]
    if ids[0] != tokenizer.bos_token_id:
        ids = [tokenizer.bos_token_id] + ids
    print(f"prompt ids {ids}", flush=True)

    cfg = Gemma4Config()
    ckpt = Checkpoint(snap)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    try:
        t0 = time.time()
        st = StepState(mesh, cfg, args.ctx)
        layers = []
        for L in range(args.layers):
            cards = [layer_weights(ckpt, cfg, L, c) for c in range(TP)]
            kind = cfg.layer_type(L)
            layers.append(SlidingChain(cards, mesh, cfg, st) if kind == "sliding"
                          else GlobalChain(cards, mesh, cfg, st, args.ctx))
            print(f"layer {L} ({kind}) staged {time.time() - t0:.0f}s", flush=True)
        embed, g_final = embed_weights(ckpt)
        chain = DecodeChain(layers, st, embed, g_final, embed.T.contiguous(), mesh, cfg)
        print(f"staged in {time.time() - t0:.0f}s", flush=True)

        out = list(ids)
        t0 = time.time()
        for i, tok in enumerate(ids):
            chain.prime(tok, i)
            chain.step()
            print(f"prefill {i} {time.time() - t0:.1f}s", flush=True)
        out.append(chain.read_token())
        for _ in range(args.tokens - 1):
            chain.step()
            out.append(chain.read_token())
            print(f"gen {len(out) - len(ids)}: {out[-1]} {time.time() - t0:.1f}s",
                  flush=True)
        print("text:", tokenizer.decode(out), flush=True)
    finally:
        ttnn.close_device(mesh)


if __name__ == "__main__":
    main()
