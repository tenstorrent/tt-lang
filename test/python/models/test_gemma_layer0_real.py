# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Layer 0 (sliding) with real checkpoint weights, card 0, vs torch ref.

Skips unless the HF snapshot is present (CI machines without it skip).
"""

import glob
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytest.importorskip("safetensors", exc_type=ImportError)

from gemma4.config import Gemma4Config
from gemma4.decode_chain import GlobalChain, SlidingChain, StepState
from gemma4.host import to_dev

SNAP = glob.glob("/root/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/*")
TILE, CTX = 32, 1024


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * w


@pytest.mark.skipif(not SNAP, reason="checkpoint not present")
@pytest.mark.parametrize("lidx,kind", [(0, "sliding"), (5, "global")])
def test_layer_real(device, lidx, kind):
    from gemma4.weights import Checkpoint, embed_weights, layer_weights
    from test_gemma_decode_chain import TorchLayer

    torch.manual_seed(0)
    cfg = Gemma4Config()
    ck = Checkpoint(SNAP[0])
    w = layer_weights(ck, cfg, lidx, card=0)
    embed, _ = embed_weights(ck)
    H = cfg.hidden

    tok, pos = 2, 0  # <bos>
    x = embed[tok].float() * H ** 0.5
    want = TorchLayer(w, cfg, kind).step(x, pos)

    st = StepState(device, cfg, CTX)
    layer = (SlidingChain(w, device, cfg, st) if kind == "sliding"
             else GlobalChain(w, device, cfg, st, CTX))
    st.prime(device, pos)
    st.step()
    x_t = torch.zeros(TILE, H, dtype=torch.bfloat16)
    x_t[0] = x.to(torch.bfloat16)
    out = layer.step(to_dev(x_t, device))

    got = ttnn.to_torch(out).float()[0]
    pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
    assert pcc > 0.99, f"layer {lidx} pcc {pcc}"
