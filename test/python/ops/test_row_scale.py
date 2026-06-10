# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""row_scale: out = a * s[0,0] (or a / s with recip), multi-chunk widths.

The multi-chunk case is a regression test: a scalar tile hoisted out of the
width loop was popped after the first chunk, scaling later chunks by stale L1.
"""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.ops.elementwise import make_row_scale

TILE = 32


def to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("Dt,WCt", [(16, 8), (8, 8)])
@pytest.mark.parametrize("recip", [False, True])
def test_row_scale(device, Dt, WCt, recip):
    torch.manual_seed(0)
    a = torch.randn(TILE, Dt * TILE)
    s = torch.full((TILE, TILE), 4.0)
    out = torch.zeros(TILE, Dt * TILE)
    want = a[0] / 4.0 if recip else a[0] * 4.0

    a_d, s_d, out_d = (to_dev(t, device) for t in (a, s, out))
    make_row_scale(Dt, WCt, recip=recip)(a_d, s_d, out_d)
    got = ttnn.to_torch(out_d).float()[0]
    pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
    assert pcc > 0.999, f"Dt={Dt} WCt={WCt} recip={recip} pcc {pcc}"
