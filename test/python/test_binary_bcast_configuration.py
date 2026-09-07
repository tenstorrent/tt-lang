# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Broadcast initialization preserves live results across DST bank switches."""

import pytest
import torch

import ttl
from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device


def make_normalized_broadcast_pairs(fp32_dest_acc_en, dst_full_sync_en):
    @ttl.operation(
        grid=(1, 1),
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
    )
    def normalized_broadcast_pairs(a, b, c, d, norm, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 8), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 8), block_count=2)
        d_dfb = ttl.make_dataflow_buffer_like(d, shape=(1, 1), block_count=2)
        norm_dfb = ttl.make_dataflow_buffer_like(norm, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 8), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_dfb.wait() as av,
                b_dfb.wait() as bv,
                c_dfb.wait() as cv,
                d_dfb.wait() as dv,
                norm_dfb.wait() as nv,
                out_dfb.reserve() as output,
            ):
                br = ttl.block.broadcast(bv, dims=[1], shape=(1, 8))
                dr = ttl.block.broadcast(dv, dims=[1], shape=(1, 8))
                merged = av * br + cv * dr
                normalization = ttl.block.broadcast(
                    ttl.math.recip(nv), dims=[1], shape=(1, 8)
                )
                output.store(merged * normalization)

        @ttl.datamovement()
        def reader():
            ttl.copy(a[0:1, 0:8], a_dfb.reserve())
            ttl.copy(b[0:1, 0:1], b_dfb.reserve())
            ttl.copy(c[0:1, 0:8], c_dfb.reserve())
            ttl.copy(d[0:1, 0:1], d_dfb.reserve())
            ttl.copy(norm[0:1, 0:1], norm_dfb.reserve())

        @ttl.datamovement()
        def writer():
            ttl.copy(out_dfb.wait(), out[0:1, 0:8])

    return normalized_broadcast_pairs


@pytest.mark.parametrize("dst_full_sync_en", [False, True])
@pytest.mark.parametrize(
    "torch_dtype,fp32_dest_acc_en", [(torch.bfloat16, False), (torch.float32, True)]
)
def test_binary_then_unary_broadcast(
    device, fp32_dest_acc_en, dst_full_sync_en, torch_dtype
):
    # The tile domain spans multiple DST acquisitions, including the upper bank
    # in half-sync mode. Unary broadcast must preserve the preceding products.
    torch.manual_seed(947)
    a = torch.randn(32, 256, dtype=torch_dtype) * 0.1
    b = torch.rand(32, 32, dtype=torch_dtype) + 0.5
    c = torch.randn(32, 256, dtype=torch_dtype) * 0.1
    d = torch.rand(32, 32, dtype=torch_dtype) + 0.5
    norm = torch.rand(32, 32, dtype=torch_dtype) + 0.5
    expected = (a.float() * b[:, :1].float() + c.float() * d[:, :1].float()) / norm[
        :, :1
    ].float()
    tensors = [
        to_dram(value, device) for value in (a, b, c, d, norm, torch.zeros_like(a))
    ]
    kernel = make_normalized_broadcast_pairs(fp32_dest_acc_en, dst_full_sync_en)
    kernel(*tensors)
    actual = ttnn.to_torch(tensors[-1]).float()
    assert_allclose(actual, expected, rtol=0.03, atol=0.005)
