# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Sub-tile data-movement coverage.

1. Tiny-page tensor->CB->tensor copies: when ``page < aligned_page_size``,
   confirm the NOC transfer uses logical tile bytes (not the aligned size)
   so padding/neighbor tiles are not pulled into the destination.

2. Tilization mismatch rejection: ``ttl.copy`` (tensor<->CB) and
   ``block.store`` (CB->CB) must error when tile HxW differs.
"""

from __future__ import annotations

import pytest
import torch

import ttl
from ttl.diagnostics import TTLangCompileError

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)


@ttl.operation(grid=(1, 1))
def subtile_tensor_copy(src, dst):
    """Copy one tile tensor->CB->tensor with an empty compute thread."""
    cb = ttl.make_dataflow_buffer_like(src, shape=(1, 1), block_count=2)
    blk_in = cb.reserve()
    ttl.copy(src[0:1, 0:1], blk_in)
    blk_out = cb.wait()
    ttl.copy(blk_out, dst[0:1, 0:1])


@ttl.operation(grid=(1, 1))
def _copy_tensor_32_into_cb_16(src, dst):
    """INVALID: copy a 32x32 tensor tile into a 16x16 CB."""
    cb = ttl.make_dfb(ttnn.bfloat16, shape=(1, 1), block_count=2, tile=(16, 16))
    blk = cb.reserve()
    ttl.copy(src[0:1, 0:1], blk)
    out_blk = cb.wait()
    ttl.copy(out_blk, dst[0:1, 0:1])


@ttl.operation(grid=(1, 1))
def _store_cb_16_into_cb_32(out):
    """INVALID: store a 16x16 CB block into a 32x32 CB reserve."""
    src_dfb = ttl.make_dfb(ttnn.bfloat16, shape=(1, 1), block_count=2, tile=(16, 16))
    dst_dfb = ttl.make_dfb(ttnn.bfloat16, shape=(1, 1), block_count=2, tile=(32, 32))

    @ttl.compute()
    def compute():
        o = dst_dfb.reserve()
        o.store(src_dfb.wait())


def _to_device(torch_tensor, device, tile_hw, dtype, memory_config):
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=ttnn.Tile(tile_hw),
        memory_config=memory_config,
    )


def _torch_dtype(tt_dtype):
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    if tt_dtype == ttnn.uint8:
        return torch.uint8
    raise ValueError(f"unsupported dtype {tt_dtype}")


# (tile_hw, tt_dtype, memory_config, label)
TINY_COPY_CASES = [
    ((1, 16), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, "dram_bf16_1x16"),
    ((1, 16), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, "l1_bf16_1x16"),
    ((1, 16), ttnn.uint8, ttnn.DRAM_MEMORY_CONFIG, "dram_u8_1x16"),
    ((1, 16), ttnn.uint8, ttnn.L1_MEMORY_CONFIG, "l1_u8_1x16"),
    ((1, 32), ttnn.uint8, ttnn.DRAM_MEMORY_CONFIG, "dram_u8_1x32"),
]


@pytest.mark.parametrize(
    "tile_hw, tt_dtype, memory_config, label",
    TINY_COPY_CASES,
    ids=[c[-1] for c in TINY_COPY_CASES],
)
def test_subtile_tensor_copy_single_tile(
    device, tile_hw, tt_dtype, memory_config, label
):
    """Round-trip one tiny tile and report page vs aligned sizes."""
    h, w = tile_hw
    torch_dtype = _torch_dtype(tt_dtype)
    tile = ttnn.Tile(tile_hw)
    src_t = (
        (torch.arange(h * w, dtype=torch.int64) % 200 + 1).to(torch_dtype).reshape(h, w)
    )

    src = _to_device(src_t, device, tile_hw, tt_dtype, memory_config)
    dst = _to_device(
        torch.zeros(h, w, dtype=torch_dtype), device, tile_hw, tt_dtype, memory_config
    )

    tile_bytes = tile.get_tile_size(tt_dtype)
    page = src.buffer_page_size()
    aligned = src.buffer_aligned_page_size()
    print(
        f"[{label}] tile_bytes={tile_bytes} page={page} aligned={aligned} "
        f"(aligned-page gap={aligned - page})"
    )

    subtile_tensor_copy(src, dst)

    got = ttnn.to_torch(dst).reshape(h, w)
    if torch_dtype == torch.bfloat16:
        got = got.to(torch.bfloat16)
        assert torch.equal(got, src_t), f"[{label}] mismatch got={got} exp={src_t}"
    else:
        assert torch.equal(
            got.to(torch.int64), src_t.to(torch.int64)
        ), f"[{label}] mismatch got={got} exp={src_t}"

    ttnn.deallocate(src)
    ttnn.deallocate(dst)


@pytest.mark.parametrize(
    "tile_hw, tt_dtype, memory_config, label",
    [
        ((1, 16), ttnn.uint8, ttnn.DRAM_MEMORY_CONFIG, "dram_u8_1x16"),
        ((1, 16), ttnn.uint8, ttnn.L1_MEMORY_CONFIG, "l1_u8_1x16"),
        ((1, 16), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, "dram_bf16_1x16"),
    ],
    ids=["dram_u8_1x16", "l1_u8_1x16", "dram_bf16_1x16"],
)
def test_subtile_tensor_copy_neighbor_untouched(
    device, tile_hw, tt_dtype, memory_config, label
):
    """Copy tile 0 of a 2-tile tensor; tile 1 must stay at its sentinel.

    If the NOC transfer uses ``aligned_page_size`` (> logical tile bytes),
    writing tile 0 can spill into the padding/next page and corrupt tile 1.
    """
    h, w = tile_hw
    torch_dtype = _torch_dtype(tt_dtype)
    # Two tiles side-by-side along W.
    shape = (h, 2 * w)
    src_t = torch.zeros(shape, dtype=torch_dtype)
    dst_t = torch.zeros(shape, dtype=torch_dtype)
    # Tile 0: 1..N, tile 1: 200+i (distinct sentinel).
    src_t[:, :w] = (
        (torch.arange(h * w, dtype=torch.int64) % 100 + 1).to(torch_dtype).reshape(h, w)
    )
    src_t[:, w:] = (
        (torch.arange(h * w, dtype=torch.int64) % 100 + 200)
        .to(torch_dtype)
        .reshape(h, w)
    )
    dst_t[:, :w] = 0
    dst_t[:, w:] = 77  # sentinel that must survive a tile-0-only copy

    src = _to_device(src_t, device, tile_hw, tt_dtype, memory_config)
    dst = _to_device(dst_t, device, tile_hw, tt_dtype, memory_config)

    tile = ttnn.Tile(tile_hw)
    print(
        f"[{label} neighbor] tile_bytes={tile.get_tile_size(tt_dtype)} "
        f"page={src.buffer_page_size()} aligned={src.buffer_aligned_page_size()}"
    )

    # Kernel always copies src[0:1,0:1] -> dst[0:1,0:1] (first tile only).
    subtile_tensor_copy(src, dst)

    got = ttnn.to_torch(dst).reshape(shape)
    if torch_dtype == torch.bfloat16:
        got = got.to(torch.bfloat16)

    got0, got1 = got[:, :w], got[:, w:]
    exp0, exp1 = src_t[:, :w], dst_t[:, w:]

    if torch_dtype == torch.uint8:
        assert torch.equal(
            got0.to(torch.int64), exp0.to(torch.int64)
        ), f"[{label}] tile0 mismatch got={got0} exp={exp0}"
        assert torch.equal(got1.to(torch.int64), exp1.to(torch.int64)), (
            f"[{label}] tile1 corrupted (possible aligned-page oversize write): "
            f"got={got1} exp={exp1}"
        )
    else:
        assert torch.equal(
            got0, exp0
        ), f"[{label}] tile0 mismatch got={got0} exp={exp0}"
        assert torch.equal(got1, exp1), (
            f"[{label}] tile1 corrupted (possible aligned-page oversize write): "
            f"got={got1} exp={exp1}"
        )

    ttnn.deallocate(src)
    ttnn.deallocate(dst)


def test_copy_rejects_mismatched_tile_shape(device):
    """ttl.copy must reject tensor tile HxW that differs from the CB tile."""
    src = _to_device(
        torch.ones((32, 32), dtype=torch.bfloat16),
        device,
        (32, 32),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    dst = _to_device(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        device,
        (32, 32),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(
        TTLangCompileError,
        match=r"tensor tile shape 32x32 must match CB tile shape 16x16",
    ):
        _copy_tensor_32_into_cb_16(src, dst)
    ttnn.deallocate(src)
    ttnn.deallocate(dst)


def test_store_rejects_mismatched_tile_shape(device):
    """CB->CB store must reject source/destination tile HxW mismatch."""
    out = _to_device(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        device,
        (32, 32),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(
        TTLangCompileError,
        match=r"source tile shape 16x16 must match destination CB tile shape 32x32",
    ):
        _store_cb_16_into_cb_32(out)
    ttnn.deallocate(out)
