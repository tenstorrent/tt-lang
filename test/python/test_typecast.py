# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end correctness tests for ttl.math.typecast.

Each parametrized case uses a different SFPU LLK kernel in tt-metal
(a distinct typecast_tile<> template instantiation), so the suite provides
runtime coverage that the compile-only lit test in simple_typecast.py cannot.

Supported floating-point pairs:
  bf16 -> f32   (lossless widening, hardware result must be exact)
  f32  -> bf16  (lossy narrowing, hardware rounding must match torch)

Block floating-point pairs use PyTorch-backed bf16/f32 tensors with explicit
TTNN bf4/bf8 tensor dtypes because PyTorch has no native bf4/bf8 dtype.

Integer pairs (int32 <-> float) are omitted: the SFPU typecast_tile LLK is
defined only for floating-point data formats; issuing an integer typecast
would require a different hardware path not currently exposed by ttl.
"""

import atexit
import importlib.util
import os
import tempfile

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttlang_test_utils import assert_allclose

TILE = 32
BLOCK_ROWS = 3
BLOCK_COLS = 5
BLOCK_SHAPE = (BLOCK_ROWS, BLOCK_COLS)
BLOCK_ID = f"{BLOCK_ROWS}x{BLOCK_COLS}"


# =============================================================================
# Kernel definitions
#
# Each kernel must be defined at module scope so that the TTL source inspector
# can locate the decorated function's source text. The target dtype is encoded
# as a literal in the kernel body because the TTL AST compiler resolves
# torch.float32 / torch.bfloat16 / ttnn.bfloat*_b as compile-time constants.
# =============================================================================


@ttl.operation(grid=(1, 1))
def _typecast_bf16_f32_long_fusion_multi_tile(bf16_inp, out):
    """Cast bf16 to f32 and use it through a longer multi-tile fusion."""
    bf16_dfb = ttl.make_dataflow_buffer_like(bf16_inp, shape=BLOCK_SHAPE, block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=BLOCK_SHAPE, block_count=2)

    @ttl.compute()
    def compute_fn():
        with bf16_dfb.wait() as a, out_dfb.reserve() as o:
            a_f32 = ttl.math.typecast(a, torch.float32)
            fused = ((a_f32 + a_f32) * a_f32) + a_f32
            o.store(fused)

    @ttl.datamovement()
    def dm_read():
        with bf16_dfb.reserve() as blk:
            tx = ttl.copy(bf16_inp[0:BLOCK_ROWS, 0:BLOCK_COLS], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:BLOCK_ROWS, 0:BLOCK_COLS])
            tx.wait()


@ttl.operation(grid=(1, 1))
def _typecast_bf16_f32_stray_f32_fusion_multi_tile(bf16_inp, f32_inp, out):
    """Mix a casted bf16 input with an unrelated direct f32 input."""
    bf16_dfb = ttl.make_dataflow_buffer_like(bf16_inp, shape=BLOCK_SHAPE, block_count=2)
    f32_dfb = ttl.make_dataflow_buffer_like(f32_inp, shape=BLOCK_SHAPE, block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=BLOCK_SHAPE, block_count=2)

    @ttl.compute()
    def compute_fn():
        with bf16_dfb.wait() as a, f32_dfb.wait() as b, out_dfb.reserve() as o:
            a_f32 = ttl.math.typecast(a, torch.float32)
            fused = ((a_f32 + b) * b) + a_f32
            o.store(fused)

    @ttl.datamovement()
    def dm_read():
        with bf16_dfb.reserve() as blk:
            tx = ttl.copy(bf16_inp[0:BLOCK_ROWS, 0:BLOCK_COLS], blk)
            tx.wait()
        with f32_dfb.reserve() as blk:
            tx = ttl.copy(f32_inp[0:BLOCK_ROWS, 0:BLOCK_COLS], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:BLOCK_ROWS, 0:BLOCK_COLS])
            tx.wait()


# =============================================================================
# Parametrized configurations
# =============================================================================

_DTYPE_SPECS = {
    "bf4": {
        "torch_dtype": torch.bfloat16,
        "ttnn_dtype": ttnn.bfloat4_b,
        "typecast_target": "ttnn.bfloat4_b",
        "rtol": 0.1,
        "atol": 0.25,
    },
    "bf8": {
        "torch_dtype": torch.float32,
        "ttnn_dtype": ttnn.bfloat8_b,
        "typecast_target": "ttnn.bfloat8_b",
        "rtol": 0.02,
        "atol": 0.02,
    },
    "bf16": {
        "torch_dtype": torch.bfloat16,
        "ttnn_dtype": ttnn.bfloat16,
        "typecast_target": "torch.bfloat16",
        "rtol": 0.01,
        "atol": 0.01,
    },
    "f32": {
        "torch_dtype": torch.float32,
        "ttnn_dtype": ttnn.float32,
        "typecast_target": "torch.float32",
        "rtol": 5e-3,
        "atol": 1e-2,
    },
}

_MULTI_TILE_SWEEP_CONFIGS = [
    pytest.param("bf4", "f32", id=f"bf4_to_f32_{BLOCK_ID}"),
    pytest.param("bf8", "f32", id=f"bf8_to_f32_{BLOCK_ID}"),
    pytest.param("bf16", "f32", id=f"bf16_to_f32_{BLOCK_ID}"),
    pytest.param("f32", "bf16", id=f"f32_to_bf16_{BLOCK_ID}"),
    pytest.param("bf16", "bf8", id=f"bf16_to_bf8_{BLOCK_ID}"),
    pytest.param("bf16", "bf4", id=f"bf16_to_bf4_{BLOCK_ID}"),
]

TYPECAST_KERNEL_TEMPLATE = '''
import torch
import ttnn
import ttl

BLOCK_ROWS = {block_rows}
BLOCK_COLS = {block_cols}
BLOCK_SHAPE = (BLOCK_ROWS, BLOCK_COLS)

@ttl.operation(grid=(1, 1))
def typecast_kernel(inp, out):
    """Elementwise typecast to {dst_name} on a {block_rows}x{block_cols} tile block."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=BLOCK_SHAPE, block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=BLOCK_SHAPE, block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(ttl.math.typecast(x, {typecast_target}))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0:BLOCK_ROWS, 0:BLOCK_COLS], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:BLOCK_ROWS, 0:BLOCK_COLS])
            tx.wait()
'''

_kernel_cache = {}
_temp_files = []


def _make_typecast_kernel(dst_name):
    """Generate and cache a multi-tile typecast kernel for the destination dtype."""
    if dst_name in _kernel_cache:
        return _kernel_cache[dst_name]

    code = TYPECAST_KERNEL_TEMPLATE.format(
        dst_name=dst_name,
        typecast_target=_DTYPE_SPECS[dst_name]["typecast_target"],
        block_rows=BLOCK_ROWS,
        block_cols=BLOCK_COLS,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"typecast_{BLOCK_ROWS}x{BLOCK_COLS}_{dst_name}_",
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location(
        f"typecast_{BLOCK_ROWS}x{BLOCK_COLS}_{dst_name}_module", temp_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.typecast_kernel
    _kernel_cache[dst_name] = kernel
    return kernel


def _cleanup_temp_files():
    for path in _temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass


atexit.register(_cleanup_temp_files)


def _to_l1_with_dtype(torch_tensor, device, ttnn_dtype):
    """Create an L1 tiled TTNN tensor with an explicit TTNN dtype."""
    dram_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_memory_config(dram_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)


def _reference_typecast(inp, out_ttnn_dtype, out_torch_dtype, device):
    """Build a torch golden that includes TTNN quantization for bf4/bf8."""
    source = ttnn.to_torch(inp)
    if out_ttnn_dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b):
        quantized = _to_l1_with_dtype(source, device, out_ttnn_dtype)
        return ttnn.to_torch(quantized).float()
    return source.to(out_torch_dtype).float()


def _combined_tolerance(*specs):
    """Use the loosest tolerance among the participating formats."""
    return {
        "rtol": max(spec["rtol"] for spec in specs),
        "atol": max(spec["atol"] for spec in specs),
    }


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("src_name, dst_name", _MULTI_TILE_SWEEP_CONFIGS)
def test_typecast_dtype_sweep_multi_tile(device, src_name, dst_name):
    """Exercise selected dtype conversions on a multi-tile block."""
    src_spec = _DTYPE_SPECS[src_name]
    dst_spec = _DTYPE_SPECS[dst_name]
    kernel = _make_typecast_kernel(dst_name)
    shape = (BLOCK_ROWS * TILE, BLOCK_COLS * TILE)

    inp_torch = torch.rand(shape, dtype=src_spec["torch_dtype"])
    out_torch = torch.zeros(shape, dtype=dst_spec["torch_dtype"])

    inp = _to_l1_with_dtype(inp_torch, device, src_spec["ttnn_dtype"])
    out = _to_l1_with_dtype(out_torch, device, dst_spec["ttnn_dtype"])

    kernel(inp, out)

    result = ttnn.to_torch(out).float()
    expected = _reference_typecast(
        inp, dst_spec["ttnn_dtype"], dst_spec["torch_dtype"], device
    )

    tol = _combined_tolerance(src_spec, dst_spec)
    assert_allclose(result, expected, rtol=tol["rtol"], atol=tol["atol"])
    max_diff = (result - expected).abs().max().item()
    print(f"  [{src_name}->{dst_name} {BLOCK_ID}] max_diff={max_diff:.2e}  PASSED")


def test_typecast_mixed_bf16_f32_long_fusion_multi_tile(device):
    """Cast bf16 to f32 and use it in a longer multi-tile fusion."""
    shape = (BLOCK_ROWS * TILE, BLOCK_COLS * TILE)
    a_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(shape, dtype=torch.float32)

    a = _to_l1_with_dtype(a_torch, device, ttnn.bfloat16)
    out = _to_l1_with_dtype(out_torch, device, ttnn.float32)

    _typecast_bf16_f32_long_fusion_multi_tile(a, out)

    a_ref = ttnn.to_torch(a).float()
    result = ttnn.to_torch(out).float()
    expected = ((a_ref + a_ref) * a_ref) + a_ref
    assert_allclose(result, expected, rtol=5e-3, atol=1e-2)


def test_typecast_mixed_bf16_f32_stray_f32_long_fusion_rejected(device):
    """Reject a fusion that mixes typecasted bf16 with an unrelated f32 input."""
    shape = (BLOCK_ROWS * TILE, BLOCK_COLS * TILE)
    a_torch = torch.rand(shape, dtype=torch.bfloat16)
    b_torch = torch.rand(shape, dtype=torch.float32)
    out_torch = torch.zeros(shape, dtype=torch.float32)

    a = _to_l1_with_dtype(a_torch, device, ttnn.bfloat16)
    b = _to_l1_with_dtype(b_torch, device, ttnn.float32)
    out = _to_l1_with_dtype(out_torch, device, ttnn.float32)

    with pytest.raises(
        RuntimeError,
        match="mixed f32 and non-f32 tile arguments",
    ):
        _typecast_bf16_f32_stray_f32_fusion_multi_tile(a, b, out)
