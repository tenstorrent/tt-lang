# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host compilation of bidirectional PipeNet receives into DRAM regions."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl._src import ttl_ast
from ttl.compiler_options import CompilerOptions
from ttl.domains import DeviceDomain
from ttl.layouts import BUFFER_TYPE_DRAM
from ttl.ttl_api import _compile_kernel

from test_pipe_dram_destination import (
    _make_concurrent_bidirectional_direct_dram_receive,
)

pytestmark = pytest.mark.compile_only


@pytest.mark.parametrize("mesh_shape", [(2, 1), (32, 1)], ids=["two", "32"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_compile_bidirectional_dram_receive(monkeypatch, mesh_shape, dtype):
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    # Host tensors provide dtype/layout; DRAM metadata exercises distributed
    # page addressing without allocating device storage.
    monkeypatch.setattr(ttl_ast, "detect_buffer_type", lambda tensor: BUFFER_TYPE_DRAM)
    operation = _make_concurrent_bidirectional_direct_dram_receive(mesh_shape, 4)
    tensors = tuple(
        ttnn.from_torch(
            torch.zeros((128, 32)),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        for _index in range(3)
    )
    compiled = _compile_kernel(
        operation.__wrapped__,
        tensors,
        {},
        (1, 4),
        [],
        [],
        1,
        "DRAM",
        True,
        12345,
        target_arch="blackhole",
        compiler_options=CompilerOptions(),
        device_domain=DeviceDomain(mesh_shape),
    )
    assert compiled is not None
    assert compiled.kernel_paths
