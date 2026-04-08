# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test for TTLANG_EMIT_RUNNER: compile a kernel, emit a runner file,
import it, and verify it can execute the kernel independently.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import os
import importlib.util
import tempfile

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttlang_test_utils import assert_allclose, to_l1


@ttl.operation(grid=(1, 1))
def add_kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with lhs_dfb.wait() as l, rhs_dfb.wait() as r, out_dfb.reserve() as o:
            o.store(l + r)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as blk:
            tx = ttl.copy(lhs[0, 0], blk)
            tx.wait()
        with rhs_dfb.reserve() as blk:
            tx = ttl.copy(rhs[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_emit_runner(device):
    """Compile with TTLANG_EMIT_RUNNER, import the runner, and execute it."""
    with tempfile.NamedTemporaryFile(suffix="_runner.py", delete=False) as f:
        runner_path = f.name

    try:
        os.environ["TTLANG_EMIT_RUNNER"] = runner_path

        lhs = to_l1(torch.full((32, 32), 2.0, dtype=torch.bfloat16), device)
        rhs = to_l1(torch.full((32, 32), 3.0, dtype=torch.bfloat16), device)
        out = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)

        # This compiles the kernel AND writes the runner file.
        add_kernel(lhs, rhs, out)

        # Verify the kernel itself worked.
        result = ttnn.to_torch(out)
        expected = torch.full((32, 32), 5.0, dtype=torch.bfloat16)
        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

        # Verify runner file was emitted and is importable.
        assert os.path.exists(runner_path), f"Runner not written to {runner_path}"

        spec = importlib.util.spec_from_file_location("emitted_runner", runner_path)
        runner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner_module)

        assert hasattr(runner_module, "run"), "Runner module missing run() function"
        assert hasattr(
            runner_module, "KERNEL_PATHS"
        ), "Runner module missing KERNEL_PATHS"
        assert hasattr(runner_module, "CB_CONFIGS"), "Runner module missing CB_CONFIGS"

        # Run the emitted runner on fresh output tensor.
        out2 = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)
        runner_module.run([lhs, rhs, out2])

        result2 = ttnn.to_torch(out2)
        assert_allclose(result2.float(), expected.float(), rtol=1e-2, atol=1e-2)

    finally:
        os.environ.pop("TTLANG_EMIT_RUNNER", None)
        if os.path.exists(runner_path):
            os.unlink(runner_path)
