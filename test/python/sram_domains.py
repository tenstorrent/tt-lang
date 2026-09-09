# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# REQUIRES: ttnn, tt-device
# RUN: env TT_METAL_ALLOCATOR_MODE_HYBRID=1 TTLANG_COMPILER_OPTIONS=--ttl-sram-allocation-mode=per-core %python %s
# Verify independent SRAM allocation in a process configured before device opening.

import json
from pathlib import Path

import pytest
import torch
import ttl
from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_allclose, assert_pcc
from test_compiler_l1 import l1_copy, _make_allocation_stress, ALLOCATION_SCHEDULES

import ttnn

pytestmark = pytest.mark.requires_device


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "sram"])
def test_per_core_copy(device, dtype, allocator):
    expected = torch.randn(32, 32, dtype=dtype)
    source = allocator(expected, device)
    destination = allocator(torch.zeros_like(expected), device)
    for invocation in range(2):
        l1_copy(
            source,
            destination,
            options="--ttl-memory-model=compiler-l1 --ttl-sram-allocation-mode=per-core",
        )
        assert_allclose(
            ttnn.to_torch(destination).float(), expected.float(), rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_per_core_mixed_extents(device, dtype, tmp_path):
    operation, pages, capacities, conflicts = _make_allocation_stress(
        tmp_path, ALLOCATION_SCHEDULES[0], (2, 2)
    )
    expected = torch.randn(sum(pages) * 4 * 32, 32, dtype=dtype)
    source = to_dram(expected, device)
    destination = to_dram(torch.zeros_like(expected), device)
    operation(
        source,
        destination,
        options="--ttl-memory-model=compiler-l1 --ttl-sram-allocation-mode=per-core",
    )
    assert_allclose(
        ttnn.to_torch(destination).float(), expected.float(), rtol=0, atol=0
    )


@ttl.operation(grid=(2, 1))
def uneven_core_copy(source, destination):
    large = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=16)
    small = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def unused_transfer():
        pass

    @ttl.datamovement()
    def transfer():
        column, row = ttl.node(dims=2)
        if column == 0:
            with large.reserve() as block:
                ttl.copy(source[0, 0], block).wait()
            with large.wait() as block:
                ttl.copy(block, destination[0, 0]).wait()
        else:
            with small.reserve() as block:
                ttl.copy(source[1, 0], block).wait()
            with small.wait() as block:
                ttl.copy(block, destination[1, 0]).wait()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "sram"])
def test_per_core_uneven_reservation(device, dtype, allocator, monkeypatch, capfd):
    import ttl.kernel_runner as runner

    expected = torch.randn(64, 32, dtype=dtype)
    source = allocator(expected, device)
    destination = allocator(torch.zeros_like(expected), device)
    allocate_storage = runner._allocate_l1_sharded_storage_tensor
    allocations = []

    def retain_allocation(core_ranges, num_bytes, current_device, **kwargs):
        arena = allocate_storage(core_ranges, num_bytes, current_device, **kwargs)
        allocations.append((arena, num_bytes, core_ranges.num_cores()))
        return arena

    monkeypatch.setattr(
        runner, "_allocate_l1_sharded_storage_tensor", retain_allocation
    )
    # Explicit options take precedence over the process-wide test mode.
    monkeypatch.delenv("TTLANG_COMPILER_OPTIONS", raising=False)
    totals = {}
    for mode in ("uniform", "per-core"):
        allocations.clear()
        destination = allocator(torch.zeros_like(expected), device)
        capfd.readouterr()
        uneven_core_copy(
            source,
            destination,
            options=f"--ttl-memory-model=compiler-l1 --ttl-sram-allocation-mode={mode} --ttl-sram-allocation-report",
        )
        assert_allclose(
            ttnn.to_torch(destination).float(), expected.float(), rtol=0, atol=0
        )
        reports = [
            json.loads(line.split("ttlang-sram-report: ", 1)[1])
            for line in capfd.readouterr().err.splitlines()
            if line.startswith("ttlang-sram-report: ")
        ]
        runtime_reports = [record for record in reports if record["phase"] == "runtime"]
        compiler_reports = [
            record for record in reports if record["phase"] == "compiler"
        ]
        totals[mode] = sum(
            record["reserved_bytes_on_reference_device"] for record in runtime_reports
        )
        assert totals[mode] == sum(
            num_bytes * cores for arena, num_bytes, cores in allocations
        )
        assert len(compiler_reports) == len(runtime_reports) == len(allocations)
        if mode == "per-core":
            assert len(allocations) == 2
            assert all(arena.is_per_core_allocated() for arena, _, _ in allocations)
            assert len({num_bytes for _, num_bytes, _ in allocations}) == 2
    # The second core requires one tile instead of the first core's 16 tiles.
    assert (
        totals["uniform"] - totals["per-core"] == 15 * 32 * 32 * expected.element_size()
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "sram"])
def test_per_core_receiver_order_is_deterministic(
    device, dtype, allocator, monkeypatch, tmp_path
):
    from pipe.test_compiler_l1_pipenet import compiler_l1_pipe_matmul

    lhs_host = torch.randn(64, 64, dtype=dtype)
    rhs_host = torch.randn(64, 64, dtype=dtype)
    lhs = allocator(lhs_host, device)
    rhs = allocator(rhs_host, device)
    final_ir = tmp_path / "receivers.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_ir))
    layouts = []
    for compilation in range(3):
        operation = ttl.operation(grid=(2, 2))(compiler_l1_pipe_matmul.__wrapped__)
        output = allocator(torch.zeros_like(lhs_host), device)
        operation(
            lhs,
            rhs,
            output,
            options="--ttl-memory-model=compiler-l1 --ttl-sram-allocation-mode=per-core",
        )
        assert_pcc(
            lhs_host.float() @ rhs_host.float(),
            ttnn.to_torch(output).float(),
            threshold=0.999 if dtype == torch.bfloat16 else 0.99999,
        )
        targets = tuple(
            line.strip()
            for line in final_ir.read_text().splitlines()
            if "ttl.sram_receiver_targets" in line
        )
        assert targets and max(line.count("dfb_index") for line in targets) >= 2
        layouts.append(targets)
    assert layouts[0] == layouts[1] == layouts[2]


@ttl.operation(grid=(2, 1))
def core_sensitive_copy(source, destination):
    first = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=1)
    second = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def unused_transfer():
        pass

    @ttl.datamovement()
    def transfer():
        column, row = ttl.node(dims=2)
        offset = column * 2
        with first.reserve() as block:
            ttl.copy(source[offset, 0], block).wait()
        if column == 0:
            with first.wait() as block:
                ttl.copy(block, destination[offset, 0]).wait()
            with second.reserve() as block:
                ttl.copy(source[offset + 1, 0], block).wait()
        else:
            with second.reserve() as block:
                ttl.copy(source[offset + 1, 0], block).wait()
            with first.wait() as block:
                ttl.copy(block, destination[offset, 0]).wait()
        with second.wait() as block:
            ttl.copy(block, destination[offset + 1, 0]).wait()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "sram"])
@pytest.mark.parametrize(
    "strategy",
    ["first-fit-decreasing", "best-fit-decreasing", "multi-order-decreasing", "exact"],
)
@pytest.mark.parametrize("reuse", [True, False], ids=["reuse", "no-reuse"])
def test_per_core_temporal_reuse(
    device, dtype, allocator, strategy, reuse, monkeypatch, capfd
):
    # Both buffers execute on both cores; only core zero completes the first
    # buffer before acquiring the second. The two-vertex optimum is one page
    # on core zero and two pages on core one when reuse is enabled.
    monkeypatch.delenv("TTLANG_COMPILER_OPTIONS", raising=False)
    expected = torch.randn(128, 32, dtype=dtype)
    source = allocator(expected, device)
    totals = {}
    for mode in ("uniform", "per-core"):
        destination = allocator(torch.zeros_like(expected), device)
        capfd.readouterr()
        core_sensitive_copy(
            source,
            destination,
            options=f"--ttl-memory-model=compiler-l1 --ttl-sram-allocation-mode={mode} --ttl-l1-allocation-strategy={strategy} {'--ttl-reuse-user-dfbs' if reuse else '--no-ttl-reuse-user-dfbs'} --ttl-sram-allocation-report",
        )
        assert_allclose(
            ttnn.to_torch(destination).float(), expected.float(), rtol=0, atol=0
        )
        records = [
            json.loads(line.split("ttlang-sram-report: ", 1)[1])
            for line in capfd.readouterr().err.splitlines()
            if line.startswith("ttlang-sram-report: ")
        ]
        runtime = [record for record in records if record["phase"] == "runtime"]
        assert len(runtime) == (1 if mode == "uniform" else 2)
        totals[mode] = sum(
            record["reserved_bytes_on_reference_device"] for record in runtime
        )
    assert totals["uniform"] - totals["per-core"] == (
        32 * 32 * expected.element_size() if reuse else 0
    )


if __name__ == "__main__":
    directory = Path(__file__).parent
    tests = [
        str(Path(__file__)),
        str(directory / "pipe/test_compiler_l1_pipenet.py"),
        f"{directory / 'test_external_dfb_reuse.py'}::test_compiler_l1_external_composition_exceeds_metal_index_limit",
    ]
    for test_name in (
        "test_l1_rms_normalization",
        "test_l1_gated_mlp_residual",
        "test_l1_attention",
        "test_l1_expert_merge",
        "test_l1_compute_above_descriptor_limit",
    ):
        tests.append(f"{directory / 'test_compiler_l1_compute.py'}::{test_name}")
    # Synchronized lifecycle operations require the Blackhole synchronization LLK.
    if ttnn.get_arch_name() == "blackhole":
        tests.append(str(directory / "test_compiler_l1_lifecycle.py"))
    raise SystemExit(pytest.main([*tests, "-k", "not metal-cb", "-xq"]))
