# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compiler-owned L1 transfer correctness and descriptor independence."""
import importlib.util
import re

import pytest
import torch
import ttl
from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_allclose

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device


@ttl.operation(grid=(1, 1))
def l1_copy(source, destination):
    storage = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=3)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def unused_transfer():
        pass

    @ttl.datamovement()
    def transfer():
        for iteration in range(7):
            with storage.reserve() as block:
                ttl.copy(source[0:1, 0:1], block).wait()
            with storage.wait() as block:
                ttl.copy(block, destination[0:1, 0:1]).wait()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_l1_copy(device, dtype, memory_model, allocator, monkeypatch):
    expected = torch.randn(32, 32, dtype=dtype)
    source = allocator(expected, device)
    destination = allocator(torch.zeros_like(expected), device)
    arenas = []
    if memory_model == "compiler-l1":
        import ttl.kernel_runner as runner

        allocate_storage = runner._allocate_l1_sharded_storage_tensor

        def retain_arena(*args, **kwargs):
            arena = allocate_storage(*args, **kwargs)
            arenas.append(arena)
            return arena

        monkeypatch.setattr(runner, "_allocate_l1_sharded_storage_tensor", retain_arena)

        def reject_descriptors(*args, **kwargs):
            pytest.fail("compiler-l1 constructed Metal DFB descriptors")

        monkeypatch.setattr(runner, "build_cb_descriptors", reject_descriptors)
    for invocation in range(3):
        l1_copy(source, destination, options=f"--ttl-memory-model={memory_model}")
        assert_allclose(
            ttnn.to_torch(destination).float(), expected.float(), rtol=0, atol=0
        )
    if memory_model == "compiler-l1":
        assert len({arena.buffer_address() for arena in arenas}) == 3
        for arena in arenas:
            words = ttnn.to_torch(arena).view(torch.int32).flatten()
            assert words[:2].tolist() == [1, 1]


def _make_many_buffers(tmp_path, count, simultaneous):
    lines = [
        "import ttl",
        "@ttl.operation(grid=(1, 1))",
        "def transfer(source, destination):",
    ]
    for buffer_index in range(count):
        lines.append(
            f"    storage_{buffer_index} = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=1)"
        )
    lines += [
        "    @ttl.compute()",
        "    def compute():",
        "        pass",
        "    @ttl.datamovement()",
        "    def unused_transfer():",
        "        pass",
        "    @ttl.datamovement()",
        "    def movement():",
    ]

    def reserve(buffer_index):
        return [
            f"        with storage_{buffer_index}.reserve() as block:",
            f"            ttl.copy(source[{buffer_index}:{buffer_index + 1}, 0:1], block).wait()",
        ]

    def consume(buffer_index):
        return [
            f"        with storage_{buffer_index}.wait() as block:",
            f"            ttl.copy(block, destination[{buffer_index}:{buffer_index + 1}, 0:1]).wait()",
        ]

    if simultaneous:
        for buffer_index in range(count):
            lines.extend(reserve(buffer_index))
        for buffer_index in range(count):
            lines.extend(consume(buffer_index))
    else:
        for buffer_index in range(count):
            lines.extend(reserve(buffer_index) + consume(buffer_index))
    source_file = tmp_path / "many_buffers.py"
    source_file.write_text("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location("many_buffers", source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.transfer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("simultaneous", [False, True], ids=["reuse", "96_live"])
@pytest.mark.parametrize("specialize", [False, True], ids=["generic", "specialized"])
def test_many_buffers(device, dtype, simultaneous, specialize, tmp_path, monkeypatch):
    count = 96
    operation = _make_many_buffers(tmp_path, count, simultaneous)
    expected = torch.randn(count * 32, 32, dtype=dtype)
    source = to_dram(expected, device)
    destination = to_dram(torch.zeros_like(expected), device)
    final_ir = tmp_path / "final.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_ir))
    for invocation in range(2):
        options = "--ttl-memory-model=compiler-l1"
        if specialize:
            options += " --ttl-specialize-cores"
        operation(source, destination, options=options)
        assert_allclose(
            ttnn.to_torch(destination).float(), expected.float(), rtol=0, atol=0
        )
    ir = final_ir.read_text()
    offsets = [int(value) for value in re.findall(r"l1_payload_offset = (\d+)", ir)]
    assert len(offsets) == count
    assert len(set(offsets)) == (count if simultaneous else 1)


def _make_l1_cross_processor(block_count, pages_per_block):
    @ttl.operation(grid=(1, 1))
    def l1_cross_processor(source, destination):
        storage = ttl.make_dataflow_buffer_like(
            source, shape=(pages_per_block, 1), block_count=block_count
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def producer():
            for tile_index in range(32):
                with storage.reserve() as block:
                    ttl.copy(
                        source[
                            tile_index
                            * pages_per_block : (tile_index + 1)
                            * pages_per_block,
                            0:1,
                        ],
                        block,
                    ).wait()

        @ttl.datamovement()
        def consumer():
            for tile_index in range(32):
                with storage.wait() as block:
                    ttl.copy(
                        block,
                        destination[
                            tile_index
                            * pages_per_block : (tile_index + 1)
                            * pages_per_block,
                            0:1,
                        ],
                    ).wait()

    return l1_cross_processor


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("block_count", [1, 2, 3, 5])
@pytest.mark.parametrize("pages_per_block", [1, 2])
def test_l1_cross_processor(device, dtype, block_count, pages_per_block):
    l1_cross_processor = _make_l1_cross_processor(block_count, pages_per_block)
    expected = torch.randn(32 * 32 * pages_per_block, 32, dtype=dtype)
    source = to_dram(expected, device)
    destination = to_dram(torch.zeros_like(expected), device)
    for invocation in range(3):
        l1_cross_processor(
            source, destination, options="--ttl-memory-model=compiler-l1"
        )
        assert_allclose(
            ttnn.to_torch(destination).float(), expected.float(), rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("initial_sequence", range(6))
def test_l1_sequence_wrap(device, monkeypatch, dtype, initial_sequence):
    expected = torch.randn(32, 32, dtype=dtype)
    source = to_dram(expected, device)
    destination = to_dram(torch.zeros_like(expected), device)
    original_from_torch = ttnn.from_torch
    arenas = []

    def initialize_near_wrap(value, **kwargs):
        # Seven transactions cross the modulo-six boundary from every initial state.
        words = value.view(torch.int32).flatten()
        words[0:2] = initial_sequence
        arena = original_from_torch(value, **kwargs)
        arenas.append(arena)
        return arena

    monkeypatch.setattr(ttnn, "from_torch", initialize_near_wrap)
    l1_copy(source, destination, options="--ttl-memory-model=compiler-l1")
    assert len(arenas) == 1
    words = ttnn.to_torch(arenas[0]).view(torch.int32).flatten()
    assert words[:2].tolist() == [(initial_sequence + 7) % 6] * 2
    assert_allclose(
        ttnn.to_torch(destination).float(), expected.float(), rtol=0, atol=0
    )


# Even events acquire a region; the following odd event releases that region.
ALLOCATION_SCHEDULES = (
    (6, 2, 12, 14, 0, 15, 4, 10, 8, 1, 9, 3, 13, 11, 7, 5),
    (14, 15, 6, 10, 12, 2, 4, 0, 8, 3, 9, 1, 11, 5, 7, 13),
    (2, 14, 3, 6, 0, 15, 7, 1, 12, 10, 4, 8, 11, 5, 9, 13),
    (10, 8, 4, 12, 0, 11, 9, 6, 5, 1, 2, 14, 15, 13, 7, 3),
)


def _make_allocation_stress(tmp_path, schedule, grid):
    pages = [1, 4, 2, 3, 1, 5, 2, 4]
    capacities = [1, 1, 3, 1, 2, 1, 2, 1]
    assert sorted(schedule) == list(range(2 * len(pages)))
    active = set()
    events = []
    conflicts = set()
    for event in schedule:
        region, action = divmod(event, 2)
        if action == 0:
            conflicts.update(tuple(sorted((region, other))) for other in active)
            active.add(region)
            events.append(("produce", region))
        else:
            active.remove(region)
            events.append(("consume", region))
    assert not active
    total_pages = sum(pages)
    lines = [
        "import ttl",
        f"@ttl.operation(grid={grid})",
        "def transfer(source, destination):",
    ]
    for region, (page_count, capacity) in enumerate(zip(pages, capacities)):
        lines.append(
            f"    storage_{region} = ttl.make_dataflow_buffer_like(source, shape=({page_count}, 1), block_count={capacity})"
        )
    lines += [
        "    @ttl.compute()",
        "    def compute():",
        "        pass",
        "    @ttl.datamovement()",
        "    def unused_transfer():",
        "        pass",
        "    @ttl.datamovement()",
        "    def movement():",
        "        node_x, node_y = ttl.node(dims=2)",
        f"        base = (node_y * {grid[0]} + node_x) * {total_pages}",
    ]
    for action, region in events:
        start = sum(pages[:region])
        end = start + pages[region]
        if action == "produce":
            lines += [
                f"        with storage_{region}.reserve() as block:",
                f"            ttl.copy(source[base + {start}:base + {end}, 0:1], block).wait()",
            ]
        else:
            lines += [
                f"        with storage_{region}.wait() as block:",
                f"            ttl.copy(block, destination[base + {start}:base + {end}, 0:1]).wait()",
            ]
    source_file = tmp_path / "allocation_stress.py"
    source_file.write_text("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location("allocation_stress", source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.transfer, pages, capacities, conflicts


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "schedule",
    ALLOCATION_SCHEDULES,
    ids=["schedule_0", "schedule_1", "schedule_2", "schedule_3"],
)
@pytest.mark.parametrize("grid", [(1, 1), (2, 2)], ids=["one_core", "four_cores"])
@pytest.mark.parametrize("reuse", [False, True], ids=["distinct", "reuse"])
def test_allocation_stress(device, dtype, schedule, grid, reuse, tmp_path, monkeypatch):
    operation, pages, capacities, conflicts = _make_allocation_stress(
        tmp_path, schedule, grid
    )
    expected = torch.randn(sum(pages) * grid[0] * grid[1] * 32, 32, dtype=dtype)
    source = to_dram(expected, device)
    destination = to_dram(torch.zeros_like(expected), device)
    final_ir = tmp_path / "stress.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_ir))
    options = "--ttl-memory-model=compiler-l1"
    if not reuse:
        options += " --no-ttl-reuse-user-dfbs"
    operation(source, destination, options=options)
    assert_allclose(
        ttnn.to_torch(destination).float(), expected.float(), rtol=0, atol=0
    )
    ir = final_ir.read_text()
    offsets = [int(value) for value in re.findall(r"l1_payload_offset = (\d+)", ir)]
    sizes = [int(value) for value in re.findall(r"l1_allocation_bytes = (\d+)", ir)]
    states = [int(value) for value in re.findall(r"l1_offset = (\d+)", ir)]
    arena_bytes = int(re.search(r"ttl.l1_arena_bytes = (\d+)", ir).group(1))
    assert len(offsets) == len(pages)
    assert states == list(range(0, len(pages) * 8, 8))
    # A 32-byte quantum is common to both supported architectures.
    assert all(offset % 32 == 0 and offset >= len(pages) * 8 for offset in offsets)
    assert sizes == [
        page_count * capacity * 1024 * expected.element_size()
        for page_count, capacity in zip(pages, capacities)
    ]
    assert arena_bytes == max(offset + size for offset, size in zip(offsets, sizes))
    assert arena_bytes <= len(pages) * 8 + sum(sizes)
    for first in range(len(pages)):
        for second in range(first + 1, len(pages)):
            if not reuse or (first, second) in conflicts:
                assert (
                    offsets[first] + sizes[first] <= offsets[second]
                    or offsets[second] + sizes[second] <= offsets[first]
                )
