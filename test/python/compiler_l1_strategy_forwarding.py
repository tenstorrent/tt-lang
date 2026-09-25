# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s

"""Verify that the Python SRAM strategy option changes a fragmented placement."""

import importlib.util
import os
import re
import tempfile
from pathlib import Path

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn


CAPACITIES = [10, 5, 8, 10, 7, 7]
# The lifetime order leaves a five-tile gap that best-fit uses while first-fit
# extends the arena.
EVENTS = [
    (4, "produce"),
    (0, "produce"),
    (3, "produce"),
    (2, "produce"),
    (0, "consume"),
    (1, "produce"),
    (2, "consume"),
    (5, "produce"),
    (1, "consume"),
    (3, "consume"),
    (5, "consume"),
    (4, "consume"),
]


def load_fragmented_operation(directory):
    # TTNN interop requires all three kernel threads; the reader owns the DFB
    # events that determine placement.
    source = ["import ttl", "@ttl.operation(grid=(1, 1))", "def fragmented(source):"]
    for region_index, capacity in enumerate(CAPACITIES):
        source.append(
            f"    storage_{region_index} = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count={capacity})"
        )
    source += [
        "    @ttl.compute()",
        "    def compute():",
        "        pass",
        "    @ttl.datamovement()",
        "    def reader():",
    ]
    for event_index, (region_index, action) in enumerate(EVENTS):
        acquire = "reserve" if action == "produce" else "wait"
        release = "push" if action == "produce" else "pop"
        source.append(
            f"        slot_{event_index} = storage_{region_index}.{acquire}()"
        )
        source.append(f"        slot_{event_index}.{release}()")
    source += ["    @ttl.datamovement()", "    def writer():", "        pass"]

    module_file = directory / "fragmented.py"
    module_file.write_text("\n".join(source) + "\n")
    specification = importlib.util.spec_from_file_location("fragmented", module_file)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.fragmented


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        operation = load_fragmented_operation(directory)
        host_tensor = ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        final_ir = directory / "final.mlir"
        os.environ["TTLANG_FINAL_MLIR"] = str(final_ir)
        arena_sizes = {}
        for strategy in ("first-fit-decreasing", "best-fit-decreasing"):
            operation(
                host_tensor,
                options=(
                    "--ttl-memory-model=compiler-sram "
                    f"--ttl-sram-allocation-strategy={strategy}"
                ),
            )
            arena_sizes[strategy] = int(
                re.search(r"ttl.l1_arena_bytes = (\d+)", final_ir.read_text()).group(1)
            )

        assert arena_sizes == {
            "first-fit-decreasing": 81984,
            "best-fit-decreasing": 71744,
        }
