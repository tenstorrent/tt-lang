# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s
# Verify report accounting, conflicts, ownership, deterministic output, and unchanged IR.

import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "allocation_stress", HERE / "compiler_l1_stress.py"
)
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)
PREFIX = "ttlang-sram-report: "


def run(
    source,
    enabled=True,
    reuse=True,
    strategy="multi-order-decreasing",
    model="compiler-l1",
):
    result = subprocess.run(
        [
            "ttlang-opt",
            "--split-input-file",
            "-pass-pipeline=builtin.module(ttl-finalize-dfb-indices{"
            f"memory-model={model} sram-allocation-report={str(enabled).lower()} "
            f"reuse-user-dfbs={str(reuse).lower()} l1-allocation-strategy={strategy}"
            + "})",
        ],
        input=source,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    reports = [
        json.loads(line[len(PREFIX) :])
        for line in result.stderr.splitlines()
        if line.startswith(PREFIX)
    ]
    return result.stdout, reports


def verify(source, reuse=True, strategy="multi-order-decreasing"):
    baseline, silent = run(source, False, reuse, strategy)
    output, reports = run(source, True, reuse, strategy)
    repeated, repeated_reports = run(source, True, reuse, strategy)
    assert baseline == output == repeated
    assert silent == [] and reports == repeated_reports and reports
    for record in reports:
        owners = record["owners"]
        assert record["phase"] == "compiler" and record["schema_version"] == 1
        assert record["strategy"] == strategy and record["reuse_enabled"] == reuse
        assert record["control_record_bytes"] == len(owners) * 8
        assert (
            record["control_padding_bytes"]
            == record["control_and_padding_bytes"] - len(owners) * 8
        )
        intervals = [
            (owner["arena_payload_offset"], owner["arena_payload_bytes"])
            for owner in owners
            if owner["arena_payload_bytes"]
        ]
        used = {
            address
            for offset, size in intervals
            for address in range(offset, offset + size)
        }
        assert record["payload_union_bytes"] == len(used)
        assert record["payload_extent_sum_bytes"] == sum(
            size for offset, size in intervals
        )
        assert record["payload_reuse_bytes"] == sum(
            size for offset, size in intervals
        ) - len(used)
        assert record["payload_gap_bytes"] == record["payload_high_water_bytes"] - len(
            used
        )
        assert (
            record["arena_bytes_per_core"]
            == record["payload_high_water_bytes"] + record["control_and_padding_bytes"]
        )
        for overlap in record["reused_ranges"]:
            left, right = (owners[index] for index in overlap["owners"])
            assert overlap["offset"] == max(
                left["arena_payload_offset"], right["arena_payload_offset"]
            )
            assert (
                overlap["bytes"]
                == min(
                    left["arena_payload_offset"] + left["arena_payload_bytes"],
                    right["arena_payload_offset"] + right["arena_payload_bytes"],
                )
                - overlap["offset"]
            )
        if not reuse:
            assert record["payload_reuse_bytes"] == 0 and not record["reused_ranges"]
    return reports


for architecture in ("blackhole", "wormhole_b0"):
    sequential = stress.make_control_prefix_module(architecture, 4)
    for strategy in (
        "first-fit-decreasing",
        "best-fit-decreasing",
        "multi-order-decreasing",
        "exact",
    ):
        report = verify(sequential, strategy=strategy)[0]
        assert report["payload_reuse_bytes"] == 3 * 2048
        assert len(report["reused_ranges"]) == 6
        assert report["logical_conflicts"] == []
    verify(sequential, reuse=False)
    live = verify(stress.make_module((0, 2, 4, 6, 1, 3, 5, 7), architecture))[0]
    assert live["logical_conflicts"] and live["payload_reuse_bytes"] == 0
    assert all(conflict["reason"] for conflict in live["logical_conflicts"])
    assert all(lifetime["known_cores"] for lifetime in live["lifetimes"])

empty = verify("module {}", strategy="exact")[0]
assert empty["arena_bytes_per_core"] == 0 and empty["owners"] == []
groups = verify((HERE / "compiler_l1_allocation_groups.mlir").read_text())
assert any(
    len(owner["logical_dfbs"]) > 1 for record in groups for owner in record["owners"]
)
tensors = verify((HERE / "compiler_l1_tensor_backing.mlir").read_text())
assert any("tensor" in region for record in tensors for region in record["regions"])
assert any(record["payload_extent_sum_bytes"] == 0 for record in tensors)
verify(stress.make_module((0, 1, 2, 3), "blackhole", unknown=True))
_, metal_reports = run(
    stress.make_control_prefix_module("blackhole", 1), model="metal-cb"
)
assert metal_reports == []
print(
    "Verified SRAM report modes, accounting, ownership, conflicts, and deterministic unchanged IR."
)
