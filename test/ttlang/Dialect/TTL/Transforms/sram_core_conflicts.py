# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s
# Compare per-core conflicts and exact placement with independent event schedules.

import importlib.util
import itertools
import json
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location(
    "allocation_stress", Path(__file__).with_name("compiler_l1_stress.py")
)
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)


def conflicts_for(events):
    live = set()
    conflicts = set()
    for event in events:
        region, action = divmod(event, 2)
        if action == 0:
            conflicts.update(tuple(sorted((region, other))) for other in live)
            live.add(region)
        else:
            live.remove(region)
    assert not live
    return conflicts


def make_module(schedules, architecture, unknown):
    lines = [
        f"module attributes {{ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<{architecture}>}} {{",
        "func.func @schedule() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {",
    ]
    for region, (tile, dtype, _, capacity) in enumerate(stress.FORMATS[:3]):
        lines.append(
            f"%storage_{region} = ttl.bind_cb {{cb_index = {region}, block_count = {capacity}}} {{dfb_id = {region} : index}} : !ttl.cb<[1, 1], !ttcore.tile<{tile}, {dtype}>, {capacity}>"
        )
    lines += [
        "%column = ttl.core_x : index",
        "%zero = arith.constant 0 : index",
        "%first = arith.cmpi eq, %column, %zero : index",
        "scf.if %first {",
    ]
    for core, events in enumerate(schedules):
        if core:
            lines.append("} else {")
        for event_index, event in enumerate(events):
            region, action = divmod(event, 2)
            tile, dtype, _, capacity = stress.FORMATS[region]
            signature = f"<[1, 1], !ttcore.tile<{tile}, {dtype}>, {capacity}>"
            acquire, release = ("reserve", "push") if action == 0 else ("wait", "pop")
            lines += [
                f"%view_{core}_{event_index} = ttl.cb_{acquire} %storage_{region} : {signature} -> tensor<1x1x!ttcore.tile<{tile}, {dtype}>>",
                f"ttl.cb_{release} %storage_{region} : {signature}",
            ]
            if unknown and event_index == 1:
                lines.append(
                    'ttl.opaque_call "unrelated" () {header = "unrelated.hpp", unknown_dfb_access} : () -> ()'
                )
    return "\n".join(lines + ["}", "return", "}", "}"])


def run(cases, mode, strategy, reuse):
    result = subprocess.run(
        [
            "ttlang-opt",
            "--split-input-file",
            "-pass-pipeline=builtin.module(ttl-finalize-dfb-indices{"
            + f"memory-model=compiler-l1 sram-allocation-mode={mode} sram-allocation-report=true l1-allocation-strategy={strategy} reuse-user-dfbs={str(reuse).lower()}"
            + "})",
        ],
        input="\n// -----\n".join(make_module(*case) for case in cases),
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    prefix = "ttlang-sram-report: "
    return result.stdout, [
        json.loads(line[len(prefix) :])
        for line in result.stderr.splitlines()
        if line.startswith(prefix)
    ]


def main():
    schedules = [
        events
        for events in itertools.permutations(range(6))
        if all(
            events.index(2 * region) < events.index(2 * region + 1)
            for region in range(3)
        )
    ]
    assert len(schedules) == 90
    # Pair every valid schedule with a differently ordered schedule, and include
    # both core assignments so a conflict on either core must be respected.
    pairs = [
        (events, schedules[(index + 37) % len(schedules)])
        for index, events in enumerate(schedules)
    ]
    pairs += [(second, first) for first, second in pairs]
    cases = [
        (pair, architecture, False)
        for architecture in ("blackhole", "wormhole_b0")
        for pair in pairs
    ]
    cases += [
        ((schedules[0], schedules[0]), architecture, True)
        for architecture in ("blackhole", "wormhole_b0")
    ]
    assert len(cases) == 362
    checked = 0
    for strategy in (
        "first-fit-decreasing",
        "best-fit-decreasing",
        "multi-order-decreasing",
        "exact",
    ):
        for mode in ("uniform", "per-core"):
            for reuse in (False, True):
                output, reports = run(cases, mode, strategy, reuse)
                assert (output, reports) == run(cases, mode, strategy, reuse)
                domains = 1 if mode == "uniform" else 2
                assert len(reports) == len(cases) * domains
                for case_index, (pair, architecture, unknown) in enumerate(cases):
                    quantum = 64 if architecture == "blackhole" else 32
                    sizes = [
                        (page_bytes * capacity + quantum - 1) // quantum * quantum
                        for _, _, page_bytes, capacity in stress.FORMATS[:3]
                    ]
                    per_core = [conflicts_for(events) for events in pair]
                    for domain in range(domains):
                        report = reports[case_index * domains + domain]
                        expected = (
                            per_core[0] | per_core[1]
                            if mode == "uniform"
                            else per_core[domain]
                        )
                        if unknown:
                            expected = set(itertools.combinations(range(3), 2))
                        reported = {
                            tuple(sorted(entry["logical_dfbs"]))
                            for entry in report["logical_conflicts"]
                        }
                        assert reported == expected, (
                            case_index,
                            mode,
                            domain,
                            reported,
                            expected,
                        )
                        if not reuse:
                            expected = set(itertools.combinations(range(3), 2))
                        owners = report["owners"]
                        assert len(owners) == 3
                        offsets = [owner["arena_payload_offset"] for owner in owners]
                        for first, second in expected:
                            assert (
                                offsets[first] + sizes[first] <= offsets[second]
                                or offsets[second] + sizes[second] <= offsets[first]
                            )
                        control = report["control_and_padding_bytes"]
                        if strategy == "exact":
                            assert report[
                                "arena_bytes_per_core"
                            ] == control + stress.minimum_payload_bytes(sizes, expected)
                        assert report["arena_bytes_per_core"] == max(
                            offset + size for offset, size in zip(offsets, sizes)
                        )
                        checked += 1
    assert checked == 8688
    print(f"core-specific placement cases={checked}")


if __name__ == "__main__":
    main()
