# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s
# Compare per-node and multicast conflicts with independent event schedules.

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
    for region, (tile, dtype, _, capacity, pages) in enumerate(stress.FORMATS[:3]):
        lines.append(
            f"%storage_{region} = ttl.bind_cb {{cb_index = {region}, block_count = {capacity}}} {{dfb_id = {region} : index}} : !ttl.cb<[{pages}, 1], !ttcore.tile<{tile}, {dtype}>, {capacity}>"
        )
    lines += [
        "%column = ttl.core_x : index",
        "%zero = arith.constant 0 : index",
        "%first = arith.cmpi eq, %column, %zero : index",
        "scf.if %first {",
    ]
    for node, events in enumerate(schedules):
        if node:
            lines.append("} else {")
        for event_index, event in enumerate(events):
            region, action = divmod(event, 2)
            tile, dtype, _, capacity, pages = stress.FORMATS[region]
            signature = f"<[{pages}, 1], !ttcore.tile<{tile}, {dtype}>, {capacity}>"
            acquire, release = ("reserve", "push") if action == 0 else ("wait", "pop")
            lines += [
                f"%view_{node}_{event_index} = ttl.cb_{acquire} %storage_{region} : {signature} -> tensor<{pages}x1x!ttcore.tile<{tile}, {dtype}>>",
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
            + f"memory-model=compiler-sram sram-allocation-mode={mode} "
            + "sram-allocation-report=true "
            + f"sram-allocation-strategy={strategy} "
            + f"reuse-user-dfbs={str(reuse).lower()}"
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


def validate_multicast_receiver_conflict():
    # Both receivers use the same payload address. Only row one overlaps the
    # first and second scratch DFB lifetimes, so that conflict applies to both.
    module = """
module attributes {ttl.launch_grid = [2, 2], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @multicast() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %receiver = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 3, block_count = 1} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 1) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0> {
      %received = ttl.cb_reserve %receiver : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %receive = ttl.copy %pipe, %received : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %receiver : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %consumed = ttl.cb_wait %receiver : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %receiver : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %row = ttl.core_y : index
      %zero = arith.constant 0 : index
      %first_row = arith.cmpi eq, %row, %zero : index
      %first_written = ttl.cb_reserve %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      scf.if %first_row {
        %first_read = ttl.cb_wait %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        %second_written = ttl.cb_reserve %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      } else {
        %second_written = ttl.cb_reserve %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        %first_read = ttl.cb_wait %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      }
      %second_read = ttl.cb_wait %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0> {
      %produced = ttl.cb_reserve %source : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %source : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %send = ttl.copy %source, %pipe : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      %read = ttl.cb_wait %source : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %source : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    }
    return
  }
}
"""
    result = subprocess.run(
        [
            "ttlang-opt",
            "-pass-pipeline=builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-sram sram-allocation-mode=per-node sram-allocation-report=true reuse-user-dfbs=true})",
        ],
        input=module,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    prefix = "ttlang-sram-report: "
    reports = [
        json.loads(line[len(prefix) :])
        for line in result.stderr.splitlines()
        if line.startswith(prefix)
    ]
    assert len(reports) == 3
    receivers = next(
        report for report in reports if sorted(report["nodes"]) == [[1, 0], [1, 1]]
    )
    conflicting_nodes = {
        tuple(entry["node"])
        for entry in receivers["logical_conflicts"]
        if tuple(sorted(entry["logical_dfbs"])) == (2, 3)
    }
    assert conflicting_nodes == {(1, 1)}
    owners = {
        member: owner
        for owner in receivers["owners"]
        for member in owner["logical_dfbs"]
    }
    first = owners[2]
    second = owners[3]
    assert (
        first["arena_payload_offset"] + first["arena_payload_bytes"]
        <= second["arena_payload_offset"]
        or second["arena_payload_offset"] + second["arena_payload_bytes"]
        <= first["arena_payload_offset"]
    )


def main():
    validate_multicast_receiver_conflict()
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
    # both node assignments so a conflict on either node must be respected.
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
        "minimum-arena",
    ):
        for mode in ("uniform", "per-node"):
            for reuse in (False, True):
                output, reports = run(cases, mode, strategy, reuse)
                assert (output, reports) == run(cases, mode, strategy, reuse)
                domains = 1 if mode == "uniform" else 2
                assert len(reports) == len(cases) * domains
                for case_index, (pair, architecture, unknown) in enumerate(cases):
                    quantum = 64 if architecture == "blackhole" else 32
                    sizes = [
                        (page_bytes * capacity * pages + quantum - 1)
                        // quantum
                        * quantum
                        for _, _, page_bytes, capacity, pages in stress.FORMATS[:3]
                    ]
                    per_node = [conflicts_for(events) for events in pair]
                    for domain in range(domains):
                        report = reports[case_index * domains + domain]
                        expected = (
                            per_node[0] | per_node[1]
                            if mode == "uniform"
                            else per_node[domain]
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
                        if strategy == "minimum-arena":
                            assert report[
                                "arena_bytes_per_node"
                            ] == control + stress.minimum_payload_bytes(sizes, expected)
                        assert report["arena_bytes_per_node"] == max(
                            offset + size for offset, size in zip(offsets, sizes)
                        )
                        checked += 1
    assert checked == 8688
    print(f"node-specific placement cases={checked}")


if __name__ == "__main__":
    main()
