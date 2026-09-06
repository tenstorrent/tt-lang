# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Exhaustive three- and four-region mixed-size schedules check byte-placement
# safety against execution events, independently of the compiler conflict graph.
# RUN: %python %s

import itertools
import re
import subprocess


FORMATS = [
    ("1x16", "bfp_bf4", 24, 1),
    ("2x16", "bfp_bf8", 48, 3),
    ("32x32", "f32", 4096, 1),
    ("32x32", "bf16", 2048, 2),
]


def make_module(events, architecture, unknown=False):
    count = len(events) // 2
    lines = [
        f"module attributes {{ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<{architecture}>}} {{",
        "func.func @schedule() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {",
    ]
    for region, (tile, dtype, _, capacity) in enumerate(FORMATS[:count]):
        lines.append(
            f"%storage_{region} = ttl.bind_cb {{cb_index = {region}, block_count = {capacity}}} {{dfb_id = {region} : index}} : !ttl.cb<[1, 1], !ttcore.tile<{tile}, {dtype}>, {capacity}>"
        )
    for event_index, event in enumerate(events):
        region, action = divmod(event, 2)
        tile, dtype, _, capacity = FORMATS[region]
        signature = f"<[1, 1], !ttcore.tile<{tile}, {dtype}>, {capacity}>"
        acquire, release = ("reserve", "push") if action == 0 else ("wait", "pop")
        lines += [
            f"%view_{event_index} = ttl.cb_{acquire} %storage_{region} : {signature} -> tensor<1x1x!ttcore.tile<{tile}, {dtype}>>",
            f"ttl.cb_{release} %storage_{region} : {signature}",
        ]
        if unknown and event_index == 1:
            lines.append(
                'ttl.opaque_call "unrelated" () {header = "unrelated.hpp", unknown_dfb_access} : () -> ()'
            )
    return "\n".join(lines + ["return", "}", "}"])


def run_compiler(modules, reuse):
    result = subprocess.run(
        [
            "ttlang-opt",
            "--split-input-file",
            f"-pass-pipeline=builtin.module(ttl-finalize-dfb-indices{{memory-model=compiler-l1 reuse-user-dfbs={str(reuse).lower()}}})",
        ],
        input="\n// -----\n".join(modules),
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def validate(output, events, architecture, reuse, unknown):
    count = len(events) // 2
    quantum = 32 if architecture == "wormhole_b0" else 64
    control_bytes = (count * 8 + quantum - 1) // quantum * quantum
    sizes = [
        ((page_bytes * capacity + quantum - 1) // quantum) * quantum
        for _, _, page_bytes, capacity in FORMATS[:count]
    ]
    actual_sizes = [
        int(value) for value in re.findall(r"l1_allocation_bytes = (\d+)", output)
    ]
    offsets = [int(value) for value in re.findall(r"l1_payload_offset = (\d+)", output)]
    states = [int(value) for value in re.findall(r"l1_offset = (\d+)", output)]
    arena_bytes = int(re.search(r"ttl.l1_arena_bytes = (\d+)", output).group(1))
    assert actual_sizes == sizes
    assert states == list(range(0, count * 8, 8))
    assert len(offsets) == count
    assert all(offset >= control_bytes and offset % quantum == 0 for offset in offsets)
    assert arena_bytes == max(offset + size for offset, size in zip(offsets, sizes))
    assert arena_bytes <= control_bytes + sum(sizes)
    live = set()
    peak_bytes = 0
    conflicts = set()
    for event in events:
        region, action = divmod(event, 2)
        if action == 0:
            conflicts.update(tuple(sorted((region, other))) for other in live)
            live.add(region)
            peak_bytes = max(peak_bytes, sum(sizes[other] for other in live))
        else:
            live.remove(region)
    assert arena_bytes >= control_bytes + peak_bytes
    if not reuse or unknown:
        conflicts = set(itertools.combinations(range(count), 2))
    for first, second in conflicts:
        assert (
            offsets[first] + sizes[first] <= offsets[second]
            or offsets[second] + sizes[second] <= offsets[first]
        ), (events, architecture, offsets, sizes)
    if not conflicts:
        assert arena_bytes == control_bytes + max(sizes)
    if not reuse:
        assert arena_bytes == control_bytes + sum(sizes)


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
    four_region_schedules = [
        events
        for events in itertools.permutations(range(8))
        if all(
            events.index(2 * region) < events.index(2 * region + 1)
            for region in range(4)
        )
    ]
    assert len(four_region_schedules) == 2520
    schedules.extend(four_region_schedules)
    cases = [
        (events, architecture, False)
        for architecture in ("wormhole_b0", "blackhole")
        for events in schedules
    ]
    cases += [
        ((0, 1, 2, 3, 4, 5), architecture, True)
        for architecture in ("wormhole_b0", "blackhole")
    ]
    modules = [make_module(*case) for case in cases]
    for reuse in (False, True):
        output = run_compiler(modules, reuse)
        assert output == run_compiler(modules, reuse), "placement is nondeterministic"
        results = [section for section in output.split("// -----") if section.strip()]
        assert len(results) == len(cases)
        for result, (events, architecture, unknown) in zip(results, cases):
            validate(result, events, architecture, reuse, unknown)
    print(f"Verified {len(cases) * 2} placements and deterministic repetition.")


if __name__ == "__main__":
    main()
