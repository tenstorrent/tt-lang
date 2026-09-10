# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Measure the composed operation's activation collective without reallocating."""

from collections.abc import Callable
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ActivationCollectiveWorkload:
    run: Callable
    validate: Callable
    config: dict


def compare_activation_collective(
    workloads, validate, mesh, arguments, *, measure, record_stage
):
    """Retain the complete workload while timing its collective separately."""
    if set(workloads) != {"ttlang"}:
        raise ValueError("activation CCL comparison requires only TT-Lang")
    full = workloads["ttlang"]
    collective = full.activation_collective
    if collective is None or full.program_count != 2:
        raise ValueError(
            "activation CCL comparison requires multi-device replicated matmul"
        )
    if not arguments.trace:
        raise ValueError("activation CCL comparison requires device trace replay")
    isolated = replace(
        full,
        run=collective.run,
        gathered=None,
        program_count=1,
        activation_collective=None,
    )
    stages = (
        ("full_before", full, validate),
        ("isolated", isolated, collective.validate),
        ("full_after", full, validate),
    )
    results = {}
    for stage_name, workload, check in stages:
        print(f"Activation CCL comparison: {stage_name}", flush=True)
        result = measure({"ttlang": workload}, check, mesh, arguments)
        results[stage_name] = result
        record_stage(stage_name, result)
    return results
