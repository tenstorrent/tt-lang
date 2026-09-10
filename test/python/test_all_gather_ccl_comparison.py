# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The CCL comparator preserves full-workload allocations and timing scope."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from benchmarks.all_gather_minimal_matmul.ccl_comparison import (
    ActivationCollectiveWorkload,
    compare_activation_collective,
)


@dataclass
class FakeWorkload:
    run: object
    gathered: object
    cleanup: object
    program_count: int
    activation_collective: object


def make_workload():
    buffers = [object(), object()]
    collective = ActivationCollectiveWorkload(
        lambda: buffers, lambda *arguments: None, {"worker_count": 2}
    )
    return FakeWorkload(lambda: buffers, None, lambda output: None, 2, collective)


def test_comparison_reuses_instances_and_explicit_program_counts():
    full = make_workload()
    arguments = SimpleNamespace(trace=True)
    mesh = object()
    validate = object()
    observed = []
    checkpoints = []

    def measure(workloads, check, selected_mesh, selected_arguments):
        workload = workloads["ttlang"]
        assert selected_mesh is mesh
        assert selected_arguments is arguments
        assert workload.cleanup is full.cleanup
        assert workload.run() is full.run()
        if len(observed) == 1:
            assert workload.run is full.activation_collective.run
            assert check is full.activation_collective.validate
            assert workload.program_count == 1
        else:
            assert workload is full
            assert check is validate
            assert workload.program_count == 2
        observed.append(workload)
        return {"measurement": len(observed)}

    results = compare_activation_collective(
        {"ttlang": full},
        validate,
        mesh,
        arguments,
        measure=measure,
        record_stage=lambda name, result: checkpoints.append((name, result)),
    )
    assert list(results) == ["full_before", "isolated", "full_after"]
    assert checkpoints == list(results.items())


def test_completed_stages_checkpoint_before_later_failure():
    checkpoints = []

    def measure(*arguments):
        if checkpoints:
            raise RuntimeError("interrupted")
        return {"complete": True}

    with pytest.raises(RuntimeError, match="interrupted"):
        compare_activation_collective(
            {"ttlang": make_workload()},
            None,
            None,
            SimpleNamespace(trace=True),
            measure=measure,
            record_stage=lambda name, result: checkpoints.append((name, result)),
        )
    assert checkpoints == [("full_before", {"complete": True})]


@pytest.mark.parametrize(
    "condition", ["native", "no_collective", "one_program", "host_time"]
)
def test_rejects_incompatible_workloads(condition):
    full = make_workload()
    workloads = {"ttlang": full}
    arguments = SimpleNamespace(trace=True)
    if condition == "native":
        workloads["ttmetal"] = full
    elif condition == "no_collective":
        full.activation_collective = None
    elif condition == "one_program":
        full.program_count = 1
    else:
        arguments.trace = False

    def unexpected(*arguments):
        pytest.fail("invalid comparator must fail before measurement")

    with pytest.raises(ValueError):
        compare_activation_collective(
            workloads,
            None,
            None,
            arguments,
            measure=unexpected,
            record_stage=unexpected,
        )
