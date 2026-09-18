# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python -m pytest %s -v

"""Every synthesized backend kernel reaches KernelSpec with a logical identity.

A unified operation always emits one kernel per backend slot, including the
mandatory empty ones the plan never assigned. Runtime resources select kernels by
logical identity, so an empty kernel must still carry one.
"""

import inspect

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import torch  # noqa: E402
import ttl  # noqa: E402
from ttl.kernel_runner import FabricManagerIntervalKind, KernelSpec  # noqa: E402

HEADER = "/dev/null/fake_shim.hpp"


def _compiled_kernel(operation):
    """Return the single compiled artifact cached by an operation wrapper."""
    cache = inspect.getclosurevars(operation._wrapper).nonlocals["cache"]
    assert len(cache) == 1
    return next(iter(cache.values()))


def _kernel_specs(compiled):
    """Rebuild the specs the runner hands to the device, without a device."""
    return [
        KernelSpec(
            path=path,
            thread_type=thread_type,
            tensor_indices=compiled.kernel_tensor_indices[index],
            config=compiled.kernel_configs[index],
            logical_kernel=compiled.kernel_logical_selectors[index],
            fabric_manager_intervals=compiled.kernel_fabric_manager_intervals[index],
        )
        for index, (path, thread_type) in enumerate(compiled.kernel_paths)
    ]


def test_every_emitted_kernel_spec_has_a_logical_identity(monkeypatch):
    """One explicit selector still leaves every other slot identifiable."""
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(1, 1))
    def single_selected_reader(inp):
        ttl.call_extern_func(HEADER, "reader_entry", kernel=reader)

    single_selected_reader(
        ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    )

    specs = _kernel_specs(_compiled_kernel(single_selected_reader))

    assert specs
    assert all(spec.logical_kernel is not None for spec in specs)
    assert [spec.logical_kernel for spec in specs].count(reader) == 1


def test_external_fabric_manager_effects_reach_the_selected_kernel(monkeypatch):
    """Acquire, use, and release produce one external ownership interval."""
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    manager = ttl.FabricManagerClaim("external", kernel=reader)

    @ttl.operation(grid=(1, 1))
    def external_manager(inp):
        ttl.call_extern_func(
            HEADER,
            "open",
            kernel=reader,
            fabric_manager_effects=(manager.acquire(),),
        )
        ttl.call_extern_func(
            HEADER,
            "use",
            kernel=reader,
            fabric_manager_effects=(manager.use(),),
        )
        ttl.call_extern_func(
            HEADER,
            "close",
            kernel=reader,
            fabric_manager_effects=(manager.release(),),
        )

    external_manager(
        ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    )

    selected_spec = next(
        spec
        for spec in _kernel_specs(_compiled_kernel(external_manager))
        if spec.logical_kernel == reader
    )
    assert len(selected_spec.fabric_manager_intervals) == 1
    interval = selected_spec.fabric_manager_intervals[0]
    assert interval.kind is FabricManagerIntervalKind.EXTERNAL
    assert interval.claim == manager.identity


def test_external_fabric_manager_can_select_pipe_source_kernel(monkeypatch):
    """An external manager may share the compiler-owned PipeNet source."""
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    manager = ttl.FabricManagerClaim("external", kernel=ttl.PIPE_SOURCE_KERNEL)

    @ttl.operation(grid=(1, 1))
    def external_pipe_source_manager(inp):
        ttl.call_extern_func(
            HEADER,
            "open_and_close",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(manager.scoped(),),
        )

    external_pipe_source_manager(
        ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    )

    selected_spec = next(
        spec
        for spec in _kernel_specs(_compiled_kernel(external_pipe_source_manager))
        if spec.logical_kernel == ttl.PIPE_SOURCE_KERNEL
    )
    assert len(selected_spec.fabric_manager_intervals) == 1
    interval = selected_spec.fabric_manager_intervals[0]
    assert interval.kind is FabricManagerIntervalKind.EXTERNAL
    assert interval.claim == manager.identity


def test_scoped_external_managers_retain_conditional_launch_domain(monkeypatch):
    """Sequential coordinator calls can reuse one physical manager."""
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    pre_manager = ttl.FabricManagerClaim("pre", kernel=ttl.PIPE_SOURCE_KERNEL)
    post_manager = ttl.FabricManagerClaim("post", kernel=ttl.PIPE_SOURCE_KERNEL)

    @ttl.operation(grid=(3, 2))
    def conditional_managers(inp):
        core_x, core_y = ttl.node(dims=2)
        active = core_x == 1 and core_y == 0
        if active:
            ttl.call_extern_func(
                HEADER,
                "pre",
                kernel=ttl.PIPE_SOURCE_KERNEL,
                fabric_manager_effects=(pre_manager.scoped(),),
            )
        if active:
            ttl.call_extern_func(
                HEADER,
                "post",
                kernel=ttl.PIPE_SOURCE_KERNEL,
                fabric_manager_effects=(post_manager.scoped(),),
            )

    conditional_managers(
        ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    )

    selected_spec = next(
        spec
        for spec in _kernel_specs(_compiled_kernel(conditional_managers))
        if spec.logical_kernel == ttl.PIPE_SOURCE_KERNEL
    )
    assert len(selected_spec.fabric_manager_intervals) == 2
    intervals_by_claim = {
        interval.claim: interval for interval in selected_spec.fabric_manager_intervals
    }
    assert set(intervals_by_claim) == {pre_manager.identity, post_manager.identity}
    assert all(
        interval.launch_nodes == ((1, 0),) and interval.interfering_intervals == ()
        for interval in intervals_by_claim.values()
    )


def test_scoped_external_manager_retains_empty_launch_domain(monkeypatch):
    """An unreachable scoped call must not acquire on the full kernel range."""
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    manager = ttl.FabricManagerClaim("empty", kernel=ttl.PIPE_SOURCE_KERNEL)

    @ttl.operation(grid=(3, 2))
    def unreachable_manager(inp):
        core_x, core_y = ttl.node(dims=2)
        if core_x == 99:
            if core_y == 99:
                ttl.call_extern_func(
                    HEADER,
                    "unreachable",
                    kernel=ttl.PIPE_SOURCE_KERNEL,
                    fabric_manager_effects=(manager.scoped(),),
                )

    unreachable_manager(
        ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    )

    selected_spec = next(
        spec
        for spec in _kernel_specs(_compiled_kernel(unreachable_manager))
        if spec.logical_kernel == ttl.PIPE_SOURCE_KERNEL
    )
    assert len(selected_spec.fabric_manager_intervals) == 1
    assert selected_spec.fabric_manager_intervals[0].launch_nodes == ()


def test_composed_fabric_manager_lifetime_reaches_selected_kernel(monkeypatch):
    """Composed acquire, use, and release produce one ownership interval."""
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    manager = ttl.FabricManagerClaim("external", kernel=ttl.PIPE_SOURCE_KERNEL)

    @ttl.operation()
    def open_manager():
        ttl.call_extern_func(
            HEADER,
            "open",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(manager.acquire(),),
        )

    @ttl.operation()
    def use_manager():
        ttl.call_extern_func(
            HEADER,
            "use",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(manager.use(),),
        )

    @ttl.operation()
    def close_manager():
        ttl.call_extern_func(
            HEADER,
            "close",
            kernel=ttl.PIPE_SOURCE_KERNEL,
            fabric_manager_effects=(manager.release(),),
        )

    @ttl.operation(grid=(1, 1))
    def composed_manager(inp):
        open_manager()
        use_manager()
        close_manager()

    composed_manager(
        ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    )

    selected_spec = next(
        spec
        for spec in _kernel_specs(_compiled_kernel(composed_manager))
        if spec.logical_kernel == ttl.PIPE_SOURCE_KERNEL
    )
    assert len(selected_spec.fabric_manager_intervals) == 1
    interval = selected_spec.fabric_manager_intervals[0]
    assert interval.kind is FabricManagerIntervalKind.EXTERNAL
    assert interval.claim == manager.identity
