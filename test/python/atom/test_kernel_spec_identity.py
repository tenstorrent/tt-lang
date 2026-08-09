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
from ttl.kernel_runner import KernelSpec  # noqa: E402

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
