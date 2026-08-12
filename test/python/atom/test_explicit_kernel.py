# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python -m pytest %s -v

"""Off-device tests for logical selectors on explicit kernel decorators."""

import pytest
import ttl

from ttl.ttl_api import (
    _clear_thread_registry,
    _get_registered_threads,
    _validate_explicit_logical_kernel_uses,
)


def test_explicit_operation_binds_captured_kernel_declaration():
    """Operation registration binds the name used by a thread decorator."""
    data_movement_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(1, 1))
    def explicit_operation(inp):
        @ttl.compute()
        def compute_thread():
            pass

        @ttl.datamovement(kernel=data_movement_kernel)
        def data_movement_thread():
            pass

    assert data_movement_kernel.identity == "data_movement_kernel"
    assert data_movement_kernel._operation_identity.startswith(
        f"{__name__}.test_explicit_operation_binds_captured_kernel_declaration"
        ".<locals>.explicit_operation[captures="
    )

    _clear_thread_registry()
    explicit_operation.__wrapped__(object())
    threads = _get_registered_threads()
    assert [thread._logical_kernel for thread in threads] == [
        ttl.KernelKind.COMPUTE,
        data_movement_kernel,
    ]


def test_explicit_kernel_decorator_rejects_wrong_kind():
    """A thread decorator requires a selector of its declared kind."""
    with pytest.raises(
        ValueError,
        match="compute thread kernel kind must be compute, got data_movement",
    ):

        @ttl.compute(kernel=ttl.KernelKind.DATA_MOVEMENT)
        def compute_thread():
            pass


def test_explicit_kernel_decorator_rejects_wrong_named_kernel_kind():
    """A named selector's kind must match its thread decorator."""
    data_movement_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    with pytest.raises(
        ValueError,
        match="compute thread kernel kind must be compute, got data_movement",
    ):

        @ttl.compute(kernel=data_movement_kernel)
        def compute_thread():
            pass


def test_explicit_kernel_decorator_rejects_invalid_type():
    """A thread decorator accepts only one logical selector."""
    with pytest.raises(
        TypeError,
        match="kernel must be a KernelKind or Kernel, got tuple",
    ):

        @ttl.datamovement(kernel=(ttl.KernelKind.DATA_MOVEMENT,))
        def data_movement_thread():
            pass


def test_explicit_threads_reject_reused_named_kernel():
    """One named logical identity cannot denote two explicit threads."""
    shared_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(1, 1))
    def explicit_operation(inp):
        @ttl.datamovement(kernel=shared_kernel)
        def reader_thread():
            pass

        @ttl.datamovement(kernel=shared_kernel)
        def writer_thread():
            pass

    _clear_thread_registry()
    explicit_operation.__wrapped__(object())
    threads = _get_registered_threads()
    with pytest.raises(
        ValueError,
        match=(
            "logical Kernel 'shared_kernel' is selected by multiple explicit "
            "threads: 'reader_thread' and 'writer_thread'"
        ),
    ):
        _validate_explicit_logical_kernel_uses(threads)


def test_explicit_threads_use_target_kernel_capacity_diagnostic():
    """Explicit threads report the same logical capacity terms as splitting."""
    _clear_thread_registry()

    @ttl.compute()
    def first_compute():
        pass

    @ttl.compute()
    def second_compute():
        pass

    threads = _get_registered_threads()
    with pytest.raises(
        ValueError,
        match=(
            "operation requires 2 compute kernels, but the target supports 1; "
            "selected kernels: compute, compute"
        ),
    ):
        _validate_explicit_logical_kernel_uses(
            threads,
            {
                ttl.KernelKind.COMPUTE: 1,
                ttl.KernelKind.DATA_MOVEMENT: 2,
            },
        )
