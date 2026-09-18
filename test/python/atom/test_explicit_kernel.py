# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python -m pytest %s -v

"""Off-device tests for logical selectors on explicit kernel decorators."""

import pytest
import ttl

from ttl.diagnostics import TTLangCompileError
from ttl.ir import Context
from ttl.kernel import _operation_identity
from ttl.ttl_api import (
    _clear_thread_registry,
    _get_registered_threads,
    _validate_explicit_logical_kernel_uses,
)

_GLOBAL_READER = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
_GLOBAL_MANAGER = ttl.FabricManagerClaim("global_manager", _GLOBAL_READER)
_IDENTITY_READER_A = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
_IDENTITY_READER_B = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
_IDENTITY_MANAGER_A = ttl.FabricManagerClaim("manager_a", _IDENTITY_READER_A)
_IDENTITY_MANAGER_B = ttl.FabricManagerClaim("manager_b", _IDENTITY_READER_B)
_GLOBAL_IDENTITY_MANAGER = _IDENTITY_MANAGER_A


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


def test_explicit_thread_rejects_fabric_claim_for_another_kernel():
    """An external manager claim belongs to its selected logical kernel."""
    reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reader_manager = ttl.FabricManagerClaim("reader_manager", reader)

    @ttl.operation(grid=(1, 1))
    def explicit_operation(inp):
        @ttl.datamovement(kernel=reader)
        def reader_thread():
            pass

        @ttl.datamovement(kernel=writer)
        def writer_thread():
            ttl.call_extern_func(
                "manager.hpp",
                "open",
                fabric_manager_effects=(reader_manager.acquire(),),
            )

    _clear_thread_registry()
    explicit_operation.__wrapped__(object())
    threads = _get_registered_threads()
    writer_thread = next(
        thread for thread in threads if thread._logical_kernel == writer
    )
    with (
        Context(),
        pytest.raises(
            TTLangCompileError,
            match=(
                "fabric manager claim 'reader_manager' selects data_movement "
                "kernel 'reader', but the external call is compiled for "
                "data_movement kernel 'writer'"
            ),
        ),
    ):
        writer_thread()


def test_explicit_operation_binds_claim_referenced_only_by_nested_thread():
    """Nested thread globals participate in operation resource binding."""

    @ttl.operation(grid=(1, 1))
    def explicit_operation(inp):
        @ttl.datamovement(kernel=_GLOBAL_READER)
        def reader_thread():
            ttl.call_extern_func(
                "manager.hpp",
                "run",
                fabric_manager_effects=(_GLOBAL_MANAGER.scoped(),),
            )

    assert _GLOBAL_MANAGER.operation_identity == _GLOBAL_READER._operation_identity
    assert _GLOBAL_MANAGER.operation_identity == _operation_identity(
        explicit_operation.__wrapped__
    )


def test_nested_global_claim_contributes_to_operation_identity(monkeypatch):
    """Changing a nested global claim changes the operation cache identity."""

    def explicit_operation():
        def reader_thread():
            return _GLOBAL_IDENTITY_MANAGER.scoped()

        return reader_thread

    first_identity = _operation_identity(explicit_operation)
    monkeypatch.setitem(globals(), "_GLOBAL_IDENTITY_MANAGER", _IDENTITY_MANAGER_B)
    second_identity = _operation_identity(explicit_operation)

    assert first_identity != second_identity


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
