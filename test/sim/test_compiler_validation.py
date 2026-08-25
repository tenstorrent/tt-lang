# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the simulator's optional compiler-validation bridge."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from sim import compiler_validation
from sim.decorators import compute


class _FakeTensorSpec:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _FakeSimTensor:
    __module__ = "sim.ttnnsim"

    def __init__(self, shape=(32, 32), dtype="bfloat16"):
        self.shape = shape
        self.dtype = dtype


_FakeSimTensor.__name__ = "Tensor"


@pytest.fixture(autouse=True)
def reset_compiler_validation_configuration(monkeypatch):
    compiler_validation.configure("off", "blackhole")
    compiler_validation._warn_unavailable.cache_clear()
    yield
    compiler_validation._warn_unavailable.cache_clear()


@pytest.fixture
def validation_state(monkeypatch):
    compiler_module = types.ModuleType("ttl")
    calls = []

    def build_validator(function, **options):
        calls.append(("build", function, options))

        def validate(*args, **kwargs):
            calls.append(("validate", args, kwargs))

        return validate

    backend = compiler_validation._CompilerBackend(
        ttl_package=compiler_module,
        build_validator=build_validator,
        tensor_spec_type=_FakeTensorSpec,
    )
    load = compiler_validation._BackendLoad(backend, None)
    monkeypatch.setattr(compiler_validation, "_load_backend", lambda: load)
    compiler_validation.configure("required", "blackhole")
    return backend, calls


def _function_with_ttl_global(ttl_value):
    def template(tensor):
        return ttl_alias, tensor

    function = types.FunctionType(
        template.__code__,
        {"ttl_alias": ttl_value, "__name__": __name__},
        "operation",
    )
    return function


def test_off_mode_does_not_load_compiler(monkeypatch):
    load_attempted = False

    def load_backend():
        nonlocal load_attempted
        load_attempted = True
        raise AssertionError("compiler should not load")

    monkeypatch.setattr(compiler_validation, "_load_backend", load_backend)

    assert (
        compiler_validation.prepare_operation_validator(
            lambda tensor: None,
            grid=(1, 1),
            fp32_dest_acc_en=None,
            dst_full_sync_en=None,
            math_fidelity=None,
        )
        is None
    )
    assert load_attempted is False


def test_find_compiler_python_root_behind_source_tree(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    compiler_root = tmp_path / "build" / "python_packages"
    source_root.mkdir()
    extension_dir = compiler_root / "ttl" / "_mlir_libs"
    extension_dir.mkdir(parents=True)
    (extension_dir / "_ttlang.cpython-test.so").touch()
    monkeypatch.setattr(sys, "path", [str(source_root), str(compiler_root)])

    assert compiler_validation._find_compiler_python_root() == str(compiler_root)


def test_compiler_import_scope_restores_module_environment_and_path(monkeypatch):
    simulator_ttl = object()
    compiler_ttl = types.ModuleType("ttl")
    monkeypatch.setitem(sys.modules, "ttl", simulator_ttl)
    monkeypatch.setenv("TTLANG_SIM_ONLY", "1")
    original_path = list(sys.path)

    with pytest.raises(RuntimeError, match="scope failure"):
        with compiler_validation._compiler_import_scope(compiler_ttl, "/compiler"):
            assert sys.modules["ttl"] is compiler_ttl
            assert "TTLANG_SIM_ONLY" not in compiler_validation.os.environ
            assert sys.path[0] == "/compiler"
            raise RuntimeError("scope failure")

    assert sys.modules["ttl"] is simulator_ttl
    assert compiler_validation.os.environ["TTLANG_SIM_ONLY"] == "1"
    assert sys.path == original_path


def test_compiler_import_scope_removes_new_sim_only_environment_value(monkeypatch):
    compiler_ttl = types.ModuleType("ttl")
    monkeypatch.delenv("TTLANG_SIM_ONLY", raising=False)

    with compiler_validation._compiler_import_scope(compiler_ttl):
        compiler_validation.os.environ["TTLANG_SIM_ONLY"] = "compiler-value"

    assert "TTLANG_SIM_ONLY" not in compiler_validation.os.environ


def test_prepare_rebinds_ttl_and_forwards_compiler_options(
    monkeypatch, validation_state
):
    backend, calls = validation_state
    simulator_ttl = object()
    monkeypatch.setitem(sys.modules, "ttl", simulator_ttl)
    function = _function_with_ttl_global(simulator_ttl)

    validator = compiler_validation.prepare_operation_validator(
        function,
        grid=(2, 3),
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
        math_fidelity="HiFi4",
    )

    assert validator is not None
    _, compiler_function, options = calls[0]
    assert compiler_function.__globals__["ttl_alias"] is backend.ttl_package
    assert options == {
        "grid": (2, 3),
        "fp32_dest_acc_en": True,
        "dst_full_sync_en": False,
        "math_fidelity": "HiFi4",
        "target_arch": "blackhole",
    }
    assert sys.modules["ttl"] is simulator_ttl


def test_prepare_rebinds_from_imported_ttl_symbols(monkeypatch, validation_state):
    backend, calls = validation_state
    simulator_compute = object()
    compiler_compute = object()
    simulator_math = object()
    compiler_math = object()
    simulator_ttl = types.SimpleNamespace(
        compute=simulator_compute,
        math=simulator_math,
    )
    backend.ttl_package.compute = compiler_compute
    backend.ttl_package.math = compiler_math
    monkeypatch.setitem(sys.modules, "ttl", simulator_ttl)

    def template(tensor):
        return compute_alias, math_alias, tensor

    function = types.FunctionType(
        template.__code__,
        {
            "compute_alias": simulator_compute,
            "math_alias": simulator_math,
            "__name__": __name__,
        },
        "operation",
    )
    compiler_validation.prepare_operation_validator(
        function,
        grid=(1, 1),
        fp32_dest_acc_en=None,
        dst_full_sync_en=None,
        math_fidelity=None,
    )

    _, compiler_function, _ = calls[0]
    assert compiler_function.__globals__["compute_alias"] is compiler_compute
    assert compiler_function.__globals__["math_alias"] is compiler_math


def test_validator_converts_tensor_arguments_and_preserves_aliases(
    monkeypatch, validation_state
):
    _, calls = validation_state
    simulator_ttl = object()
    monkeypatch.setitem(sys.modules, "ttl", simulator_ttl)
    validator = compiler_validation.prepare_operation_validator(
        _function_with_ttl_global(simulator_ttl),
        grid=(1, 1),
        fp32_dest_acc_en=None,
        dst_full_sync_en=None,
        math_fidelity=None,
    )
    tensor = _FakeSimTensor((64, 96), "bfloat8_b")

    validator(tensor, output=tensor)

    _, args, kwargs = calls[1]
    assert isinstance(args[0], _FakeTensorSpec)
    assert args[0].shape == (64, 96)
    assert args[0].dtype == "bfloat8_b"
    assert kwargs["output"] is args[0]
    assert sys.modules["ttl"] is simulator_ttl


def test_validator_converts_simulator_tensor_subclasses(monkeypatch, validation_state):
    _, calls = validation_state
    simulator_ttl = object()
    monkeypatch.setitem(sys.modules, "ttl", simulator_ttl)
    validator = compiler_validation.prepare_operation_validator(
        _function_with_ttl_global(simulator_ttl),
        grid=(1, 1),
        fp32_dest_acc_en=None,
        dst_full_sync_en=None,
        math_fidelity=None,
    )

    class TensorSubclass(_FakeSimTensor):
        pass

    validator(TensorSubclass())

    _, args, _ = calls[1]
    assert isinstance(args[0], _FakeTensorSpec)


def test_direct_tensor_capture_is_converted_but_tuple_capture_is_not(
    monkeypatch, validation_state
):
    _, calls = validation_state
    simulator_ttl = object()
    monkeypatch.setitem(sys.modules, "ttl", simulator_ttl)
    captured_tensor = _FakeSimTensor()
    captured_tuple = (1, 2)

    def operation(tensor):
        return captured_tensor, captured_tuple, tensor

    compiler_validation.prepare_operation_validator(
        operation,
        grid=(1, 1),
        fp32_dest_acc_en=None,
        dst_full_sync_en=None,
        math_fidelity=None,
    )

    _, compiler_function, _ = calls[0]
    closure_values = tuple(cell.cell_contents for cell in compiler_function.__closure__)
    assert any(isinstance(value, _FakeTensorSpec) for value in closure_values)
    assert captured_tuple in closure_values


def test_tensor_capture_and_argument_preserve_identity(monkeypatch, validation_state):
    _, calls = validation_state
    simulator_ttl = object()
    monkeypatch.setitem(sys.modules, "ttl", simulator_ttl)
    captured_tensor = _FakeSimTensor()

    def operation(tensor):
        return captured_tensor, tensor

    validator = compiler_validation.prepare_operation_validator(
        operation,
        grid=(1, 1),
        fp32_dest_acc_en=None,
        dst_full_sync_en=None,
        math_fidelity=None,
    )
    validator(captured_tensor)

    _, compiler_function, _ = calls[0]
    captured_spec = compiler_function.__closure__[0].cell_contents
    _, args, _ = calls[1]
    assert args[0] is captured_spec


def test_required_mode_reports_unavailable_compiler(monkeypatch):
    unavailable = ModuleNotFoundError("ttl._mlir_libs._ttlang")
    load = compiler_validation._BackendLoad(None, unavailable)
    monkeypatch.setattr(compiler_validation, "_load_backend", lambda: load)

    with pytest.raises(
        compiler_validation.CompilerValidationUnavailable,
        match="full tt-lang compiler build",
    ):
        compiler_validation.configure("required", "blackhole")


def test_configuration_is_process_local(monkeypatch, validation_state):
    monkeypatch.delenv("TTLANG_SIM_COMPILER_VALIDATION", raising=False)
    monkeypatch.delenv("TTLANG_SIM_COMPILER_TARGET", raising=False)

    compiler_validation.configure("auto", "wormhole_b0")

    config = compiler_validation.get_context().config
    assert config.compiler_validation_mode == "auto"
    assert config.compiler_validation_target == "wormhole_b0"
    assert "TTLANG_SIM_COMPILER_VALIDATION" not in compiler_validation.os.environ
    assert "TTLANG_SIM_COMPILER_TARGET" not in compiler_validation.os.environ


def test_auto_mode_warns_and_continues_when_compiler_is_unavailable(monkeypatch):
    load = compiler_validation._BackendLoad(None, ImportError("compiler unavailable"))
    monkeypatch.setattr(compiler_validation, "_load_backend", lambda: load)
    compiler_validation._warn_unavailable.cache_clear()

    try:
        with pytest.warns(
            RuntimeWarning, match="compiler validation is unavailable"
        ) as seen:
            compiler_validation.configure("auto", "wormhole_b0")
            compiler_validation.configure("auto", "wormhole_b0")
        assert len(seen) == 1
    finally:
        compiler_validation._warn_unavailable.cache_clear()


def test_backend_loader_treats_broken_extension_as_unavailable(monkeypatch):
    compiler_validation._load_backend.cache_clear()
    monkeypatch.setattr(
        compiler_validation,
        "_initial_compiler_package",
        lambda: types.ModuleType("ttl"),
    )
    monkeypatch.setattr(
        compiler_validation.importlib,
        "reload",
        lambda _package: (_ for _ in ()).throw(OSError("dlopen failed")),
    )

    try:
        load = compiler_validation._load_backend()
        assert load.backend is None
        assert isinstance(load.error, OSError)
        assert str(load.error) == "dlopen failed"
    finally:
        compiler_validation._load_backend.cache_clear()


@pytest.mark.parametrize("mode", ["sometimes", "yes", ""])
def test_configure_rejects_unknown_modes(mode):
    with pytest.raises(ValueError, match="compiler validation mode"):
        compiler_validation.configure(mode, "blackhole")


def test_compiler_validation_runs_before_simulator(monkeypatch):
    operation_module = importlib.import_module("sim.operation")
    events = []

    def prepare(*_args, **_kwargs):
        return lambda *_args, **_kwargs: events.append("compiler")

    monkeypatch.setattr(operation_module, "prepare_operation_validator", prepare)
    program_module = importlib.import_module("sim.program")
    monkeypatch.setattr(
        program_module,
        "run_operation",
        lambda *_args, **_kwargs: events.append("simulator"),
    )

    @operation_module.operation(grid=(1, 1))
    def kernel(tensor):
        @compute()
        def compute_kernel():
            pass

    kernel(object())
    assert events == ["compiler", "simulator"]


def test_compiler_error_prevents_simulator_execution(monkeypatch):
    operation_module = importlib.import_module("sim.operation")

    def prepare(*_args, **_kwargs):
        def reject(*_args, **_kwargs):
            raise ValueError("compiler rejected kernel")

        return reject

    monkeypatch.setattr(operation_module, "prepare_operation_validator", prepare)
    program_module = importlib.import_module("sim.program")
    simulator_called = False

    def run_operation(*_args, **_kwargs):
        nonlocal simulator_called
        simulator_called = True

    monkeypatch.setattr(program_module, "run_operation", run_operation)

    @operation_module.operation(grid=(1, 1))
    def kernel(tensor):
        @compute()
        def compute_kernel():
            pass

    with pytest.raises(ValueError, match="compiler rejected kernel"):
        kernel(object())
    assert simulator_called is False
