# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Optional bridge from the simulator to compiler static analysis."""

from __future__ import annotations

import functools
import importlib
import os
import sys
import threading
import types
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .context import get_context

_VALID_MODES = {"off", "auto", "required"}
_VALID_TARGETS = {"blackhole", "wormhole_b0"}
_COMPILER_SCOPE_LOCK = threading.RLock()


@dataclass(frozen=True)
class _CompilerBackend:
    ttl_package: Any
    build_validator: Callable
    tensor_spec_type: type


@dataclass(frozen=True)
class _BackendLoad:
    backend: Optional[_CompilerBackend]
    error: Optional[BaseException]


class CompilerValidationUnavailable(RuntimeError):
    """Raised when compiler validation was required but cannot be loaded."""


def _find_compiler_python_root() -> Optional[str]:
    """Find a full TTL package when the source simulator shadows the build."""
    for root in sys.path:
        extension_dir = Path(root) / "ttl" / "_mlir_libs"
        if any(extension_dir.glob("_ttlang*.so")):
            return root
    return None


def _initial_compiler_package(package=sys.modules.get("ttl")):
    """Retain the TTL package that existed before simulator shadowing."""
    return package


@contextmanager
def _compiler_import_scope(
    ttl_package: Any, compiler_python_root: Optional[str] = None
) -> Iterator[None]:
    """Temporarily restore the real TTL package while compiler code runs."""
    with _COMPILER_SCOPE_LOCK:
        original_ttl = sys.modules.get("ttl")
        original_path = list(sys.path)
        sim_only = os.environ.pop("TTLANG_SIM_ONLY", None)
        sys.modules["ttl"] = ttl_package
        if compiler_python_root is not None:
            sys.path.insert(0, compiler_python_root)
        try:
            yield
        finally:
            sys.path[:] = original_path
            if original_ttl is None:
                sys.modules.pop("ttl", None)
            else:
                sys.modules["ttl"] = original_ttl
            if sim_only is None:
                os.environ.pop("TTLANG_SIM_ONLY", None)
            else:
                os.environ["TTLANG_SIM_ONLY"] = sim_only


@functools.cache
def _load_backend() -> _BackendLoad:
    compiler_package = _initial_compiler_package()
    try:
        if compiler_package is None:
            compiler_package = importlib.import_module("ttl")
        compiler_python_root = _find_compiler_python_root()
        with _compiler_import_scope(compiler_package, compiler_python_root):
            compiler_package = importlib.reload(compiler_package)
            static_analysis = importlib.import_module("ttl.static_analysis")
        return _BackendLoad(
            _CompilerBackend(
                ttl_package=compiler_package,
                build_validator=static_analysis.build_operation_validator,
                tensor_spec_type=static_analysis.StaticTensorSpec,
            ),
            None,
        )
    except Exception as error:
        return _BackendLoad(None, error)


def _validate_configuration(mode: str, target_arch: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(
            f"compiler validation mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
        )
    if target_arch not in _VALID_TARGETS:
        raise ValueError(
            f"compiler target must be one of {sorted(_VALID_TARGETS)}, got {target_arch!r}"
        )


@functools.cache
def _warn_unavailable(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _report_unavailable(mode: str, load: _BackendLoad) -> None:
    detail = f": {load.error}" if load.error is not None else ""
    message = (
        "TT-Lang compiler validation is unavailable. Install or activate a full "
        f"tt-lang compiler build, or use --compiler-validation off{detail}"
    )
    if mode == "required":
        raise CompilerValidationUnavailable(message) from load.error
    _warn_unavailable(message)


def configure(mode: str, target_arch: str) -> None:
    """Configure optional compiler validation for subsequently defined ops."""
    _validate_configuration(mode, target_arch)
    config = get_context().config
    config.compiler_validation_mode = mode
    config.compiler_validation_target = target_arch
    if mode == "off":
        return

    load = _load_backend()
    if load.backend is not None:
        return
    _report_unavailable(mode, load)


def _is_sim_tensor(value: Any) -> bool:
    return any(
        base.__name__ == "Tensor" and base.__module__.endswith("sim.ttnnsim")
        for base in type(value).__mro__
    )


def _read_metadata(value: Any, name: str, default: Any = None) -> Any:
    """Read a public simulator/TTNN-style property or zero-argument method."""
    try:
        result = getattr(value, name)
        return result() if callable(result) else result
    except AttributeError:
        return default


def _enum_name(value: Any, default: str) -> str:
    if value is None:
        return default
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).upper()
    return str(value).rsplit(".", maxsplit=1)[-1].upper()


def _tuple_metadata(value: Any) -> Optional[tuple]:
    if value is None:
        return None
    return tuple(value)


def _core_ranges_metadata(core_ranges: Any) -> tuple[tuple[int, int, int, int], ...]:
    ranges = _read_metadata(core_ranges, "ranges", ())
    result = []
    for region in ranges or ():
        start = _read_metadata(region, "start")
        end = _read_metadata(region, "end")
        if start is None or end is None:
            continue
        result.append(
            (
                int(_read_metadata(start, "x")),
                int(_read_metadata(start, "y")),
                int(_read_metadata(end, "x")),
                int(_read_metadata(end, "y")),
            )
        )
    return tuple(result)


def _shard_grid_metadata(shard_spec: Any, memory_layout: Any) -> Optional[tuple]:
    if shard_spec is None:
        return None
    try:
        return _tuple_metadata(_read_metadata(shard_spec, "shard_grid"))
    except ValueError:
        resolver = getattr(shard_spec, "with_resolved_shard_grid", None)
        if not callable(resolver) or memory_layout is None:
            raise
        resolved = resolver(memory_layout)
        return _tuple_metadata(_read_metadata(resolved, "shard_grid"))


def _compiler_tensor_metadata(value: Any) -> dict[str, Any]:
    """Serialize every compiler-relevant property exposed by a sim tensor."""
    layout = _enum_name(_read_metadata(value, "layout"), "TILE")
    memory_config = _read_metadata(value, "memory_config")
    memory_space = _enum_name(_read_metadata(memory_config, "buffer_type"), "DRAM")
    raw_memory_layout = _read_metadata(memory_config, "memory_layout")
    memory_layout = _enum_name(raw_memory_layout, "INTERLEAVED")

    tile_shape = None
    tile_size_bytes = None
    if "TILE" in layout:
        tile = _read_metadata(value, "tile")
        if tile is None:
            tile = _read_metadata(value, "get_tile")
        tile_shape = _tuple_metadata(_read_metadata(tile, "tile_shape"))
        if callable(getattr(tile, "get_tile_size", None)):
            tile_size_bytes = int(tile.get_tile_size(value.dtype))

    shard_spec = _read_metadata(memory_config, "shard_spec")
    shard_shape = _tuple_metadata(
        _read_metadata(shard_spec, "shape", _read_metadata(shard_spec, "shard_shape"))
    )
    shard_grid = _shard_grid_metadata(shard_spec, raw_memory_layout)
    shard_orientation = None
    if shard_spec is not None:
        shard_orientation = _enum_name(
            _read_metadata(shard_spec, "orientation"), "UNKNOWN"
        )
    shard_core_ranges = _core_ranges_metadata(_read_metadata(shard_spec, "grid"))

    nd_shard_spec = _read_metadata(memory_config, "nd_shard_spec")
    nd_shard_shape = _tuple_metadata(_read_metadata(nd_shard_spec, "shard_shape"))
    nd_shard_grid = _tuple_metadata(_read_metadata(nd_shard_spec, "shard_grid"))
    nd_shard_distribution = None
    if nd_shard_spec is not None:
        nd_shard_distribution = _enum_name(
            _read_metadata(nd_shard_spec, "shard_distribution_strategy"),
            "UNKNOWN",
        )
    nd_shard_core_ranges = _core_ranges_metadata(_read_metadata(nd_shard_spec, "grid"))
    nd_shard_num_cores = _read_metadata(nd_shard_spec, "num_cores")

    mesh_info = _read_metadata(value, "mesh_shard_info")
    return {
        "padded_shape": _tuple_metadata(_read_metadata(value, "padded_shape")),
        "layout": layout,
        "memory_space": memory_space,
        "memory_layout": memory_layout,
        "tile_shape": tile_shape,
        "tile_size_bytes": tile_size_bytes,
        "shard_shape": shard_shape,
        "shard_grid": shard_grid,
        "shard_orientation": shard_orientation,
        "shard_core_ranges": shard_core_ranges,
        "nd_shard_shape": nd_shard_shape,
        "nd_shard_grid": nd_shard_grid,
        "nd_shard_distribution": nd_shard_distribution,
        "nd_shard_core_ranges": nd_shard_core_ranges,
        "nd_shard_num_cores": nd_shard_num_cores,
        "mesh_shape": _tuple_metadata(_read_metadata(mesh_info, "mesh_shape")),
        "mesh_dims": _tuple_metadata(_read_metadata(mesh_info, "dims")),
    }


def _to_compiler_value(value: Any, backend: _CompilerBackend, memo: dict[int, Any]):
    """Convert a direct simulator tensor while preserving argument aliases."""
    if not _is_sim_tensor(value):
        return value
    identity = id(value)
    if identity not in memo:
        memo[identity] = backend.tensor_spec_type(
            value.shape,
            value.dtype,
            **_compiler_tensor_metadata(value),
        )
    return memo[identity]


def _make_cell(value: Any):
    return (lambda: value).__closure__[0]


def _clone_for_compiler(
    function: Callable,
    backend: _CompilerBackend,
    simulator_ttl: Any,
    tensor_memo: dict[int, Any],
) -> Callable:
    """Clone a simulator operation with compiler TTL globals and tensor captures."""
    replacements = {id(simulator_ttl): backend.ttl_package}
    for name in dir(simulator_ttl):
        if not hasattr(backend.ttl_package, name):
            continue
        simulator_value = getattr(simulator_ttl, name)
        replacements[id(simulator_value)] = getattr(backend.ttl_package, name)

    def convert_capture(value: Any):
        replacement = replacements.get(id(value))
        if replacement is not None:
            return replacement
        return _to_compiler_value(value, backend, tensor_memo)

    globals_copy = {
        name: convert_capture(value) for name, value in function.__globals__.items()
    }
    closure = function.__closure__
    if closure is not None:
        closure = tuple(
            _make_cell(convert_capture(cell.cell_contents)) for cell in closure
        )
    cloned = types.FunctionType(
        function.__code__,
        globals_copy,
        function.__name__,
        function.__defaults__,
        closure,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__annotations__ = dict(function.__annotations__)
    cloned.__dict__.update(function.__dict__)
    cloned.__module__ = function.__module__
    cloned.__qualname__ = function.__qualname__
    return cloned


def prepare_operation_validator(
    function: Callable,
    *,
    grid,
    fp32_dest_acc_en: Optional[bool],
    dst_full_sync_en: Optional[bool],
    math_fidelity: Optional[str],
    simulator_ttl: Any = None,
) -> Optional[Callable]:
    """Build the optional compiler-side validator for a simulator operation."""
    config = get_context().config
    mode = config.compiler_validation_mode
    target_arch = config.compiler_validation_target
    if mode == "off":
        return None
    load = _load_backend()
    if load.backend is None:
        _report_unavailable(mode, load)
        return None
    backend = load.backend

    if simulator_ttl is None:
        simulator_ttl = sys.modules.get("ttl")
    captured_tensor_memo: dict[int, Any] = {}
    compiler_function = _clone_for_compiler(
        function, backend, simulator_ttl, captured_tensor_memo
    )
    with _compiler_import_scope(backend.ttl_package):
        validator = backend.build_validator(
            compiler_function,
            grid=grid,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            math_fidelity=math_fidelity,
            target_arch=target_arch,
        )

    def _validate(*args: Any, **kwargs: Any) -> None:
        # Seed each call with converted captures so a simulator tensor used as
        # both a capture and an argument remains one compiler tensor value.
        memo = dict(captured_tensor_memo)
        compiler_args = tuple(
            _to_compiler_value(value, backend, memo) for value in args
        )
        compiler_kwargs = {
            name: _to_compiler_value(value, backend, memo)
            for name, value in kwargs.items()
        }
        with _compiler_import_scope(backend.ttl_package):
            validator(*compiler_args, **compiler_kwargs)

    return _validate


__all__ = [
    "CompilerValidationUnavailable",
    "configure",
    "prepare_operation_validator",
]
