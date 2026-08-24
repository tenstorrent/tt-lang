# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for @ttl.operation cache and program-hash behavior."""

import gc
import itertools
import threading
import weakref

import pytest

import ttl.kernel_runner as kernel_runner
import ttl.ttl_api as ttl_api


class _FakeMemoryConfig:
    def __init__(self, memory_space, memory_layout):
        self.buffer_type = memory_space
        self.memory_layout = memory_layout


class _FakeTile:
    def __init__(self, tile_shape):
        self.tile_shape = tile_shape


class _FakeDevice:
    arch = "blackhole"


class _FakeTensor:
    def __init__(
        self,
        shape=(32, 32),
        padded_shape=None,
        dtype="ttnn.bfloat16",
        memory_space="L1",
        memory_layout="HEIGHT_SHARDED",
        layout="TILE",
        tile=(32, 32),
        allocation_capacity=1 << 20,
        device=None,
    ):
        self.shape = shape
        self.padded_shape = padded_shape or shape
        self.dtype = dtype
        self.layout = layout
        self._memory_config = _FakeMemoryConfig(memory_space, memory_layout)
        self._tile = _FakeTile(tile)
        self.allocation_capacity = allocation_capacity
        self._device = device

    def memory_config(self):
        return self._memory_config

    def get_tile(self):
        return self._tile

    def device(self):
        return self._device


class _RecordingCompiledKernel:
    def __init__(self, program_hash):
        self.program_hash = program_hash
        self.runtime_args = []

    def __call__(self, *runtime_args):
        self.runtime_args.append(runtime_args)
        return self.program_hash


def _install_recording_compile(monkeypatch):
    compile_calls = []
    kernel_id_counter = itertools.count(1)
    hash_values = {}

    def deterministic_hash(value):
        if value not in hash_values:
            hash_values[value] = len(hash_values) + 1
        return hash_values[value]

    monkeypatch.setattr(
        ttl_api.random, "getrandbits", lambda bit_count: next(kernel_id_counter)
    )
    monkeypatch.setattr("builtins.hash", deterministic_hash)

    def fake_compile(
        kernel_function,
        runtime_args,
        runtime_kwargs,
        grid,
        indexing_maps,
        iterator_types,
        num_outs,
        memory_space,
        tiled,
        program_hash,
        **compile_options,
    ):
        compiled_kernel = _RecordingCompiledKernel(program_hash)
        compile_calls.append(
            {
                "kernel_function": kernel_function,
                "runtime_args": runtime_args,
                "runtime_kwargs": runtime_kwargs,
                "grid": grid,
                "program_hash": program_hash,
                "compiled_kernel": compiled_kernel,
                "compile_options": compile_options,
            }
        )
        return compiled_kernel

    monkeypatch.setattr(
        ttl_api, "is_ttnn_tensor", lambda arg: isinstance(arg, _FakeTensor)
    )
    monkeypatch.setattr(ttl_api, "_compile_kernel", fake_compile)
    return compile_calls


def test_operation_cache_reuses_compiled_kernel_for_same_tensor_config(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    first_input = _FakeTensor()
    first_output = _FakeTensor()
    second_input = _FakeTensor()
    second_output = _FakeTensor()
    first_result = copy_kernel(first_input, first_output)
    second_result = copy_kernel(second_input, second_output)

    assert len(compile_calls) == 1
    assert first_result == second_result == compile_calls[0]["program_hash"]
    assert len(compile_calls[0]["compiled_kernel"].runtime_args) == 2
    assert compile_calls[0]["compiled_kernel"].runtime_args == [
        (first_input, first_output),
        (second_input, second_output),
    ]


def test_operation_propagates_math_fidelity(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)

    @ttl_api.operation(grid=(1, 1), math_fidelity="HiFi3")
    def copy_kernel(input_tensor, output_tensor):
        pass

    copy_kernel(_FakeTensor(), _FakeTensor())

    assert compile_calls[0]["compile_options"]["math_fidelity"] == "HiFi3"


def test_explicit_operation_propagates_runtime_resource_factory(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)

    def make_resources(**_kwargs):
        return None

    @ttl_api.operation(grid=(1, 1), runtime_resource_factory=make_resources)
    def copy_kernel(input_tensor, output_tensor):
        pass

    copy_kernel(_FakeTensor(), _FakeTensor())

    assert (
        compile_calls[0]["compile_options"]["runtime_resource_factory"]
        is make_resources
    )
    assert isinstance(
        compile_calls[0]["compile_options"]["runtime_resource_cache"],
        kernel_runner.KernelRuntimeResourceCache,
    )


def test_cache_key_separates_math_fidelity(monkeypatch):
    monkeypatch.setattr(
        ttl_api, "is_ttnn_tensor", lambda arg: isinstance(arg, _FakeTensor)
    )
    tensors = (_FakeTensor(), _FakeTensor())
    common_options = {
        "args": tensors,
        "resolved_grid": (1, 1),
        "fp32_dest_acc_en": False,
        "dst_full_sync_en": False,
        "target_arch": "blackhole",
    }

    hifi2_key = ttl_api._make_cache_key(math_fidelity="HiFi2", **common_options)
    hifi4_key = ttl_api._make_cache_key(math_fidelity="HiFi4", **common_options)

    assert hifi2_key != hifi4_key


def test_operation_rejects_invalid_math_fidelity():
    with pytest.raises(ValueError, match="math_fidelity must be one of"):
        ttl_api.operation(grid=(1, 1), math_fidelity="HiFi5")


def test_operation_cache_reuses_kernel_across_allocation_capacities(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    copy_kernel(
        _FakeTensor(allocation_capacity=2048),
        _FakeTensor(allocation_capacity=2048),
    )
    copy_kernel(
        _FakeTensor(allocation_capacity=4096),
        _FakeTensor(allocation_capacity=4096),
    )

    assert len(compile_calls) == 1


def test_operation_cache_separates_tensor_alias_partitions(monkeypatch):
    """Backing argument identity affects captured tensor-storage indices."""
    compile_calls = _install_recording_compile(monkeypatch)

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    shared_tensor = _FakeTensor()
    copy_kernel(shared_tensor, shared_tensor)
    copy_kernel(_FakeTensor(), _FakeTensor())
    copy_kernel(shared_tensor, shared_tensor)

    assert len(compile_calls) == 2


def test_operation_cache_separates_tensor_config_changes(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    copy_kernel(_FakeTensor(), _FakeTensor())
    copy_kernel(_FakeTensor(shape=(64, 32)), _FakeTensor(shape=(64, 32)))
    copy_kernel(_FakeTensor(memory_space="DRAM"), _FakeTensor(memory_space="DRAM"))
    copy_kernel(_FakeTensor(layout="ROW_MAJOR"), _FakeTensor(layout="ROW_MAJOR"))
    copy_kernel(_FakeTensor(dtype="ttnn.float32"), _FakeTensor(dtype="ttnn.float32"))
    copy_kernel(
        _FakeTensor(padded_shape=(64, 32)),
        _FakeTensor(padded_shape=(64, 32)),
    )
    copy_kernel(
        _FakeTensor(memory_layout="WIDTH_SHARDED"),
        _FakeTensor(memory_layout="WIDTH_SHARDED"),
    )
    copy_kernel(_FakeTensor(tile=(16, 32)), _FakeTensor(tile=(16, 32)))

    program_hashes = {call["program_hash"] for call in compile_calls}
    assert len(compile_calls) == 8
    assert len(program_hashes) == 8

    runtime_caches = {
        id(call["compile_options"]["runtime_resource_cache"]) for call in compile_calls
    }
    assert len(runtime_caches) == 1


def test_operation_cache_compilation_is_single_flight(monkeypatch):
    compile_started = threading.Event()
    release_compile = threading.Event()
    first_dispatch_started = threading.Event()
    release_first_dispatch = threading.Event()
    compilation_count = 0
    dispatches = []

    class SerializedCompiledKernel:
        all_source_lines = {}

        def __init__(self, runtime_resource_cache):
            self.runtime_resource_cache = runtime_resource_cache

        def __call__(self, *runtime_args):
            with self.runtime_resource_cache.lock:
                dispatches.append(runtime_args)
                if len(dispatches) == 1:
                    first_dispatch_started.set()
                    assert release_first_dispatch.wait(timeout=5)
            return len(dispatches)

    def compile_kernel(*_args, runtime_resource_cache, **_kwargs):
        nonlocal compilation_count
        compilation_count += 1
        compile_started.set()
        assert release_compile.wait(timeout=5)
        return SerializedCompiledKernel(runtime_resource_cache)

    monkeypatch.setattr(
        ttl_api, "is_ttnn_tensor", lambda arg: isinstance(arg, _FakeTensor)
    )
    monkeypatch.setattr(ttl_api, "_compile_kernel", compile_kernel)

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    def invoke():
        copy_kernel(_FakeTensor(), _FakeTensor())

    first_thread = threading.Thread(target=invoke)
    second_thread = threading.Thread(target=invoke)
    first_thread.start()
    assert compile_started.wait(timeout=5)
    second_thread.start()
    release_compile.set()
    assert first_dispatch_started.wait(timeout=5)
    assert compilation_count == 1
    assert len(dispatches) == 1
    release_first_dispatch.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert compilation_count == 1
    assert len(dispatches) == 2


def test_operation_cache_synchronizes_before_owner_destruction(monkeypatch):
    events = []

    class LifetimeOwner:
        def __del__(self):
            events.append("release")

    class ResourceCompiledKernel:
        all_source_lines = {}

        def __init__(self, runtime_resource_cache):
            self.runtime_resource_cache = runtime_resource_cache

        def __call__(self, *_runtime_args):
            self.runtime_resource_cache.compatibility_key = ("resources",)
            self.runtime_resource_cache.device = "device"
            self.runtime_resource_cache.pipe_resources = LifetimeOwner()
            return None

    def compile_kernel(
        _runtime_args,
        _runtime_kwargs,
        _resolved_grid,
        _program_hash,
        _target_arch,
        _compiler_options,
        _l1_budget_override,
        runtime_resource_cache,
    ):
        return ResourceCompiledKernel(runtime_resource_cache)

    monkeypatch.setattr(
        ttl_api, "is_ttnn_tensor", lambda arg: isinstance(arg, _FakeTensor)
    )
    monkeypatch.setattr(ttl_api, "_resolve_l1_budget", lambda *_args: 98304)
    monkeypatch.setattr(
        kernel_runner,
        "ttnn",
        type(
            "FakeTTNN",
            (),
            {
                "synchronize_device": staticmethod(
                    lambda device: events.append(f"synchronize:{device}")
                )
            },
        )(),
    )

    def operation(input_tensor, output_tensor):
        pass

    wrapper = ttl_api._make_operation_wrapper(
        operation,
        compile_kernel,
        grid=(1, 1),
        fp32_dest_acc_en=None,
        dst_full_sync_en=None,
        math_fidelity=None,
        options=None,
    )
    wrapper(_FakeTensor(), _FakeTensor())
    wrapper_reference = weakref.ref(wrapper)

    del wrapper
    gc.collect()

    assert wrapper_reference() is None
    assert events == ["synchronize:device", "release"]


def test_private_compiled_kernel_synchronizes_before_owner_destruction(monkeypatch):
    events = []

    class LifetimeOwner:
        def __del__(self):
            events.append("release")

    monkeypatch.setattr(
        kernel_runner,
        "ttnn",
        type(
            "FakeTTNN",
            (),
            {
                "synchronize_device": staticmethod(
                    lambda device: events.append(f"synchronize:{device}")
                )
            },
        )(),
    )
    compiled_kernel = ttl_api.CompiledTTNNKernel(
        kernel_paths=[],
        kernel_configs=[],
        kernel_arg_specs=[],
        num_tensors=0,
        core_ranges=None,
        kernel_tensor_indices=[],
    )
    compiled_kernel._runtime_resource_cache.compatibility_key = ("resources",)
    compiled_kernel._runtime_resource_cache.device = "device"
    compiled_kernel._runtime_resource_cache.pipe_resources = LifetimeOwner()
    compiled_reference = weakref.ref(compiled_kernel)

    del compiled_kernel
    gc.collect()

    assert compiled_reference() is None
    assert events == ["synchronize:device", "release"]


@pytest.mark.parametrize(
    "synchronization_error",
    [RuntimeError("device synchronization failed"), KeyboardInterrupt()],
    ids=["runtime-error", "keyboard-interrupt"],
)
def test_runtime_resource_finalizer_retains_owners_when_sync_fails(
    monkeypatch, synchronization_error
):
    events = []

    class LifetimeOwner:
        def __del__(self):
            events.append("release")

    def fail_synchronization(_device):
        events.append("synchronize")
        raise synchronization_error

    fake_ttnn = type(
        "FakeTTNN", (), {"synchronize_device": staticmethod(fail_synchronization)}
    )()
    monkeypatch.setattr(kernel_runner, "ttnn", fake_ttnn)
    runtime_resource_cache = kernel_runner.KernelRuntimeResourceCache(
        compatibility_key=("resources",),
        device="device",
        pipe_resources=LifetimeOwner(),
    )

    with pytest.warns(RuntimeWarning, match="failed to synchronize"):
        kernel_runner.finalize_runtime_resource_cache(runtime_resource_cache)

    assert events == ["synchronize"]
    assert runtime_resource_cache in kernel_runner._RETAINED_RUNTIME_RESOURCE_CACHES
    assert runtime_resource_cache.pipe_resources is not None

    fake_ttnn.synchronize_device = lambda _device: events.append("cleanup-sync")
    kernel_runner._RETAINED_RUNTIME_RESOURCE_CACHES.remove(runtime_resource_cache)
    kernel_runner.release_cached_runtime_resources(runtime_resource_cache)
    assert events == ["synchronize", "cleanup-sync", "release"]


def test_operation_cache_separates_resolved_grid_changes(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)
    grid_widths = iter((1, 2, 1))

    @ttl_api.operation(grid=lambda input_tensor, output_tensor: (next(grid_widths), 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    first_result = copy_kernel(_FakeTensor(), _FakeTensor())
    second_result = copy_kernel(_FakeTensor(), _FakeTensor())
    repeated_first_result = copy_kernel(_FakeTensor(), _FakeTensor())

    assert len(compile_calls) == 2
    assert first_result != second_result
    assert first_result == repeated_first_result


def test_operation_cache_separates_effective_l1_budgets(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    input_tensor = _FakeTensor()
    output_tensor = _FakeTensor()
    first_result = copy_kernel(
        input_tensor, output_tensor, options="--ttl-l1-budget 98304"
    )
    second_result = copy_kernel(
        input_tensor, output_tensor, options="--ttl-l1-budget 73760"
    )
    repeated_first_result = copy_kernel(
        input_tensor, output_tensor, options="--ttl-l1-budget 98304"
    )

    assert len(compile_calls) == 2
    assert first_result != second_result
    assert first_result == repeated_first_result
    assert compile_calls[0]["compile_options"]["l1_budget_override"] == 98304
    assert compile_calls[1]["compile_options"]["l1_budget_override"] == 73760


def test_operation_cache_rechecks_device_derived_l1_budget(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)
    budget_queries = []

    def resolve_budget(runtime_args, compiler_options, _runtime_resource_cache):
        budget_queries.append((runtime_args, compiler_options))
        return 98304 if len(budget_queries) == 1 else 73760

    monkeypatch.setattr(
        ttl_api,
        "_resolve_l1_budget",
        resolve_budget,
    )

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    input_tensor = _FakeTensor()
    output_tensor = _FakeTensor()
    first_result = copy_kernel(input_tensor, output_tensor)
    second_result = copy_kernel(input_tensor, output_tensor)
    repeated_first_result = copy_kernel(input_tensor, output_tensor)

    assert len(compile_calls) == 2
    assert len(budget_queries) == 3
    assert first_result != second_result
    assert second_result == repeated_first_result
    assert compile_calls[0]["compile_options"]["l1_budget_override"] == 98304


def test_operation_cache_uses_l1_budget_without_owned_resources(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)
    device = _FakeDevice()
    monkeypatch.setattr(
        kernel_runner,
        "ttnn",
        type(
            "FakeTTNN",
            (),
            {"synchronize_device": staticmethod(lambda _device: None)},
        )(),
    )
    remaining_budgets = iter((98304, 98240, 98240))
    monkeypatch.setattr(
        ttl_api,
        "get_min_remaining_l1_excluding_cached_resources",
        lambda resource_cache, selected_device: next(remaining_budgets),
    )

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    copy_kernel(_FakeTensor(device=device), _FakeTensor(device=device))
    resource_cache = compile_calls[0]["compile_options"]["runtime_resource_cache"]
    resource_cache.compatibility_key = ("variant-a",)
    resource_cache.device = device
    resource_cache.pipe_resources = object()

    copy_kernel(_FakeTensor(device=device), _FakeTensor(device=device))
    copy_kernel(_FakeTensor(device=device), _FakeTensor(device=device))

    assert len(compile_calls) == 2
    assert [
        call["compile_options"]["l1_budget_override"] for call in compile_calls
    ] == [98304, 98240]


def test_operation_cache_separates_device_derived_budget_contracts(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)
    budgets = iter((98304, 73760))
    monkeypatch.setattr(
        ttl_api,
        "_resolve_l1_budget",
        lambda runtime_args, compiler_options, runtime_resource_cache: next(budgets),
    )

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    first_device = _FakeDevice()
    second_device = _FakeDevice()
    copy_kernel(_FakeTensor(device=first_device), _FakeTensor(device=first_device))
    copy_kernel(_FakeTensor(device=second_device), _FakeTensor(device=second_device))

    assert len(compile_calls) == 2
    assert compile_calls[0]["compile_options"]["l1_budget_override"] == 98304
    assert compile_calls[1]["compile_options"]["l1_budget_override"] == 73760


def _make_scaled_kernel(scale):
    @ttl_api.operation(grid=(1, 1))
    def scaled_kernel(input_tensor, output_tensor):
        return scale

    return scaled_kernel


def test_factory_kernels_with_captured_constants_get_separate_hashes(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)
    scale_by_two = _make_scaled_kernel(2)
    scale_by_three = _make_scaled_kernel(3)

    two_result = scale_by_two(_FakeTensor(), _FakeTensor())
    three_result = scale_by_three(_FakeTensor(), _FakeTensor())
    repeated_two_result = scale_by_two(_FakeTensor(), _FakeTensor())

    assert len(compile_calls) == 2
    assert two_result == repeated_two_result
    assert two_result != three_result


def test_factory_level_kernel_cache_reuses_matching_decorated_kernel(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)
    kernel_cache = {}

    def make_cached_scaled_kernel(scale):
        if scale not in kernel_cache:
            kernel_cache[scale] = _make_scaled_kernel(scale)
        return kernel_cache[scale]

    scale_by_two = make_cached_scaled_kernel(2)
    repeated_scale_by_two = make_cached_scaled_kernel(2)
    scale_by_three = make_cached_scaled_kernel(3)

    two_result = scale_by_two(_FakeTensor(), _FakeTensor())
    repeated_two_result = repeated_scale_by_two(_FakeTensor(), _FakeTensor())
    three_result = scale_by_three(_FakeTensor(), _FakeTensor())

    assert scale_by_two is repeated_scale_by_two
    assert len(compile_calls) == 2
    assert two_result == repeated_two_result
    assert two_result != three_result
