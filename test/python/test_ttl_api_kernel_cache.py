# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for @ttl.operation cache and program-hash behavior."""

import itertools
from types import SimpleNamespace

import ttl.ttl_api as ttl_api


def test_data_movement_configs_are_opt_in_and_preserve_default_roles(monkeypatch):
    class ReaderConfigDescriptor:
        pass

    class WriterConfigDescriptor:
        pass

    class DataMovementConfigDescriptor:
        def __init__(self, processor, noc, noc_mode):
            self.processor = processor
            self.noc = noc
            self.noc_mode = noc_mode

    riscv_0 = object()
    riscv_1 = object()
    noc_0 = object()
    noc_1 = object()
    dynamic_noc = object()
    fake_ttnn = SimpleNamespace(
        ReaderConfigDescriptor=ReaderConfigDescriptor,
        WriterConfigDescriptor=WriterConfigDescriptor,
        DataMovementConfigDescriptor=DataMovementConfigDescriptor,
        DataMovementProcessor=SimpleNamespace(RISCV_0=riscv_0, RISCV_1=riscv_1),
        NOC=SimpleNamespace(RISCV_0_default=noc_0, RISCV_1_default=noc_1),
        NOC_MODE=SimpleNamespace(DM_DYNAMIC_NOC=dynamic_noc),
    )
    monkeypatch.setattr(ttl_api, "ttnn", fake_ttnn)

    assert isinstance(
        ttl_api._make_data_movement_config(0, dynamic_noc=False),
        ReaderConfigDescriptor,
    )
    assert isinstance(
        ttl_api._make_data_movement_config(1, dynamic_noc=False),
        WriterConfigDescriptor,
    )

    ncrisc = ttl_api._make_data_movement_config(0, dynamic_noc=True)
    assert (ncrisc.processor, ncrisc.noc, ncrisc.noc_mode) == (
        riscv_1,
        noc_0,
        dynamic_noc,
    )
    brisc = ttl_api._make_data_movement_config(1, dynamic_noc=True)
    assert (brisc.processor, brisc.noc, brisc.noc_mode) == (
        riscv_0,
        noc_1,
        dynamic_noc,
    )


class _FakeMemoryConfig:
    def __init__(self, memory_space):
        self.buffer_type = memory_space


class _FakeTensor:
    def __init__(
        self,
        shape=(32, 32),
        dtype="ttnn.bfloat16",
        memory_space="L1",
        layout="TILE",
    ):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self._memory_config = _FakeMemoryConfig(memory_space)

    def memory_config(self):
        return self._memory_config

    def device(self):
        return None


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

    first_result = copy_kernel(_FakeTensor(), _FakeTensor())
    second_result = copy_kernel(_FakeTensor(), _FakeTensor())

    assert len(compile_calls) == 1
    assert first_result == second_result == compile_calls[0]["program_hash"]
    assert len(compile_calls[0]["compiled_kernel"].runtime_args) == 2


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

    program_hashes = {call["program_hash"] for call in compile_calls}
    assert len(compile_calls) == 5
    assert len(program_hashes) == 5


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


def test_operation_cache_separates_dynamic_noc_option(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)

    @ttl_api.operation(grid=(1, 1))
    def copy_kernel(input_tensor, output_tensor):
        pass

    default_result = copy_kernel(_FakeTensor(), _FakeTensor())
    dynamic_result = copy_kernel(
        _FakeTensor(), _FakeTensor(), options="--ttl-dynamic-noc"
    )
    repeated_dynamic_result = copy_kernel(
        _FakeTensor(), _FakeTensor(), options="--ttl-dynamic-noc"
    )

    assert len(compile_calls) == 2
    assert default_result != dynamic_result
    assert dynamic_result == repeated_dynamic_result
    assert compile_calls[0]["compile_options"]["compiler_options"].dynamic_noc is False
    assert compile_calls[1]["compile_options"]["compiler_options"].dynamic_noc is True


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


def test_caller_owned_factory_cache_reuses_separate_wrappers(monkeypatch):
    compile_calls = _install_recording_compile(monkeypatch)
    factory_cache = {}

    def make_scaled_kernel(scale):
        @ttl_api.operation(grid=(1, 1), factory_cache=factory_cache, factory_cache_key=("scaled", scale))
        def scaled_kernel(input_tensor, output_tensor):
            return scale

        return scaled_kernel

    first = make_scaled_kernel(2)
    repeated = make_scaled_kernel(2)
    different = make_scaled_kernel(3)

    first_result = first(_FakeTensor(), _FakeTensor())
    repeated_result = repeated(_FakeTensor(), _FakeTensor())
    different_result = different(_FakeTensor(), _FakeTensor())

    assert first is not repeated
    assert len(compile_calls) == 2
    assert first_result == repeated_result
    assert first_result != different_result


def test_factory_cache_requires_key_and_mapping_together(monkeypatch):
    _install_recording_compile(monkeypatch)

    try:
        ttl_api.operation(grid=(1, 1), factory_cache={})(lambda a, b: None)
    except ValueError as error:
        assert "must be supplied together" in str(error)
    else:
        raise AssertionError("factory cache without a key was accepted")
