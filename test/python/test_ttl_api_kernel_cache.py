# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for @ttl.operation cache and program-hash behavior."""

import itertools

import ttl.ttl_api as ttl_api


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
