# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for external functions receiving typed tensor accessors."""

import importlib.util
import os
import shutil
from functools import lru_cache, partial

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttl.dtype_utils import tile_bytes_from_dtype
from ttl.layouts import get_supported_ttnn_dtype_layouts, get_tensor_configuration
from ttlang_test_utils import (
    is_hardware_available,
    to_dram,
    to_l1,
    to_l1_sharded,
)
from utils.correctness import assert_allclose

TENSOR_ACCESSOR_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "tensor_accessor_read.hpp"
)
LOCAL_TENSOR_ACCESSOR_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "local_tensor_accessor_copy.hpp"
)
MEMORY_CONFIGS = [
    pytest.param(to_dram, id="dram-interleaved"),
    pytest.param(to_l1, id="l1-interleaved"),
    pytest.param(partial(to_l1_sharded, layout="height"), id="l1-height-sharded"),
    pytest.param(partial(to_l1_sharded, layout="width"), id="l1-width-sharded"),
    pytest.param(partial(to_l1_sharded, layout="block"), id="l1-block-sharded"),
]

PUBLIC_DTYPE_LAYOUTS = [
    pytest.param(
        dtype,
        tensor_layout,
        id=(
            f"{str(dtype).rsplit('.', maxsplit=1)[-1].lower()}-"
            f"{'tile' if tensor_layout == ttnn.TILE_LAYOUT else 'row-major'}"
        ),
    )
    for dtype, tensor_layout in get_supported_ttnn_dtype_layouts(ttnn)
]
COMPUTE_MEMORY_CONFIGS = [
    pytest.param(storage, distribution, id=f"{storage}-{distribution}")
    for storage, distributions in (
        ("l1", ("height", "width", "block")),
        ("l1-small", ("height", "width", "block")),
    )
    for distribution in distributions
]
DATA_MOVEMENT_MEMORY_CONFIGS = [
    pytest.param(storage, distribution, id=f"{storage}-{distribution}")
    for storage, distributions in (
        ("dram", ("interleaved", "height", "width", "block", "nd")),
        ("l1", ("interleaved", "height", "width", "block", "nd")),
        ("l1-small", ("height", "width", "block", "nd")),
    )
    for distribution in distributions
]

_TORCH_DTYPE_BY_TTNN_DTYPE = {
    ttnn.DataType.FLOAT32: torch.float32,
    ttnn.DataType.BFLOAT16: torch.bfloat16,
    ttnn.DataType.BFLOAT8_B: torch.bfloat16,
    ttnn.DataType.BFLOAT4_B: torch.bfloat16,
    ttnn.DataType.INT32: torch.int32,
    ttnn.DataType.UINT32: torch.uint32,
    ttnn.DataType.UINT16: torch.uint16,
    ttnn.DataType.UINT8: torch.uint8,
}
_ROW_MAJOR_ELEMENT_BYTES = {
    ttnn.DataType.FLOAT32: 4,
    ttnn.DataType.BFLOAT16: 2,
    ttnn.DataType.INT32: 4,
    ttnn.DataType.UINT32: 4,
    ttnn.DataType.UINT16: 2,
    ttnn.DataType.UINT8: 1,
}


@pytest.fixture(scope="module")
def accessor_device():
    """Open one device with SRAM reserved for L1Small tensor coverage."""

    if not is_hardware_available():
        pytest.skip("No Tenstorrent device available")
    dispatch_core_config = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)
    device = ttnn.open_device(
        device_id=0,
        dispatch_core_config=dispatch_core_config,
        l1_small_size=32 * 1024,
    )
    yield device
    ttnn.close_device(device)


def _tensor_shape(tensor_layout):
    return (32, 32) if tensor_layout == ttnn.TILE_LAYOUT else (1, 32)


def _page_size_bytes(dtype, tensor_layout):
    if tensor_layout == ttnn.TILE_LAYOUT:
        return tile_bytes_from_dtype(dtype)
    return 32 * _ROW_MAJOR_ELEMENT_BYTES[dtype]


def _make_memory_config(storage, distribution, tensor_shape, *, core_x=0):
    buffer_type = {
        "dram": ttnn.BufferType.DRAM,
        "l1": ttnn.BufferType.L1,
        "l1-small": ttnn.BufferType.L1_SMALL,
    }[storage]
    memory_layout = {
        "interleaved": ttnn.TensorMemoryLayout.INTERLEAVED,
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "nd": ttnn.TensorMemoryLayout.ND_SHARDED,
    }[distribution]
    if distribution == "interleaved":
        return ttnn.MemoryConfig(memory_layout, buffer_type)
    worker_core = ttnn.CoreCoord(core_x, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(worker_core, worker_core)})
    if distribution == "nd":
        return ttnn.MemoryConfig(
            buffer_type,
            ttnn.NdShardSpec(
                tensor_shape,
                core_grid,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    shard_spec = ttnn.ShardSpec(
        core_grid,
        tensor_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(memory_layout, buffer_type, shard_spec)


def _make_host_tensor(dtype, tensor_layout, *, zeros=False):
    tensor_shape = _tensor_shape(tensor_layout)
    torch_dtype = _TORCH_DTYPE_BY_TTNN_DTYPE[dtype]
    if zeros:
        return torch.zeros(tensor_shape, dtype=torch_dtype)
    element_count = tensor_shape[0] * tensor_shape[1]
    values = torch.arange(element_count, dtype=torch.int64).reshape(tensor_shape)
    return (values % 23).to(torch_dtype)


def _to_configured_tensor(
    host_tensor, device, dtype, tensor_layout, storage, distribution, *, core_x=0
):
    return ttnn.from_torch(
        host_tensor,
        dtype=dtype,
        layout=tensor_layout,
        device=device,
        memory_config=_make_memory_config(
            storage, distribution, tuple(host_tensor.shape), core_x=core_x
        ),
    )


def _make_tensor_accessor_copy(data_format):
    """Compile each page byte size into its DFB descriptor template argument."""

    @ttl.operation(grid=(1, 1))
    def tensor_accessor_copy(inp, out):
        transfer_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm_read():
            ttl.call_extern_func(
                TENSOR_ACCESSOR_HEADER,
                "tensor_accessor_read",
                template_args=[ttl.dfb_descriptor(transfer_dfb)],
                func_args=[inp],
            )

        @ttl.datamovement()
        def dm_write():
            source = transfer_dfb.wait()
            ttl.copy(source, out[0, 0]).wait()
            source.pop()

    return tensor_accessor_copy


BF16_TENSOR_ACCESSOR_COPY = _make_tensor_accessor_copy("bf16")
F32_TENSOR_ACCESSOR_COPY = _make_tensor_accessor_copy("float32")


def _make_multitile_tensor_accessor_copy(data_format):
    @ttl.operation(grid=(1, 1))
    def multitile_tensor_accessor_copy(inp, out):
        transfer_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm_read():
            ttl.call_extern_func(
                TENSOR_ACCESSOR_HEADER,
                "tensor_accessor_read_page",
                template_args=[ttl.dfb_descriptor(transfer_dfb), 3],
                func_args=[inp],
            )

        @ttl.datamovement()
        def dm_write():
            source = transfer_dfb.wait()
            ttl.copy(source, out[0, 0]).wait()
            source.pop()

    return multitile_tensor_accessor_copy


BF16_MULTITILE_TENSOR_ACCESSOR_COPY = _make_multitile_tensor_accessor_copy("bf16")
F32_MULTITILE_TENSOR_ACCESSOR_COPY = _make_multitile_tensor_accessor_copy("float32")


def _make_tensor_accessor_pair_copy(data_format):
    """Preserve argument order for two independent TensorAccessor values."""

    @ttl.operation(grid=(1, 1))
    def tensor_accessor_pair_copy(first_inp, second_inp, first_out, second_out):
        first_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm_read():
            ttl.call_extern_func(
                TENSOR_ACCESSOR_HEADER,
                "tensor_accessor_pair_read",
                template_args=[
                    ttl.dfb_descriptor(first_dfb),
                    ttl.dfb_descriptor(second_dfb),
                ],
                func_args=[first_inp, second_inp],
            )

        @ttl.datamovement()
        def dm_write():
            first_source = first_dfb.wait()
            ttl.copy(first_source, first_out[0, 0]).wait()
            first_source.pop()
            second_source = second_dfb.wait()
            ttl.copy(second_source, second_out[0, 0]).wait()
            second_source.pop()

    return tensor_accessor_pair_copy


BF16_TENSOR_ACCESSOR_PAIR_COPY = _make_tensor_accessor_pair_copy("bf16")
F32_TENSOR_ACCESSOR_PAIR_COPY = _make_tensor_accessor_pair_copy("float32")


def _make_local_tensor_accessor_copy(byte_count):
    @ttl.operation(grid=(1, 1))
    def local_tensor_accessor_copy(inp, out):
        @ttl.compute()
        def compute():
            ttl.call_extern_func(
                LOCAL_TENSOR_ACCESSOR_HEADER,
                "local_tensor_accessor_copy",
                template_args=[byte_count],
                func_args=[inp, out],
            )

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            pass

    return local_tensor_accessor_copy


@lru_cache(maxsize=None)
def _get_local_tensor_accessor_copy(byte_count):
    return _make_local_tensor_accessor_copy(byte_count)


@ttl.operation(grid=(1, 1))
def tensor_accessor_page_copy(inp, out):
    transfer_dfb = ttl.make_dfb("float32", shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        ttl.call_extern_func(
            TENSOR_ACCESSOR_HEADER,
            "tensor_accessor_read_page",
            template_args=[ttl.dfb_descriptor(transfer_dfb), 0],
            func_args=[inp],
        )

    @ttl.datamovement()
    def dm_write():
        ttl.call_extern_func(
            TENSOR_ACCESSOR_HEADER,
            "tensor_accessor_write_page",
            template_args=[ttl.dfb_descriptor(transfer_dfb)],
            func_args=[out],
        )


@ttl.operation(grid=(1, 1))
def repeated_compute_tensor_accessor(tensor, device_anchor):
    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            LOCAL_TENSOR_ACCESSOR_HEADER,
            "local_tensor_accessor_copy",
            template_args=[1],
            func_args=[tensor, tensor],
        )

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(1, 1))
def repeated_compute_tensor_accessor_calls(inp, intermediate, out):
    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            LOCAL_TENSOR_ACCESSOR_HEADER,
            "local_tensor_accessor_copy",
            template_args=[2048],
            func_args=[inp, intermediate],
        )
        ttl.call_extern_func(
            LOCAL_TENSOR_ACCESSOR_HEADER,
            "local_tensor_accessor_copy",
            template_args=[2048],
            func_args=[intermediate, out],
        )

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(1, 1))
def mixed_distributed_and_local_tensor_accessors(
    distributed_in, distributed_out, local_in, local_out
):
    transfer_dfb = ttl.make_dfb("bf16", shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            LOCAL_TENSOR_ACCESSOR_HEADER,
            "local_tensor_accessor_copy",
            template_args=[2048],
            func_args=[local_in, local_out],
        )

    @ttl.datamovement()
    def dm_read():
        ttl.call_extern_func(
            TENSOR_ACCESSOR_HEADER,
            "tensor_accessor_read",
            template_args=[ttl.dfb_descriptor(transfer_dfb)],
            func_args=[distributed_in],
        )

    @ttl.datamovement()
    def dm_write():
        source = transfer_dfb.wait()
        ttl.copy(source, distributed_out[0, 0]).wait()
        source.pop()


@ttl.operation(grid=(1, 1))
def invalid_data_movement_tensor_accessor(tensor, device_anchor):
    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        ttl.call_extern_func(
            TENSOR_ACCESSOR_HEADER,
            "tensor_accessor_copy_page",
            func_args=[tensor, device_anchor],
        )

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(2, 1))
def invalid_local_tensor_shard_grid(inp, out):
    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            LOCAL_TENSOR_ACCESSOR_HEADER,
            "local_tensor_accessor_copy",
            template_args=[1],
            func_args=[inp, out],
        )

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(2, 1))
def specialized_local_tensor_subsets(first_inp, first_out, second_inp, second_out):
    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        if core_x == 0:
            ttl.call_extern_func(
                LOCAL_TENSOR_ACCESSOR_HEADER,
                "local_tensor_accessor_copy",
                template_args=[2048],
                func_args=[first_inp, first_out],
            )
        else:
            ttl.call_extern_func(
                LOCAL_TENSOR_ACCESSOR_HEADER,
                "local_tensor_accessor_copy",
                template_args=[2048],
                func_args=[second_inp, second_out],
            )

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (BF16_TENSOR_ACCESSOR_COPY, torch.bfloat16),
        (F32_TENSOR_ACCESSOR_COPY, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("to_device", MEMORY_CONFIGS)
def test_external_tensor_accessor(device, operation, dtype, to_device):
    """TensorAccessor preserves one tiled page across dtype and memory types."""

    # This legacy DFB-backed regression intentionally exercises tiled tensors.
    host = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32).to(dtype)
    inp = to_device(host, device)
    out = to_device(torch.zeros_like(host), device)

    operation(inp, out)

    actual = ttnn.to_torch(out).float()
    expected = host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (BF16_MULTITILE_TENSOR_ACCESSOR_COPY, torch.bfloat16),
        (F32_MULTITILE_TENSOR_ACCESSOR_COPY, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("to_device", MEMORY_CONFIGS)
def test_external_tensor_accessor_multitile(device, operation, dtype, to_device):
    """TensorAccessor page IDs address nonzero tiles in a larger tensor."""

    host = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64).to(dtype)
    inp = to_device(host, device)
    out = to_device(torch.zeros((32, 32), dtype=dtype), device)

    operation(inp, out)

    actual = ttnn.to_torch(out).float()
    expected = host[32:64, 32:64].float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (BF16_TENSOR_ACCESSOR_PAIR_COPY, torch.bfloat16),
        (F32_TENSOR_ACCESSOR_PAIR_COPY, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("to_device", MEMORY_CONFIGS)
def test_external_tensor_accessor_operand_order(device, operation, dtype, to_device):
    """Two TensorAccessor operands retain source order and runtime addresses."""

    element_indices = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    first_host = element_indices.to(dtype)
    second_host = (1000 - element_indices).to(dtype)
    first_inp = to_device(first_host, device)
    second_inp = to_device(second_host, device)
    first_out = to_device(torch.zeros_like(first_host), device)
    second_out = to_device(torch.zeros_like(second_host), device)
    swapped_first_out = to_device(torch.zeros_like(second_host), device)
    swapped_second_out = to_device(torch.zeros_like(first_host), device)

    operation(first_inp, second_inp, first_out, second_out)
    operation(second_inp, first_inp, swapped_first_out, swapped_second_out)

    first_actual = ttnn.to_torch(first_out).float()
    second_actual = ttnn.to_torch(second_out).float()
    swapped_first_actual = ttnn.to_torch(swapped_first_out).float()
    swapped_second_actual = ttnn.to_torch(swapped_second_out).float()
    first_expected = first_host.float()
    second_expected = second_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(first_actual, first_expected, rtol=0.05, atol=1.0)
        assert_allclose(second_actual, second_expected, rtol=0.05, atol=1.0)
        assert_allclose(swapped_first_actual, second_expected, rtol=0.05, atol=1.0)
        assert_allclose(swapped_second_actual, first_expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(first_actual, first_expected, rtol=1e-5, atol=1e-6)
        assert_allclose(second_actual, second_expected, rtol=1e-5, atol=1e-6)
        assert_allclose(swapped_first_actual, second_expected, rtol=1e-5, atol=1e-6)
        assert_allclose(swapped_second_actual, first_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("dtype", "tensor_layout"), PUBLIC_DTYPE_LAYOUTS)
@pytest.mark.parametrize(("storage", "distribution"), COMPUTE_MEMORY_CONFIGS)
def test_external_compute_local_tensor_accessor(
    accessor_device, dtype, tensor_layout, storage, distribution
):
    """LocalTensorAccessor copies one complete local tensor page."""

    host = _make_host_tensor(dtype, tensor_layout)
    device = accessor_device
    inp = _to_configured_tensor(
        host, device, dtype, tensor_layout, storage, distribution
    )
    out = _to_configured_tensor(
        _make_host_tensor(dtype, tensor_layout, zeros=True),
        device,
        dtype,
        tensor_layout,
        storage,
        distribution,
    )

    operation = _get_local_tensor_accessor_copy(_page_size_bytes(dtype, tensor_layout))
    operation(inp, out)

    actual = ttnn.to_torch(out).float()
    expected = ttnn.to_torch(inp).float()
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(("dtype", "tensor_layout"), PUBLIC_DTYPE_LAYOUTS)
@pytest.mark.parametrize(("storage", "distribution"), DATA_MOVEMENT_MEMORY_CONFIGS)
def test_external_data_movement_tensor_accessor_matrix(
    accessor_device, dtype, tensor_layout, storage, distribution
):
    """TensorAccessor reads one page from every accepted tensor configuration."""

    host = _make_host_tensor(dtype, tensor_layout)
    device = accessor_device
    inp = _to_configured_tensor(
        host, device, dtype, tensor_layout, storage, distribution
    )
    out = _to_configured_tensor(
        _make_host_tensor(dtype, tensor_layout, zeros=True),
        device,
        dtype,
        tensor_layout,
        "l1",
        "interleaved",
    )

    tensor_accessor_page_copy(inp, out)

    actual = ttnn.to_torch(out).float()
    expected = ttnn.to_torch(inp).float()
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_external_compute_tensor_accessor_rejects_dram(accessor_device):
    """Compute tensor access rejects storage without a node-local SRAM region."""

    host = torch.zeros((32, 32), dtype=torch.bfloat16)
    tensor = _to_configured_tensor(
        host,
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "dram",
        "interleaved",
    )
    device_anchor = _to_configured_tensor(
        host,
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "interleaved",
    )
    with pytest.raises(
        RuntimeError,
        match="compute kernel uses DRAM storage.*require sharded SRAM",
    ):
        repeated_compute_tensor_accessor(tensor, device_anchor)


def test_external_compute_tensor_accessor_reuses_repeated_operand(accessor_device):
    """One tensor passed twice in one call shares one local accessor."""

    host = _make_host_tensor(ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT)
    tensor = _to_configured_tensor(
        host,
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "height",
    )

    repeated_compute_tensor_accessor(tensor, tensor)
    assert_allclose(ttnn.to_torch(tensor).float(), host.float(), rtol=0, atol=0)


def test_external_compute_tensor_accessor_supports_repeated_calls(accessor_device):
    """Multiple external calls preserve their positional local tensor arguments."""

    host = _make_host_tensor(ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT)

    def make_tensor(host_tensor):
        return _to_configured_tensor(
            host_tensor,
            accessor_device,
            ttnn.DataType.BFLOAT16,
            ttnn.TILE_LAYOUT,
            "l1",
            "height",
        )

    inp = make_tensor(host)
    intermediate = make_tensor(torch.zeros_like(host))
    out = make_tensor(torch.zeros_like(host))

    repeated_compute_tensor_accessor_calls(inp, intermediate, out)

    assert_allclose(ttnn.to_torch(intermediate).float(), host.float(), rtol=0, atol=0)
    assert_allclose(ttnn.to_torch(out).float(), host.float(), rtol=0, atol=0)


def test_external_compute_tensor_accessor_updates_cached_addresses(accessor_device):
    """A cached compute kernel receives each invocation's tensor addresses."""

    def make_tensor(host_tensor):
        return _to_configured_tensor(
            host_tensor,
            accessor_device,
            ttnn.DataType.BFLOAT16,
            ttnn.TILE_LAYOUT,
            "l1",
            "height",
        )

    operation = _get_local_tensor_accessor_copy(
        _page_size_bytes(ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT)
    )
    first_host = _make_host_tensor(ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT)
    second_host = 1000 - first_host
    first_input = make_tensor(first_host)
    first_output = make_tensor(torch.zeros_like(first_host))
    second_input = make_tensor(second_host)
    second_output = make_tensor(torch.zeros_like(second_host))

    operation(first_input, first_output)
    operation(second_input, second_output)

    assert_allclose(
        ttnn.to_torch(first_output).float(), first_host.float(), rtol=0, atol=0
    )
    assert_allclose(
        ttnn.to_torch(second_output).float(), second_host.float(), rtol=0, atol=0
    )


def test_external_tensor_accessors_support_mixed_storage(accessor_device):
    """One operation selects distributed and local accessors per kernel."""

    distributed_host = _make_host_tensor(ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT)
    local_host = 1000 - distributed_host
    distributed_in = _to_configured_tensor(
        distributed_host,
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "dram",
        "interleaved",
    )
    distributed_out = _to_configured_tensor(
        torch.zeros_like(distributed_host),
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "dram",
        "interleaved",
    )
    local_in = _to_configured_tensor(
        local_host,
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "height",
    )
    local_out = _to_configured_tensor(
        torch.zeros_like(local_host),
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "height",
    )

    mixed_distributed_and_local_tensor_accessors(
        distributed_in, distributed_out, local_in, local_out
    )

    assert_allclose(
        ttnn.to_torch(distributed_out).float(), distributed_host.float(), rtol=0, atol=0
    )
    assert_allclose(
        ttnn.to_torch(local_out).float(), local_host.float(), rtol=0, atol=0
    )


def test_external_compute_tensor_accessor_rejects_system_memory(accessor_device):
    """Compute tensor access rejects a host TTNN tensor as SystemMemory."""

    host_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
    )
    device_anchor = _to_configured_tensor(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "interleaved",
    )
    with pytest.raises(
        RuntimeError,
        match="compute kernel uses SystemMemory storage.*require sharded SRAM",
    ):
        repeated_compute_tensor_accessor(host_tensor, device_anchor)


def test_external_compute_tensor_accessor_rejects_interleaved_l1(
    accessor_device,
):
    """Compute tensor access requires one local shard on each executing core."""

    host = torch.zeros((32, 32), dtype=torch.bfloat16)
    tensor = _to_configured_tensor(
        host,
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "interleaved",
    )
    with pytest.raises(
        RuntimeError,
        match="compute kernel uses an unsupported memory layout.*require height-, width-, or block-sharded SRAM",
    ):
        repeated_compute_tensor_accessor(tensor, tensor)


def test_external_compute_tensor_accessor_rejects_missing_local_shard(
    accessor_device,
):
    """Every executing compute core must have storage for a local accessor."""

    host = torch.zeros((32, 32), dtype=torch.bfloat16)
    inp = _to_configured_tensor(
        host,
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "height",
    )
    out = _to_configured_tensor(
        host,
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "height",
    )
    with pytest.raises(ValueError, match=r"no shard on executing cores.*\(1, 0\)"):
        invalid_local_tensor_shard_grid(inp, out)


def test_external_compute_tensor_accessors_prune_specialized_subsets(
    accessor_device,
):
    """Each specialized clone receives only tensors used by its core branch."""

    first_host = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    first_host = first_host.to(torch.bfloat16)
    second_host = 1000 - first_host

    def to_core_tensor(host_tensor, core_x):
        return _to_configured_tensor(
            host_tensor,
            accessor_device,
            ttnn.DataType.BFLOAT16,
            ttnn.TILE_LAYOUT,
            "l1",
            "height",
            core_x=core_x,
        )

    first_inp = to_core_tensor(first_host, 0)
    first_out = to_core_tensor(torch.zeros_like(first_host), 0)
    second_inp = to_core_tensor(second_host, 1)
    second_out = to_core_tensor(torch.zeros_like(second_host), 1)

    specialized_local_tensor_subsets(
        first_inp,
        first_out,
        second_inp,
        second_out,
        options="--ttl-specialize-cores",
    )

    assert_allclose(ttnn.to_torch(first_out).float(), first_host.float())
    assert_allclose(ttnn.to_torch(second_out).float(), second_host.float())


def test_external_compute_tensor_accessor_emitted_runner(
    accessor_device, monkeypatch, tmp_path
):
    """Emitted and ME2E runners preserve compute-local tensor metadata."""

    runner_path = tmp_path / "external_local_tensor_runner.py"
    monkeypatch.setenv("TTLANG_COMPILE_ONLY", "1")
    monkeypatch.setenv("TTLANG_EMIT_RUNNER", str(runner_path))

    host = _make_host_tensor(ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT)
    inp = _to_configured_tensor(
        host,
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "height",
    )
    out = _to_configured_tensor(
        torch.zeros_like(host),
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "height",
    )

    operation = _make_local_tensor_accessor_copy(
        _page_size_bytes(ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT)
    )
    operation(inp, out)
    assert runner_path.exists()

    module_spec = importlib.util.spec_from_file_location(
        "external_local_tensor_runner", runner_path
    )
    runner_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner_module)
    assert [0, 1] in runner_module.KERNEL_LOCAL_TENSOR_INDICES
    assert runner_module.TENSOR_CONFIGURATIONS == tuple(
        get_tensor_configuration(tensor) for tensor in (inp, out)
    )

    monkeypatch.delenv("TTLANG_COMPILE_ONLY")
    replay_out = _to_configured_tensor(
        torch.zeros_like(host),
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "height",
    )
    runner_module.run([inp, replay_out], device=accessor_device)
    assert_allclose(
        ttnn.to_torch(replay_out).float(), ttnn.to_torch(inp).float(), rtol=0, atol=0
    )

    from me2e.builder.kernels import KernelSpec as ME2EKernelSpec, ThreadType
    from me2e.builder.ttnn_runner import run_unary_op
    from me2e.config import BufferType, E2EConfig, MemoryLayout

    me2e_kernel_dir = tmp_path / "me2e_kernels"
    me2e_kernel_dir.mkdir()
    me2e_noc_kernels = []
    me2e_compute_kernel = None
    for kernel_index, (
        kernel_path_and_thread,
        tensor_indices,
        local_indices,
    ) in enumerate(
        zip(
            runner_module.KERNEL_PATHS,
            runner_module.KERNEL_TENSOR_INDICES,
            runner_module.KERNEL_LOCAL_TENSOR_INDICES,
            strict=True,
        )
    ):
        kernel_path, thread_type = kernel_path_and_thread
        if thread_type == "compute":
            kernel_name = "compute"
            me2e_thread_type = ThreadType.COMPUTE
        else:
            kernel_name = {0: "reader", 1: "writer"}[
                runner_module.KERNEL_NOC_INDICES[kernel_index]
            ]
            me2e_thread_type = ThreadType.NOC
        copied_kernel_path = me2e_kernel_dir / f"{kernel_name}.cpp"
        shutil.copyfile(kernel_path, copied_kernel_path)
        kernel_spec = ME2EKernelSpec(
            name=kernel_name,
            thread_type=me2e_thread_type,
            source=copied_kernel_path.read_text(),
            tensor_indices=tensor_indices,
            local_tensor_indices=local_indices,
        )
        if me2e_thread_type == ThreadType.COMPUTE:
            me2e_compute_kernel = kernel_spec
        else:
            me2e_noc_kernels.append(kernel_spec)

    assert me2e_compute_kernel is not None
    assert len(me2e_noc_kernels) == 2
    me2e_result = run_unary_op(
        device=accessor_device,
        noc_kernels=me2e_noc_kernels,
        compute_kernel=me2e_compute_kernel,
        input_a=host,
        kernel_dir=me2e_kernel_dir,
        config=E2EConfig(
            grid_shape=(1, 1),
            dtype=torch.bfloat16,
            memory_layout=MemoryLayout.HEIGHT_SHARDED,
            buffer_type=BufferType.L1,
        ),
    )
    assert_allclose(me2e_result.float(), host.float(), rtol=0, atol=0)


def test_external_data_movement_tensor_accessor_rejects_system_memory(
    accessor_device,
):
    """Data movement tensor access rejects a host TTNN tensor."""

    host_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
    )
    device_anchor = _to_configured_tensor(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        accessor_device,
        ttnn.DataType.BFLOAT16,
        ttnn.TILE_LAYOUT,
        "l1",
        "interleaved",
    )
    with pytest.raises(
        RuntimeError,
        match="data movement kernel uses SystemMemory storage.*require device DRAM or SRAM",
    ):
        invalid_data_movement_tensor_accessor(host_tensor, device_anchor)
