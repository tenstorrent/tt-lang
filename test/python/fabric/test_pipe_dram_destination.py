# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Point-to-point PipeNet receives into receiver-owned DRAM regions."""

from itertools import product
from math import prod

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh, to_dram
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32


def _make_direct_dram_receive(mesh_shape, block_shape):
    source_device = tuple(0 for _extent in mesh_shape)
    destination_device = tuple(extent - 1 for extent in mesh_shape)
    device_domain = ttl.DeviceDomain(mesh_shape)
    transfer_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain, edges=[(source_device, destination_device)]
        )
    )
    block_rows, block_columns = block_shape

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def direct_dram_receive(inp, out, observed):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=block_shape, block_count=2)
        readback_dfb = ttl.make_dataflow_buffer_like(
            out, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                send_block = send_dfb.reserve()
                ttl.copy(inp[0:block_rows, 0:block_columns], send_block).wait()
                ttl.copy(send_block, pipe).wait()

            transfer_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                receive_request = ttl.copy(
                    pipe,
                    out[0:block_rows, 0:block_columns],
                    shape=block_shape,
                )
                receive_request.wait()

                readback_block = readback_dfb.reserve()
                ttl.copy(out[0:block_rows, 0:block_columns], readback_block).wait()
                ready_readback_block = readback_dfb.wait()
                ttl.copy(
                    ready_readback_block,
                    observed[0:block_rows, 0:block_columns],
                ).wait()

            transfer_net.if_dst(receive)

    return direct_dram_receive


def _make_repeated_direct_dram_receive(
    mesh_shape, block_shape, repeat_count, disjoint_destinations
):
    source_device = tuple(0 for _extent in mesh_shape)
    destination_device = tuple(extent - 1 for extent in mesh_shape)
    device_domain = ttl.DeviceDomain(mesh_shape)
    transfer_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain, edges=[(source_device, destination_device)]
        )
    )
    block_rows, block_columns = block_shape
    destinations_per_physical_row = repeat_count if disjoint_destinations else 1
    occurrence_stride = 1 if disjoint_destinations else 0

    @ttl.operation(grid=(1, 2), device_domain=device_domain)
    def repeated_direct_dram_receive(inp, staging, observed):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=block_shape, block_count=2)
        readback_dfb = ttl.make_dataflow_buffer_like(
            staging, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                for repeat_index in range(repeat_count):
                    row_begin = repeat_index * block_rows
                    with send_dfb.reserve() as send_block:
                        ttl.copy(
                            inp[
                                row_begin : row_begin + block_rows,
                                0:block_columns,
                            ],
                            send_block,
                        ).wait()
                    with send_dfb.wait() as send_block:
                        ttl.copy(send_block, pipe).wait()

            transfer_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            _, physical_row = ttl.node(dims=2)

            def receive(pipe):
                for repeat_index in range(repeat_count):
                    staging_row = (
                        physical_row * destinations_per_physical_row
                        + repeat_index * occurrence_stride
                    )
                    staging_row_begin = staging_row * block_rows
                    staging_region = staging[
                        staging_row_begin : staging_row_begin + block_rows,
                        0:block_columns,
                    ]
                    receive_request = ttl.copy(
                        pipe,
                        staging_region,
                        shape=block_shape,
                    )
                    receive_request.wait()
                    with readback_dfb.reserve() as readback_block:
                        ttl.copy(staging_region, readback_block).wait()
                    observed_row_begin = repeat_index * block_rows
                    with readback_dfb.wait() as readback_block:
                        ttl.copy(
                            readback_block,
                            observed[
                                observed_row_begin : observed_row_begin + block_rows,
                                0:block_columns,
                            ],
                        ).wait()

            transfer_net.if_dst(receive)

    return repeated_direct_dram_receive


def _make_concurrent_bidirectional_direct_dram_receive(mesh_shape, worker_count):
    device_domain = ttl.DeviceDomain(mesh_shape)
    source_devices = tuple(product(*(range(extent) for extent in mesh_shape)))
    destination_devices = source_devices[1:] + source_devices[:1]
    worker_nodes = tuple((0, worker_index) for worker_index in range(worker_count))
    worker_pipes = tuple(ttl.Pipe(src=node, dst=node) for node in worker_nodes)
    forward_net = ttl.PipeNet(
        pipes=worker_pipes,
        graph=ttl.TransferGraph.edges(
            device_domain,
            edges=list(zip(source_devices, destination_devices, strict=True)),
        ),
    )
    reverse_net = ttl.PipeNet(
        pipes=worker_pipes,
        graph=ttl.TransferGraph.edges(
            device_domain,
            edges=list(zip(destination_devices, source_devices, strict=True)),
        ),
    )

    @ttl.operation(grid=(1, worker_count), device_domain=device_domain)
    def concurrent_bidirectional_direct_dram_receive(inp, staging, observed):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
        readback_dfb = ttl.make_dataflow_buffer_like(
            staging, shape=(1, 1), block_count=1
        )

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            _, physical_row = ttl.node(dims=2)

            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(
                        inp[physical_row : physical_row + 1, 0:1], send_block
                    ).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            if physical_row % 2 == 0:
                forward_net.if_src(send)
            else:
                reverse_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            _, physical_row = ttl.node(dims=2)
            destination = staging[physical_row : physical_row + 1, 0:1]

            def receive(pipe):
                ttl.copy(pipe, destination, shape=(1, 1)).wait()
                with readback_dfb.reserve() as readback_block:
                    ttl.copy(destination, readback_block).wait()
                with readback_dfb.wait() as readback_block:
                    ttl.copy(
                        readback_block,
                        observed[physical_row : physical_row + 1, 0:1],
                    ).wait()

            if physical_row % 2 == 0:
                forward_net.if_dst(receive)
            else:
                reverse_net.if_dst(receive)

    return concurrent_bidirectional_direct_dram_receive


def _make_repeated_one_to_many_direct_dram_receive(mesh_shape, repeat_count):
    device_domain = ttl.DeviceDomain(mesh_shape)
    device_coordinates = tuple(product(*(range(extent) for extent in mesh_shape)))
    source_device = device_coordinates[0]
    destination_devices = device_coordinates[1:3]
    transfer_net = ttl.PipeNet(
        pipes=[ttl.Pipe(src=(0, 0), dst=(0, 0))],
        graph=ttl.TransferGraph.edges(
            device_domain,
            edges=[
                (source_device, destination_device)
                for destination_device in destination_devices
            ],
        ),
    )

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def repeated_one_to_many_direct_dram_receive(inp, staging, observed):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        readback_dfb = ttl.make_dataflow_buffer_like(
            staging, shape=(1, 1), block_count=2
        )

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                for repeat_index in range(repeat_count):
                    with send_dfb.reserve() as send_block:
                        ttl.copy(
                            inp[repeat_index : repeat_index + 1, 0:1], send_block
                        ).wait()
                    with send_dfb.wait() as send_block:
                        ttl.copy(send_block, pipe).wait()

            transfer_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                for repeat_index in range(repeat_count):
                    destination = staging[repeat_index : repeat_index + 1, 0:1]
                    ttl.copy(pipe, destination, shape=(1, 1)).wait()
                    with readback_dfb.reserve() as readback_block:
                        ttl.copy(destination, readback_block).wait()
                    with readback_dfb.wait() as readback_block:
                        ttl.copy(
                            readback_block,
                            observed[repeat_index : repeat_index + 1, 0:1],
                        ).wait()

            transfer_net.if_dst(receive)

    return repeated_one_to_many_direct_dram_receive


@pytest.mark.parametrize(
    "torch_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "block_shape",
    [(1, 1), (2, 2), (1, 5), (2, 3)],
    ids=[
        "one-tile",
        "one-scatter-packet",
        "scatter-plus-unicast",
        "two-scatter-packets",
    ],
)
def test_pipe_receive_to_dram_region(torch_dtype, rtol, atol, block_shape):
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    device_count = prod(mesh_shape)
    if device_count < 2:
        pytest.skip("requires multiple devices")
    block_rows, block_columns = block_shape
    shard_shape = (block_rows * TILE_SIZE, block_columns * TILE_SIZE)
    logical_shape = (device_count * shard_shape[0], shard_shape[1])
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)
    direct_dram_receive = _make_direct_dram_receive(mesh_shape, block_shape)

    with open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as mesh:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        inp = to_dram(
            inp_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )
        out = to_dram(
            out_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )
        observed = to_dram(
            out_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )

        direct_dram_receive(inp, out, observed)

        result = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )
        observed_result = ttnn.to_torch(
            observed,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    expected = torch.zeros_like(out_torch)
    expected[-shard_shape[0] :, :] = inp_torch[: shard_shape[0], :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
    assert_allclose(observed_result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "torch_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "block_shape",
    [(1, 1), (1, 5)],
    ids=["one-tile", "scatter-plus-unicast"],
)
@pytest.mark.parametrize(
    "disjoint_destinations",
    [False, True],
    ids=["reused-destination", "disjoint-destinations"],
)
def test_repeated_pipe_receive_to_dram_region(
    torch_dtype, rtol, atol, block_shape, disjoint_destinations
):
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    device_count = prod(mesh_shape)
    if device_count < 2:
        pytest.skip("requires multiple devices")
    repeat_count = 3
    block_rows, block_columns = block_shape
    input_shard_shape = (
        repeat_count * block_rows * TILE_SIZE,
        block_columns * TILE_SIZE,
    )
    destination_count = repeat_count if disjoint_destinations else 1
    staging_shard_shape = (
        2 * destination_count * block_rows * TILE_SIZE,
        block_columns * TILE_SIZE,
    )
    inp_torch = torch.randn(
        (device_count * input_shard_shape[0], input_shard_shape[1]),
        dtype=torch_dtype,
    )
    staging_torch = torch.zeros(
        (device_count * staging_shard_shape[0], staging_shard_shape[1]),
        dtype=torch_dtype,
    )
    observed_torch = torch.zeros_like(inp_torch)
    repeated_direct_dram_receive = _make_repeated_direct_dram_receive(
        mesh_shape, block_shape, repeat_count, disjoint_destinations
    )

    with open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as mesh:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        inp = to_dram(
            inp_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )
        staging = to_dram(
            staging_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )
        observed = to_dram(
            observed_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )

        repeated_direct_dram_receive(inp, staging, observed)

        result = ttnn.to_torch(
            observed,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    expected = torch.zeros_like(inp_torch)
    expected[-input_shard_shape[0] :, :] = inp_torch[: input_shard_shape[0], :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "torch_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
def test_concurrent_bidirectional_pipe_receive_to_disjoint_dram_regions(
    torch_dtype, rtol, atol
):
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    if mesh_shape[0] < 2:
        pytest.skip("requires a one-dimensional multi-device mesh")
    worker_count = 4
    shard_shape = (worker_count * TILE_SIZE, TILE_SIZE)
    logical_shape = (prod(mesh_shape) * shard_shape[0], shard_shape[1])
    input_device_shards = torch.empty(
        (prod(mesh_shape), *shard_shape), dtype=torch_dtype
    )
    for device_index in range(prod(mesh_shape)):
        for worker_index in range(worker_count):
            row_begin = worker_index * TILE_SIZE
            input_device_shards[device_index, row_begin : row_begin + TILE_SIZE, :] = (
                device_index * worker_count + worker_index + 1
            )
    inp_torch = input_device_shards.reshape(logical_shape)
    zero_torch = torch.zeros(logical_shape, dtype=torch_dtype)
    concurrent_direct_dram_receive = _make_concurrent_bidirectional_direct_dram_receive(
        mesh_shape, worker_count
    )

    with open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as mesh:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        inp = to_dram(
            inp_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )
        staging = to_dram(
            zero_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )
        observed = to_dram(
            zero_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )

        concurrent_direct_dram_receive(inp, staging, observed)
        result = ttnn.to_torch(
            observed,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    input_worker_tiles = input_device_shards.reshape(
        prod(mesh_shape), worker_count, TILE_SIZE, TILE_SIZE
    )
    expected_worker_tiles = torch.empty_like(input_worker_tiles)
    expected_worker_tiles[:, 0::2] = torch.roll(
        input_worker_tiles[:, 0::2], shifts=1, dims=0
    )
    expected_worker_tiles[:, 1::2] = torch.roll(
        input_worker_tiles[:, 1::2], shifts=-1, dims=0
    )
    expected = expected_worker_tiles.reshape(logical_shape)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "torch_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
def test_repeated_one_to_many_pipe_receive_uses_per_record_occurrence_counters(
    torch_dtype, rtol, atol
):
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    device_count = prod(mesh_shape)
    if device_count < 3:
        pytest.skip("requires at least three devices")
    repeat_count = 2
    shard_shape = (repeat_count * TILE_SIZE, TILE_SIZE)
    logical_shape = (device_count * shard_shape[0], shard_shape[1])
    input_device_shards = torch.zeros((device_count, *shard_shape), dtype=torch_dtype)
    input_device_shards[0] = torch.randn(shard_shape, dtype=torch_dtype)
    inp_torch = input_device_shards.reshape(logical_shape)
    zero_torch = torch.zeros(logical_shape, dtype=torch_dtype)
    repeated_receive = _make_repeated_one_to_many_direct_dram_receive(
        mesh_shape, repeat_count
    )

    with open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as mesh:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        inp = to_dram(
            inp_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )
        staging = to_dram(
            zero_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )
        observed = to_dram(
            zero_torch,
            mesh,
            mesh_mapper=mesh_mapper,
        )

        repeated_receive(inp, staging, observed)
        result = ttnn.to_torch(
            observed,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    expected_device_shards = torch.zeros_like(input_device_shards)
    expected_device_shards[1:3] = input_device_shards[0]
    expected = expected_device_shards.reshape(logical_shape)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
