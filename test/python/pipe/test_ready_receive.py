# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for selecting among completed PipeNet receives."""

import pytest
import torch

import ttl
from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device


@ttl.operation(grid=(1, 1))
def ready_receive_ascending_tie(inp, landing0, landing1, landing2, landing3, out):
    pipe0 = ttl.Pipe(src=(0, 0), dst=(0, 0))
    pipe1 = ttl.Pipe(src=(0, 0), dst=(0, 0))
    pipe2 = ttl.Pipe(src=(0, 0), dst=(0, 0))
    pipe3 = ttl.Pipe(src=(0, 0), dst=(0, 0))
    net0 = ttl.PipeNet([pipe0])
    net1 = ttl.PipeNet([pipe1])
    net2 = ttl.PipeNet([pipe2])
    net3 = ttl.PipeNet([pipe3])

    input_dfb = ttl.make_tensor_backed_dfb(inp, shape=(1, 1), block_count=4)
    landing_dfb0 = ttl.make_tensor_backed_dfb(landing0, shape=(1, 1), block_count=1)
    landing_dfb1 = ttl.make_tensor_backed_dfb(landing1, shape=(1, 1), block_count=1)
    landing_dfb2 = ttl.make_tensor_backed_dfb(landing2, shape=(1, 1), block_count=1)
    landing_dfb3 = ttl.make_tensor_backed_dfb(landing3, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def data_movement():
        if net0.is_active():
            pass
        if net1.is_active():
            pass
        if net2.is_active():
            pass
        if net3.is_active():
            pass

        block0 = landing_dfb0.reserve()
        block1 = landing_dfb1.reserve()
        block2 = landing_dfb2.reserve()
        block3 = landing_dfb3.reserve()
        request0 = ttl.copy(pipe0, block0)
        request1 = ttl.copy(pipe1, block1)
        request2 = ttl.copy(pipe2, block2)
        request3 = ttl.copy(pipe3, block3)

        input_for_pipe1 = input_dfb.wait()
        ttl.copy(input_for_pipe1, pipe1).wait()
        input_for_pipe1.pop()
        input_for_pipe3 = input_dfb.wait()
        ttl.copy(input_for_pipe3, pipe3).wait()
        input_for_pipe3.pop()

        ready = ttl.wait_any((request0, request1, request2, request3), start=2)
        selected = ready.index()

        input_for_pipe0 = input_dfb.wait()
        ttl.copy(input_for_pipe0, pipe0).wait()
        input_for_pipe0.pop()
        input_for_pipe2 = input_dfb.wait()
        ttl.copy(input_for_pipe2, pipe2).wait()
        input_for_pipe2.pop()

        request0.wait()
        block0.push()
        request1.wait()
        block1.push()
        request2.wait()
        block2.push()
        request3.wait()
        block3.push()

        if selected == 0:
            result0 = landing_dfb0.wait()
            ttl.copy(result0, out[0, 0]).wait()
            result0.pop()
        elif selected == 1:
            result1 = landing_dfb1.wait()
            ttl.copy(result1, out[0, 0]).wait()
            result1.pop()
        elif selected == 2:
            result2 = landing_dfb2.wait()
            ttl.copy(result2, out[0, 0]).wait()
            result2.pop()
        else:
            if selected == 3:
                result3 = landing_dfb3.wait()
                ttl.copy(result3, out[0, 0]).wait()
                result3.pop()

    @ttl.datamovement()
    def data_movement_second():
        input_dfb.publish()


def _to_height_sharded(torch_tensor, device):
    dram_tensor = to_dram(torch_tensor, device)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        torch_tensor.shape[-2:],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    return ttnn.to_memory_config(dram_tensor, memory_config=memory_config)


@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_ready_receive_ascending_tie(device, torch_dtype):
    """Selection scans upward from index two when one and three are complete."""
    torch.manual_seed(0)
    input_torch = torch.rand((32, 128), dtype=torch_dtype)
    expected = input_torch[:, 32:64]
    inp = _to_height_sharded(input_torch, device)
    landing_tensors = [
        _to_height_sharded(torch.zeros_like(expected), device) for _ in range(4)
    ]
    out = to_dram(torch.zeros_like(expected), device)

    ready_receive_ascending_tie(inp, *landing_tensors, out)

    actual_landing = torch.cat(
        [ttnn.to_torch(landing) for landing in landing_tensors], dim=1
    )
    actual = ttnn.to_torch(out)
    threshold = 0.999 if torch_dtype == torch.bfloat16 else 0.99999
    expected_landing = torch.cat(
        [
            input_torch[:, 64:96],
            input_torch[:, :32],
            input_torch[:, 96:128],
            input_torch[:, 32:64],
        ],
        dim=1,
    )
    assert_pcc(expected_landing.float(), actual_landing.float(), threshold=threshold)
    assert_pcc(expected.float(), actual.float(), threshold=threshold)
