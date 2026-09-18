# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for PipeNet receiver bases across DFB reconfiguration."""

import pytest
import torch

import ttl
from ttl import ttl_api
from ttlang_test_utils import to_dram, to_l1, to_l1_sharded
from utils.correctness import assert_pcc

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device


def _make_reconfigured_receiver_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def reconfigured_receiver_operation(inp, tensor_backed_output, scratch_output):
        pipe = ttl.Pipe(src=(0, 0), dst=(0, 0))
        pipe_net = ttl.PipeNet([pipe])
        receiver_allocation = ttl.make_dfb_allocation_group()
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
        tensor_backed_receiver = ttl.make_tensor_backed_dfb(
            tensor_backed_output,
            shape=(1, 1),
            allocation_group=receiver_allocation,
        )
        scratch_receiver = ttl.make_dfb(
            data_format,
            shape=(1, 1),
            block_count=1,
            allocation_group=receiver_allocation,
        )

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reconfigure_dfbs(boundary)

        @ttl.datamovement(kernel=reader_kernel)
        def receive():
            if pipe_net.is_active():
                pass

            with tensor_backed_receiver.reserve() as receiver_block:
                receive_request = ttl.copy(pipe, receiver_block)
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()
                receive_request.wait()
            with tensor_backed_receiver.wait():
                pass

            ttl.reconfigure_dfbs(boundary)

            with scratch_receiver.reserve() as receiver_block:
                receive_request = ttl.copy(pipe, receiver_block)
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 1], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()
                receive_request.wait()
            with scratch_receiver.wait() as receiver_block:
                ttl.copy(receiver_block, scratch_output[0, 0]).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def synchronize():
            ttl.reconfigure_dfbs(boundary)

    return reconfigured_receiver_operation


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_reconfigured_receiver_uses_published_address(
    device, dtype, to_device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    host_input = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64).to(dtype)
    tensor_backed_output = to_l1_sharded(
        torch.zeros((32, 32), dtype=dtype), device, layout="height"
    )
    scratch_output = to_device(torch.zeros((32, 32), dtype=dtype), device)
    operation = _make_reconfigured_receiver_operation(
        "bf16" if dtype == torch.bfloat16 else "float32"
    )
    final_mlir_path = tmp_path / "reconfigured_pipe_receiver.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))

    operation(
        to_device(host_input, device),
        tensor_backed_output,
        scratch_output,
        options="--ttl-reuse-user-dfbs",
    )

    threshold = 0.999 if dtype == torch.bfloat16 else 0.99999
    assert_pcc(
        host_input[:, :32].float(),
        ttnn.to_torch(tensor_backed_output).float(),
        threshold=threshold,
    )
    assert_pcc(
        host_input[:, 32:].float(),
        ttnn.to_torch(scratch_output).float(),
        threshold=threshold,
    )
    final_mlir = final_mlir_path.read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 2
    assert "ttl.pipe_computed_address_dfb_indices" not in final_mlir
