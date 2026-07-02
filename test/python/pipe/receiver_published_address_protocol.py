# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILER_OPTIONS=--no-ttl-pipe-computed-addresses TTLANG_FINAL_MLIR=%t.final.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --input-file=%t.final.mlir --check-prefix=FINAL
# RUN: FileCheck %s --input-file=%t.output --check-prefix=CPP

"""Receiver-published address protocol coverage for PipeNet lowering.

Disabling computed receiver addresses keeps the receiver-published protocol for
the same tiled point-to-point source that is otherwise eligible for computed
addressing. The test executes on device and compares the result with a torch
expected value.
"""

import torch  # noqa: E402
import ttnn  # noqa: E402

import ttl  # noqa: E402
from utils.correctness import assert_pcc  # noqa: E402

TILE = 32


def _make_input(shape):
    numel = 1
    for extent in shape:
        numel *= extent
    return torch.arange(numel, dtype=torch.float32).reshape(shape).to(torch.bfloat16)


def _device_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.operation(grid=(2, 1))
def receiver_published_address(inp, out):
    pipe = ttl.Pipe(src=(0, 0), dst=(1, 0))
    net = ttl.PipeNet([pipe])

    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)

        def send(pipe_arg):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 0], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe_arg).wait()

        net.if_src(send)

        if node_x == 1:
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0]).wait()

    @ttl.datamovement()
    def dm_brisc():
        pass


# The compiler option keeps receiver-authored DFB addresses as the lowering
# contract.
# FINAL-LABEL: func.func @dm
# FINAL-NOT: ttl.pipe_computed_address_dfb_indices
# FINAL-NOT: .down(
# FINAL: experimental::semaphore_wait
# FINAL: noc_semaphore_set
# FINAL: noc0.inline_dw_write<NocOptions::INLINE_L1>
# FINAL: experimental::semaphore_wait_min

# CPP: noc0.async_write(
# CPP: noc0.inline_dw_write<NocOptions::INLINE_L1>
# CPP: noc0.async_write_barrier


def main():
    device = ttnn.open_device(device_id=0)
    try:
        inp_torch = _make_input((TILE, TILE))
        out_torch = torch.zeros((TILE, TILE), dtype=torch.bfloat16)
        inp = _device_ttnn(inp_torch, device)
        out = _device_ttnn(out_torch, device)
        receiver_published_address(inp, out)
        ttnn.synchronize_device(device)
        assert_pcc(inp_torch.float(), ttnn.to_torch(out).float())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
