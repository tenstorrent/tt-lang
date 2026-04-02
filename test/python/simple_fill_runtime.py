# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# CHECK: PASS

"""Runtime test for fill: fill 2x2 tiles with a negative constant and verify."""

import torch
import ttnn
import ttl
from ttlang_test_utils import to_l1


@ttl.operation(grid=(1, 1))
def fill_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as _in, out_dfb.reserve() as o:
            o.store(ttl.math.fill(o, -3.0))

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        ttl.copy(inp[0:2, 0:2], inp_blk).wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        ttl.copy(out_blk, out[0:2, 0:2]).wait()
        out_blk.pop()


def main():
    device = ttnn.open_device(device_id=0)

    inp = to_l1(torch.zeros((64, 64), dtype=torch.bfloat16), device)
    out = to_l1(torch.zeros((64, 64), dtype=torch.bfloat16), device)

    fill_kernel(inp, out)

    result = ttnn.to_torch(out)
    expected = torch.full((64, 64), -3.0, dtype=torch.bfloat16)

    if torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("PASS")
    else:
        diff = (result.float() - expected.float()).abs().max().item()
        print("FAIL: max error %.4f" % diff)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
