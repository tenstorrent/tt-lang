# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Standalone tt-metal (ttnn.generic_op) harness for the resident-intermediate
# PACK->UNPACK sync question. Single core. The compute kernel loops N_ITERS
# times, each: reader streams a fresh (32,128) chunk into cb0; compute does
# y=2*x -> pack to cb2 (RESIDENT scratch, reused, never pushed/popped) ->
# transpose_wh -> cb1; writer drains cb1. Distinct data per iteration, so a
# PACK->UNPACK race that reads the prior iteration's stale cb2 shows up as a
# PCC drop. cb0/cb1 are double-buffered so the reader prefetches (the compute's
# wait_front does not stall the unpacker -- maximizing race exposure).
#
# env: CBSYNC_MODE in {baseline,handshake,barrier,nops}; CBSYNC_ITERS (default 64);
#      CBSYNC_SEED (default 1234); CBSYNC_NOPS (default 32); CBSYNC_COMPUTE.
#
#   python3 /tmp/cbsync/run_cbsync.py

import os

import torch
import ttnn

_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_FILE = ttnn.KernelDescriptor.SourceType.FILE_PATH
_COMPUTE = os.environ.get("CBSYNC_COMPUTE", "compute.cpp")

TILE = 32
NT = 4  # per-iteration: input (1, NT) tiles; output (NT, 1) tiles
MODE = os.environ.get("CBSYNC_MODE", "baseline")
ITERS = int(os.environ.get("CBSYNC_ITERS", "64"))
SEED = int(os.environ.get("CBSYNC_SEED", "1234"))
NOPS = os.environ.get("CBSYNC_NOPS", "32")


def _compute_defines():
    if MODE == "handshake":
        return [("HANDSHAKE", "1")]
    if MODE == "barrier":
        return [("BARRIER", "1")]
    if MODE == "nops":
        return [("NOPS", NOPS)]
    return []  # baseline


def run():
    torch.manual_seed(SEED)
    cb_in, cb_out, cb_scratch = 0, 1, 2
    tile = ttnn.Tile([TILE, TILE])
    tile_bytes = TILE * TILE * 2  # bf16

    # Stacked per-iteration chunks. Input chunk it = tiles [it*NT, it*NT+NT) =
    # rows [it*TILE, (it+1)*TILE). Output chunk it = rows [it*NT*TILE, ...).
    in_shape = (ITERS * TILE, NT * TILE)          # (ITERS*32, 128)
    out_shape = (ITERS * NT * TILE, TILE)         # (ITERS*128, 32)

    x = torch.randn(in_shape, dtype=torch.bfloat16)
    golden = torch.zeros(out_shape, dtype=torch.float32)
    for it in range(ITERS):
        chunk = x[it * TILE:(it + 1) * TILE, :].float()      # (32,128)
        golden[it * NT * TILE:(it + 1) * NT * TILE, :] = (chunk * 2.0).T  # (128,32)

    device = ttnn.open_device(device_id=0)
    try:
        inp = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, tile=tile,
        )
        out = ttnn.from_torch(
            torch.zeros(out_shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, tile=tile,
        )

        cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}
        )

        def _cb(idx, nblocks):
            return ttnn.CBDescriptor(
                total_size=nblocks * NT * tile_bytes, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes,
                )],
            )

        # cb0/cb1 double-buffered (prefetch); cb2 single block (resident scratch).
        cbs = [_cb(cb_in, 2), _cb(cb_out, 2), _cb(cb_scratch, 1)]

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/reader.cpp", source_type=_FILE, core_ranges=cores,
            compile_time_args=([cb_in, NT, ITERS] + list(ttnn.TensorAccessorArgs(inp).get_compile_time_args())),
            common_runtime_args=[inp.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/writer.cpp", source_type=_FILE, core_ranges=cores,
            compile_time_args=([cb_out, NT, ITERS] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())),
            common_runtime_args=[out.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/{_COMPUTE}", source_type=_FILE, core_ranges=cores,
            compile_time_args=[cb_in, cb_out, cb_scratch, ITERS],
            defines=_compute_defines(),
            config=ttnn.ComputeConfigDescriptor(),
        )

        prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs)
        ttnn.generic_op([inp, out], prog)
        ttnn.synchronize_device(device)
        got = ttnn.to_torch(out).reshape(out_shape).float()
    finally:
        ttnn.close_device(device)

    g, a = golden.flatten(), got.flatten()
    pcc = torch.corrcoef(torch.stack([g, a]))[0, 1].item()
    # per-iteration PCC to see if corruption is iteration-dependent
    bad = []
    for it in range(ITERS):
        sl = slice(it * NT * TILE, (it + 1) * NT * TILE)
        gi, ai = golden[sl].flatten(), got[sl].flatten()
        pi = torch.corrcoef(torch.stack([gi, ai]))[0, 1].item()
        if pi < 0.99:
            bad.append((it, round(pi, 3)))
    print(f"CBSYNC_MODE={MODE} ITERS={ITERS} SEED={SEED}  overall_PCC={pcc:.6f}  bad_iters={len(bad)}/{ITERS}  first_bad={bad[:6]}")
    return pcc


if __name__ == "__main__":
    run()
