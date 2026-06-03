# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Metal side of the flash-shard cycle A/B: tt-metal's ``compute_sdpa_chunk``.

Drives the public ``compute_sdpa_chunk`` primitive (sdpa.h, referenced in-place)
on one core over the same per-core decode slice as ``ttl_shard.py``, via
``ttnn.generic_op`` with three plain kernel files. Q/K/out/stats are single-core
L1 shards (no DRAM streaming) so the whole K slice stays resident and the
measured cycles are the compute, not the reader. Run on hardware:

    TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.flash_shard.metal_sdpa
"""

import os
import struct

import torch

import ttnn

from benchmarks.common import (
    clear_profile_log,
    parse_kernel_duration,
    read_device_profiler,
)

from . import shapes
from .shapes import DHt, PNHt, SCALE, Sk_chunk_t, TILE, vDHt

_KERNELS = os.path.join(os.path.dirname(__file__), "kernels")
# sdpa.h lives here in the public tt-metal tree; referenced in-place so we track
# any upstream improvements to compute_sdpa_chunk.
_SDPA_INCLUDE = "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include"

_FILE = ttnn.KernelDescriptor.SourceType.FILE_PATH


def _float_to_uint32(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def _shard1(t, device):
    """Single-core HEIGHT-sharded L1 tensor (the CB aliases this shard)."""
    h, w = t.shape[-2], t.shape[-1]
    grid = ttnn.num_cores_to_corerangeset(
        1, device.compute_with_storage_grid_size(), row_wise=True
    )
    mem = ttnn.create_sharded_memory_config(
        shape=[h, w], core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )


def run(n_chunks=shapes.N_CHUNKS):
    """Run compute_sdpa_chunk once with the profiler on; return cycles/us/per_risc/pcc."""
    torch.manual_seed(7)
    PN, D, vD = PNHt * TILE, DHt * TILE, vDHt * TILE
    S = Sk_chunk_t * n_chunks * TILE
    num_tiles_k, num_tiles_v, chunk_size, num_tiles_stats = DHt, vDHt, Sk_chunk_t, 1
    cb_q, cb_k, cb_out, cb_stats = 0, 1, 2, 3

    device = ttnn.open_device(device_id=0)
    try:
        q_t = torch.randn(PN, D, dtype=torch.bfloat16) * 0.1
        k_t = torch.randn(S, D, dtype=torch.bfloat16) * 0.1
        q = _shard1(q_t, device)
        k = _shard1(k_t, device)
        out = _shard1(torch.zeros(PN, vD, dtype=torch.bfloat16), device)
        stats = _shard1(torch.zeros(PN, TILE, dtype=torch.bfloat16), device)
        cores = q.memory_config().shard_spec.grid

        cbs = [
            ttnn.cb_descriptor_from_sharded_tensor(cb_q, q),
            ttnn.cb_descriptor_from_sharded_tensor(cb_k, k),
            ttnn.cb_descriptor_from_sharded_tensor(cb_out, out),
            ttnn.cb_descriptor_from_sharded_tensor(cb_stats, stats),
        ]

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/reader.cpp", source_type=_FILE, core_ranges=cores,
            compile_time_args=[cb_q, cb_k, chunk_size, n_chunks, num_tiles_k],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/writer.cpp", source_type=_FILE, core_ranges=cores,
            compile_time_args=[cb_out, cb_stats, num_tiles_v, num_tiles_stats],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/compute.cpp", source_type=_FILE, core_ranges=cores,
            compile_time_args=[
                cb_q, cb_k, cb_out, cb_stats,
                chunk_size, n_chunks, num_tiles_k, num_tiles_v, num_tiles_stats,
                _float_to_uint32(SCALE),
            ],
            compiler_include_paths=[os.path.join(os.environ["TT_METAL_HOME"], _SDPA_INCLUDE)],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                fp32_dest_acc_en=False, dst_full_sync_en=False,
            ),
        )

        prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs)

        clear_profile_log()
        ttnn.generic_op([q, k, out, stats], prog)
        ttnn.synchronize_device(device)
        read_device_profiler(device)
        o_got = ttnn.to_torch(out).reshape(PN, vD).float()
    finally:
        ttnn.close_device(device)

    # Golden: unnormalized O = exp((scores - max) * scale) @ V, V = leading vDHt
    # tiles of K (MLA coupling). Matches compute_sdpa_chunk (scale fused in exp).
    scores = q_t.float() @ k_t.float().T
    gmax = scores.max(dim=-1, keepdim=True).values
    o_ref = torch.exp((scores - gmax) * SCALE) @ k_t.float()[:, :vD]
    pcc = torch.corrcoef(torch.stack([o_got.flatten(), o_ref.flatten()]))[0, 1].item()

    d = parse_kernel_duration()
    d["pcc"] = pcc
    return d


def main():
    d = run()
    shapes.print_result("compute_sdpa_chunk (metal)", d, shapes.N_CHUNKS)


if __name__ == "__main__":
    main()
