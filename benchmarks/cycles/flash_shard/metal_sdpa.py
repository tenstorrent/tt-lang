# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Metal side of the flash-shard cycle A/B: tt-metal's ``compute_sdpa_chunk``.

Drives the public ``compute_sdpa_chunk`` primitive (sdpa.h, referenced in-place)
on one core over the same per-core decode slice as ``ttl_shard.py``, via
``ttnn.generic_op`` with three plain kernel files. Q/out/stats are single-core
L1 shards; K streams from a DRAM-interleaved tensor into a double-buffered cb_k
(the reader mirrors tt-lang's emitted ncrisc DRAM stream), so the per-core K
slice need not be L1-resident and the baseline reaches 32k+ seq. Run on hardware:

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
# sdpa.h lives in the tt-metal submodule (third-party/tt-metal); referenced
# in-place so we track upstream improvements to compute_sdpa_chunk. The custom
# LLK subtree it pulls in is header-only, so the submodule need not be built.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SDPA_INCLUDE = os.environ.get(
    "CYCLES_SDPA_INCLUDE",
    os.path.join(
        _REPO_ROOT,
        "third-party/tt-metal/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include",
    ),
)

_FILE = ttnn.KernelDescriptor.SourceType.FILE_PATH

# Harness-isolation toggle: builds the compute kernel with -DSDPA_NOOP so it only
# drives the CB protocol (no compute_sdpa_chunk), to test the generic_op plumbing.
_NOOP = os.environ.get("CYCLES_NOOP") == "1"


def _float_to_uint32(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


# Q/out/stats use an 8-row tile -- compute_sdpa_chunk's 8x32 packing (the decode
# query row); K uses a full 32x32 tile. This matches the deepseek sdpa test.
Q_TILE_H = 8

# cb_k holds CB_BLOCKS chunks at once (>= 2 so the reader's next DRAM chunk
# overlaps the compute of the current one; raise if the stream becomes a bottleneck).
CB_BLOCKS = int(os.environ.get("CYCLES_CB_BLOCKS", "2"))


def _shard(torch_t, device, tile):
    """Single-core (0,0) HEIGHT-sharded L1 tensor; its CB aliases this shard."""
    h, w = torch_t.shape
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [h, w],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec
    )
    return ttnn.from_torch(
        torch_t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )


def run(num_chunks=shapes.METAL_NUM_CHUNKS, chunk_size=shapes.METAL_CHUNK_SIZE):
    """Run compute_sdpa_chunk once with the profiler on; return cycles/us/per_risc/pcc."""
    torch.manual_seed(7)
    num_tiles_k, num_tiles_v, num_tiles_stats = DHt, vDHt, 1
    cb_q, cb_k, cb_out, cb_stats = 0, 1, 2, 3

    q_tile = ttnn.Tile([Q_TILE_H, TILE])
    k_tile = ttnn.Tile([TILE, TILE])
    q_shape = (Q_TILE_H, num_tiles_k * TILE)
    k_shape = (num_chunks * chunk_size * TILE, num_tiles_k * TILE)
    out_shape = (Q_TILE_H, num_tiles_v * TILE)
    stats_shape = (Q_TILE_H, TILE)

    # bfp8 K matches the ttl shard's KV-cache dtype (fair A/B) and halves the DRAM
    # read traffic vs bf16 (1088 vs 2048 B/tile), keeping the stream off the
    # critical path; Q stays bf16. 32x32 bfp8 = 1024 mantissa + 64 exponent bytes.
    k_page_bytes = TILE * TILE + (TILE * TILE) // 16
    tiles_per_chunk = num_tiles_k * chunk_size

    device = ttnn.open_device(device_id=0)
    try:
        q_t = torch.randn(q_shape, dtype=torch.bfloat16) * 0.1
        k_t = torch.randn(k_shape, dtype=torch.bfloat16) * 0.1
        q = _shard(q_t, device, q_tile)
        # K is DRAM-interleaved: the reader streams it tile-by-tile into cb_k.
        k = ttnn.from_torch(
            k_t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tile=k_tile,
        )
        out = _shard(torch.zeros(out_shape, dtype=torch.bfloat16), device, q_tile)
        stats = _shard(torch.zeros(stats_shape, dtype=torch.bfloat16), device, q_tile)
        cores = q.memory_config().shard_spec.grid

        # cb_k is program-allocated (double-buffered DRAM stream target); the
        # rest alias their single-core L1 shards.
        cb_k_desc = ttnn.CBDescriptor(
            total_size=CB_BLOCKS * tiles_per_chunk * k_page_bytes,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_k,
                    data_format=ttnn.bfloat8_b,
                    page_size=k_page_bytes,
                )
            ],
        )
        cbs = [
            ttnn.cb_descriptor_from_sharded_tensor(cb_q, q),
            cb_k_desc,
            ttnn.cb_descriptor_from_sharded_tensor(cb_out, out),
            ttnn.cb_descriptor_from_sharded_tensor(cb_stats, stats),
        ]

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/reader.cpp",
            source_type=_FILE,
            core_ranges=cores,
            compile_time_args=(
                [cb_q, cb_k, chunk_size, num_chunks, num_tiles_k]
                + list(ttnn.TensorAccessorArgs(k).get_compile_time_args())
            ),
            common_runtime_args=[k.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/writer.cpp",
            source_type=_FILE,
            core_ranges=cores,
            compile_time_args=[cb_out, cb_stats, num_tiles_v, num_tiles_stats],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/compute.cpp",
            source_type=_FILE,
            core_ranges=cores,
            compile_time_args=[
                cb_q,
                cb_k,
                cb_out,
                cb_stats,
                chunk_size,
                num_chunks,
                num_tiles_k,
                num_tiles_v,
                num_tiles_stats,
                _float_to_uint32(SCALE),
            ],
            compiler_include_paths=[_SDPA_INCLUDE],
            defines=[("SDPA_NOOP", "1")] if _NOOP else [],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )

        prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs)

        clear_profile_log()
        ttnn.generic_op([q, k, out, stats], prog)
        ttnn.synchronize_device(device)
        read_device_profiler(device)
        o_got = ttnn.to_torch(out).reshape(out_shape).float()
    finally:
        ttnn.close_device(device)

    d = parse_kernel_duration()
    if _NOOP:
        # Output is garbage in the harness-isolation path; skip correctness.
        d["pcc"] = float("nan")
        return d

    # Golden: unnormalized O = exp((scores - max) * scale) @ V, V = leading vDHt
    # tiles of K (MLA coupling). Matches compute_sdpa_chunk (scale fused in exp).
    scores = q_t.float() @ k_t.float().T
    gmax = scores.max(dim=-1, keepdim=True).values
    o_ref = torch.exp((scores - gmax) * SCALE) @ k_t.float()[:, : num_tiles_v * TILE]
    pcc = torch.corrcoef(torch.stack([o_got.flatten(), o_ref.flatten()]))[0, 1].item()
    d["pcc"] = pcc
    return d


def main():
    d = run()
    shapes.print_result("compute_sdpa_chunk (metal)", d, shapes.N_CHUNKS)


if __name__ == "__main__":
    main()
