# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""C++ unicast baselines for the PipeNet address-protocol benchmark.

One hand-written core-to-core unicast (sender on (0,0), receiver on (1,0)) run
through ``ttnn.generic_op``. Per transfer the sender reads the source tile from
DRAM and NoC-writes it to the receiver's L1, matching the tt-lang variant's
per-transfer work so the per-transfer costs compare directly. The same N sweep,
single-pass device profiler measurement, bit-exact check, and CSV schema as
``ttlang_pipes.py`` let ``compare.py`` fit the variants together.

    python -m benchmarks.microbench.pipes.baseline_pipes --transfers 1,2,4,8,16,32,64,128,256
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

from pathlib import Path

import torch
import ttnn

from benchmarks.common import create_benchmark_arg_parser
from benchmarks.microbench import harness
from benchmarks.microbench.harness import TILE
from benchmarks.microbench.ttlang import ttl_harness

DEFAULT_CSV = "benchmarks/microbench/results/pipes_baseline.csv"
# mode -> variant name written to the CSV.
VARIANTS = {
    "floor": "cpp_baseline",  # raw NoC write, no flow control, no DFB staging
    "synced": "cpp_baseline_synced",  # + per-transfer credit handshake
    "dfb": "cpp_baseline_dfb",  # + per-transfer DFB reserve/push/wait/pop
    "optimized": "cpp_optimized",  # batched stateful NoC writes, bulk (ceiling)
    "ring": "cpp_bounded_ring",  # bounded ring + stateful NoC + batched barrier/credits
}
# The optimized (bulk) variant holds all N tiles in L1, so cap N to fit L1.
# The duration-vs-N fit stays linear well within this range.
OPTIMIZED_MAX_N = 128
# The bounded-ring variant uses a fixed-size ring (no L1 cap).
DEFAULT_RING_DEPTH = 8
SENDER = (0, 0)
RECEIVER = (1, 0)
DONE_SEM_ID = 0
DATA_SEM_ID = 0  # synced/dfb: receiver-owned, sender signals data ready
FREE_SEM_ID = 1  # synced/dfb: sender-owned, receiver returns slot credits

KERNELS = Path("benchmarks/microbench/pipes/kernels")
SENDER_KERNEL = str(KERNELS / "pipe_sender.cpp")
RECEIVER_KERNEL = str(KERNELS / "pipe_receiver.cpp")
SYNCED_SENDER_KERNEL = str(KERNELS / "pipe_sender_synced.cpp")
SYNCED_RECEIVER_KERNEL = str(KERNELS / "pipe_receiver_synced.cpp")
DFB_SENDER_KERNEL = str(KERNELS / "pipe_sender_dfb.cpp")
OPTIMIZED_SENDER_KERNEL = str(KERNELS / "pipe_sender_optimized.cpp")
RING_SENDER_KERNEL = str(KERNELS / "bounded_ring_sender.cpp")
RING_RECEIVER_KERNEL = str(KERNELS / "bounded_ring_receiver.cpp")

COLUMNS = [
    "n",
    "dtype",
    "variant",
    "arch",
    "freq_mhz",
    "sender_brisc_us",
    "sender_ncrisc_us",
    "sender_dm_max_us",
    "recv_dm_max_us",
    "wall_ms_per_run",
    "wall_us_per_transfer",
    "bitexact",
    "mismatch_tiles",
]


def _core_set(x, y):
    coord = ttnn.CoreCoord(x, y)
    return ttnn.CoreRangeSet([ttnn.CoreRange(coord, coord)])


def _two_core_set():
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(*SENDER), ttnn.CoreCoord(*RECEIVER))]
    )


def make_tensors(device, dtype, seed, n):
    """N distinct source tiles and an N-tile destination; ``(inp, out, src)``."""
    ttnn_dtype, torch_dtype, _ = harness.DTYPES[dtype]
    torch.manual_seed(seed)
    src = torch.randn(TILE, n * TILE, dtype=torch_dtype)
    inp = ttnn.from_torch(
        src,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.zeros(TILE, n * TILE, dtype=torch_dtype),
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return inp, out, src


def build_program(device, inp, out, n, dtype, mode, ring_depth=DEFAULT_RING_DEPTH):
    ttnn_dtype, _, dtype_bytes = harness.DTYPES[dtype]
    page_size = dtype_bytes * TILE * TILE
    sender_cores = _core_set(*SENDER)
    recv_cores = _core_set(*RECEIVER)
    recv_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(*RECEIVER))
    send_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(*SENDER))

    # c_1 stages the source on the sender. The dfb mode cycles two slots,
    # optimized holds all N, ring uses a bounded ring, and the remaining modes
    # use one scratch slot. c_0 is allocated on both cores so the sender can
    # address the receiver's L1 with get_write_ptr(c_0).
    cb_src_blocks = {"dfb": 2, "optimized": n, "ring": ring_depth}.get(mode, 1)
    cb_src = harness.dfb(1, ttnn_dtype, page_size, sender_cores, cb_src_blocks)

    if mode in ("floor", "optimized"):
        # No flow control: one destination slot per tile so nothing is
        # overwritten before the receiver drains all N.
        cb_dst = harness.dfb(0, ttnn_dtype, page_size, _two_core_set(), n)
        semaphores = [
            ttnn.SemaphoreDescriptor(
                id=DONE_SEM_ID, core_ranges=recv_cores, initial_value=0
            )
        ]
        sender_rt = [
            (
                ttnn.CoreCoord(*SENDER),
                [inp.buffer_address(), recv_phys.x, recv_phys.y, n, DONE_SEM_ID],
            )
        ]
        recv_rt = [(ttnn.CoreCoord(*RECEIVER), [out.buffer_address(), DONE_SEM_ID, n])]
        sender_kernel = (
            OPTIMIZED_SENDER_KERNEL if mode == "optimized" else SENDER_KERNEL
        )
        recv_kernel = RECEIVER_KERNEL
    elif mode == "ring":
        # Two rings of ring_depth slots with the data/free credit handshake,
        # stateful NoC writes, and the barrier/credit batched per ring chunk. The
        # second ring lets the sender write the next chunk while the receiver
        # drains the current one, so the two cores overlap.
        cb_dst = harness.dfb(0, ttnn_dtype, page_size, _two_core_set(), 2 * ring_depth)
        semaphores = [
            ttnn.SemaphoreDescriptor(
                id=DATA_SEM_ID, core_ranges=recv_cores, initial_value=0
            ),
            ttnn.SemaphoreDescriptor(
                id=FREE_SEM_ID, core_ranges=sender_cores, initial_value=0
            ),
        ]
        sender_rt = [
            (
                ttnn.CoreCoord(*SENDER),
                [
                    inp.buffer_address(),
                    recv_phys.x,
                    recv_phys.y,
                    n,
                    DATA_SEM_ID,
                    FREE_SEM_ID,
                    ring_depth,
                ],
            )
        ]
        recv_rt = [
            (
                ttnn.CoreCoord(*RECEIVER),
                [
                    out.buffer_address(),
                    send_phys.x,
                    send_phys.y,
                    n,
                    DATA_SEM_ID,
                    FREE_SEM_ID,
                    ring_depth,
                ],
            )
        ]
        sender_kernel = RING_SENDER_KERNEL
        recv_kernel = RING_RECEIVER_KERNEL
    else:
        # synced and dfb share the double-buffered credit handshake and receiver;
        # dfb adds the per-transfer DFB cycle on the sender. Each semaphore has
        # one incrementer, so the kernels' wait_min is race-free.
        cb_dst = harness.dfb(0, ttnn_dtype, page_size, _two_core_set(), 2)
        semaphores = [
            ttnn.SemaphoreDescriptor(
                id=DATA_SEM_ID, core_ranges=recv_cores, initial_value=0
            ),
            ttnn.SemaphoreDescriptor(
                id=FREE_SEM_ID, core_ranges=sender_cores, initial_value=0
            ),
        ]
        sender_rt = [
            (
                ttnn.CoreCoord(*SENDER),
                [
                    inp.buffer_address(),
                    recv_phys.x,
                    recv_phys.y,
                    n,
                    DATA_SEM_ID,
                    FREE_SEM_ID,
                ],
            )
        ]
        recv_rt = [
            (
                ttnn.CoreCoord(*RECEIVER),
                [
                    out.buffer_address(),
                    send_phys.x,
                    send_phys.y,
                    n,
                    DATA_SEM_ID,
                    FREE_SEM_ID,
                ],
            )
        ]
        sender_kernel = DFB_SENDER_KERNEL if mode == "dfb" else SYNCED_SENDER_KERNEL
        recv_kernel = SYNCED_RECEIVER_KERNEL

    kernels = [
        harness.file_kernel(
            sender_kernel,
            sender_cores,
            harness.accessor_args(inp),
            sender_rt,
            ttnn.WriterConfigDescriptor(),
        ),
        harness.file_kernel(
            recv_kernel,
            recv_cores,
            harness.accessor_args(out),
            recv_rt,
            ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(
        kernels=kernels, semaphores=semaphores, cbs=[cb_src, cb_dst]
    )


def run_n(device, n, dtype, args, mode, ring_depth):
    inp, out, src = make_tensors(device, dtype, args.seed, n)
    program = build_program(device, inp, out, n, dtype, mode, ring_depth)
    io_tensors = [inp, out]
    captured = {}

    def run_program(*_):
        captured["out"] = ttnn.generic_op(io_tensors, program)

    try:
        per_core, arch, freq_mhz, wall_s = ttl_harness.run_operation(
            device, run_program, (), warmup=args.warmup, runs=args.runs, wall=args.wall
        )
        got = ttnn.to_torch(captured["out"]).float()
    finally:
        ttnn.deallocate(inp)
        ttnn.deallocate(out)

    bitexact = torch.equal(src.float(), got)
    sender_key = ttl_harness.physical_core(device, SENDER)
    recv_key = ttl_harness.physical_core(device, RECEIVER)
    sender = per_core.get(sender_key, {})
    return {
        "n": n,
        "dtype": dtype,
        "variant": VARIANTS[mode] + (f"_r{ring_depth}" if mode == "ring" else ""),
        "arch": arch,
        "freq_mhz": freq_mhz,
        "sender_brisc_us": sender.get("BRISC"),
        "sender_ncrisc_us": sender.get("NCRISC"),
        "sender_dm_max_us": ttl_harness.dm_max_us(per_core, sender_key),
        "recv_dm_max_us": ttl_harness.dm_max_us(per_core, recv_key),
        "wall_ms_per_run": None if wall_s is None else wall_s * 1e3 / args.runs,
        "wall_us_per_transfer": (
            None if wall_s is None else wall_s * 1e6 / (args.runs * n)
        ),
        "bitexact": int(bitexact),
        "mismatch_tiles": 0 if bitexact else 1,
    }


def main():
    parser = create_benchmark_arg_parser(
        "C++ unicast baselines for PipeNet comparison", default_csv=DEFAULT_CSV
    )
    parser.add_argument(
        "--transfers",
        default="1,2,4,8,16,32,64,128,256",
        help="comma list of transfer counts N to sweep",
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument(
        "--wall", action="store_true", help="also record host wall time"
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "floor", "synced", "dfb", "optimized", "ring"],
        help="floor = raw NoC write; synced = + credit handshake; "
        "dfb = + per-transfer DFB cycle; "
        "optimized = batched stateful NoC writes, bulk (ceiling); "
        "ring = bounded producer/consumer ring with batched barriers",
    )
    parser.add_argument(
        "--ring-depths",
        default=str(DEFAULT_RING_DEPTH),
        help="comma list of depths for the bounded-ring variant. Each depth is "
        "written as variant cpp_bounded_ring_r<ring>.",
    )
    args = parser.parse_args()
    if args.compile_only:
        print("compile-only: nothing to execute without a device.")
        return

    modes = {
        "all": ("floor", "synced", "dfb", "optimized", "ring"),
        "floor": ("floor",),
        "synced": ("synced",),
        "dfb": ("dfb",),
        "optimized": ("optimized",),
        "ring": ("ring",),
    }[args.mode]
    transfer_counts = [int(part) for part in str(args.transfers).split(",")]
    ring_depths = [int(part) for part in str(args.ring_depths).split(",")]
    device = ttnn.open_device(device_id=args.device_id)
    rows = []
    try:
        for n in transfer_counts:
            for mode in modes:
                if mode == "optimized" and n > OPTIMIZED_MAX_N:
                    continue
                rings = ring_depths if mode == "ring" else [DEFAULT_RING_DEPTH]
                for ring_depth in rings:
                    row = run_n(device, n, args.dtype, args, mode, ring_depth)
                    rows.append(row)
                    flag = "" if row["bitexact"] else "  MISMATCH"
                    print(
                        f"n={n} {row['variant']} "
                        f"sender_dm_max={row['sender_dm_max_us']} µs{flag}",
                        flush=True,
                    )
    finally:
        ttnn.close_device(device)

    if not args.no_csv and rows:
        arch = rows[0].get("arch", "dev")
        out_csv = harness.write_csv(rows, args.csv, COLUMNS, arch, args.dtype)
        print(f"wrote {len(rows)} rows to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
