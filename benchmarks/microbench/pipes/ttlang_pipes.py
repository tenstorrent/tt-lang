# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tt-lang PipeNet address-protocol microbenchmark.

Measures the per-transfer NoC cost of a PipeNet unicast (0,0) -> (1,0) under the
two receiver-address protocols:

  ttlang_computed   : the sender computes the destination dataflow-buffer
                       address at compile time (default).
  ttlang_published  : the receiver publishes its address at run time and the
                       sender waits for it (``--no-ttl-pipe-computed-addresses``).

Each variant sends N single-tile transfers in a compile-time loop; sweeping N
and regressing the sender data-movement kernel duration (from the Tracy device
profiler) separates the per-transfer NoC cost from the protocol's one-time
setup. ``compare.py`` does the regression across the rows written here.
Correctness is bit-exact: a PipeNet transfer copies bytes, so the output tile
must equal the source tile exactly.

    python -m benchmarks.microbench.pipes.ttlang_pipes --transfers 1,2,4,8,16,32,64,128,256
    python -m benchmarks.microbench.pipes.ttlang_pipes --transfers 1,64,256 --wall
"""

import os
import tempfile

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

import torch
import ttl
import ttnn

from benchmarks.common import create_benchmark_arg_parser
from benchmarks.microbench import harness
from benchmarks.microbench.harness import TILE
from benchmarks.microbench.ttlang import ttl_harness

DEFAULT_CSV = "benchmarks/microbench/results/pipes_ttlang.csv"
SENDER = (0, 0)
RECEIVER = (1, 0)

# variant name -> compiler options passed to @ttl.operation (None = defaults).
VARIANTS = {
    "ttlang_computed": None,
    "ttlang_published": "--no-ttl-pipe-computed-addresses",
}

COLUMNS = [
    "n",
    "dtype",
    "send_block_count",
    "recv_block_count",
    "variant",
    "uses_computed_address",
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


def begin_protocol_capture(enabled):
    if not enabled:
        return None, None
    final_mlir = tempfile.NamedTemporaryFile(
        prefix="ttlang_pipe_protocol_", suffix=".mlir", delete=False
    )
    final_mlir_path = final_mlir.name
    final_mlir.close()
    previous_path = os.environ.get("TTLANG_FINAL_MLIR")
    os.environ["TTLANG_FINAL_MLIR"] = final_mlir_path
    return previous_path, final_mlir_path


def end_protocol_capture(previous_path):
    if previous_path is None:
        os.environ.pop("TTLANG_FINAL_MLIR", None)
    else:
        os.environ["TTLANG_FINAL_MLIR"] = previous_path


def verify_address_protocol(variant, final_mlir_path):
    with open(final_mlir_path) as file:
        final_mlir = file.read()
    uses_computed = "ttl.pipe_computed_address_dfb_indices" in final_mlir
    expected_computed = variant == "ttlang_computed"
    if uses_computed != expected_computed:
        expected = "computed" if expected_computed else "receiver-published"
        actual = "computed" if uses_computed else "receiver-published"
        raise RuntimeError(
            f"{variant} expected {expected} addressing, "
            f"but final MLIR selected {actual} addressing"
        )
    return int(uses_computed)


def build_op(n_transfers, options, send_block_count=2, recv_block_count=1):
    """Build a fresh ``@ttl.operation`` that sends ``n_transfers`` single tiles.

    A new operation per N gives each N its own compile; the source read is
    identical across variants, so the computed-vs-published difference is in the
    address protocol alone.

    ``send_block_count`` is the source staging DFB depth. In the current
    lowering, increasing this depth does not make the sender run ahead of the
    receiver because each single-tile transfer waits for completion before its
    staging slot is reused.

    ``recv_block_count`` defaults to one because this benchmark pops each
    receive before posting the next one. With a larger receiver DFB ring, the
    same post op would advance through multiple physical slots while computed
    addressing supplies one static slot address. The compiler must then select
    the receiver-published fallback, which is not the intended comparison.

    Each transfer stages the tile through ``send_dfb`` with two reserve/wait
    contexts (reserve+fill, then wait+send). Filling and sending from a single
    reserved block compiles but deadlocks for N > ``send_block_count`` (the
    staging block is never freed without the push/pop cycle), so the cycle is
    required here.
    """
    op_kwargs = {"grid": (2, 1)}
    if options:
        op_kwargs["options"] = options

    @ttl.operation(**op_kwargs)
    def pipe_unicast(inp, out):
        net = ttl.PipeNet([ttl.Pipe(src=SENDER, dst=RECEIVER)])
        send_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=send_block_count
        )
        recv_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(1, 1), block_count=recv_block_count
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm():
            for t in range(n_transfers):

                def send(pipe):
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, t], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                def recv(pipe):
                    with recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()
                    with recv_dfb.wait() as recv_blk:
                        ttl.copy(recv_blk, out[0, t]).wait()

                net.if_src(send)
                net.if_dst(recv)

        @ttl.datamovement()
        def dm_second():
            pass

    return pipe_unicast


def make_tensors(device, dtype, seed, n):
    """N distinct source tiles and an N-tile destination; ``(inp, out, src)``.

    Distinct tiles per transfer mean the bit-exact check validates that every
    transfer landed in the right place, not just final-tile delivery.
    """
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


def run_variant(device, variant, options, n, dtype, args):
    op = build_op(n, options, args.send_block_count, args.recv_block_count)
    inp, out, src = make_tensors(device, dtype, args.seed, n)
    previous_final_mlir, final_mlir_path = begin_protocol_capture(
        not args.no_verify_protocol
    )
    try:
        try:
            per_core, arch, freq_mhz, wall_s = ttl_harness.run_operation(
                device,
                op,
                (inp, out),
                warmup=args.warmup,
                runs=args.runs,
                wall=args.wall,
            )
        finally:
            if final_mlir_path is not None:
                end_protocol_capture(previous_final_mlir)
        got = ttnn.to_torch(out).float()
    finally:
        ttnn.deallocate(inp)
        ttnn.deallocate(out)
    if final_mlir_path is None:
        uses_computed_address = None
    else:
        try:
            uses_computed_address = verify_address_protocol(variant, final_mlir_path)
        finally:
            if os.path.exists(final_mlir_path):
                os.unlink(final_mlir_path)

    ref = src.float()
    bitexact = torch.equal(ref, got)
    sender_key = ttl_harness.physical_core(device, SENDER)
    recv_key = ttl_harness.physical_core(device, RECEIVER)
    sender = per_core.get(sender_key, {})
    row = {
        "n": n,
        "dtype": dtype,
        "send_block_count": args.send_block_count,
        "recv_block_count": args.recv_block_count,
        "variant": variant,
        "uses_computed_address": uses_computed_address,
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
    return row


def main():
    parser = create_benchmark_arg_parser(
        "tt-lang PipeNet address-protocol benchmark", default_csv=DEFAULT_CSV
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
        "--send-block-count",
        type=int,
        default=2,
        help="dataflow-buffer depth for the source staging DFB",
    )
    parser.add_argument(
        "--recv-block-count",
        type=int,
        default=1,
        help="dataflow-buffer depth for the receiver DFB",
    )
    parser.add_argument(
        "--no-verify-protocol",
        action="store_true",
        help="do not check final MLIR for the selected address protocol",
    )
    args = parser.parse_args()
    if args.compile_only:
        print("compile-only: nothing to execute without a device.")
        return

    transfer_counts = [int(part) for part in str(args.transfers).split(",")]
    device = ttnn.open_device(device_id=args.device_id)
    rows = []
    try:
        for n in transfer_counts:
            for variant, options in VARIANTS.items():
                row = run_variant(device, variant, options, n, args.dtype, args)
                rows.append(row)
                flag = "" if row["bitexact"] else "  MISMATCH"
                print(
                    f"n={n} {variant} sender={row['sender_dm_max_us']} "
                    f"recv={row['recv_dm_max_us']} µs{flag}",
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
