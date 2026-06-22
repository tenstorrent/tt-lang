# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark PipeNet computed-address and receiver-published protocols."""

from __future__ import annotations

import argparse
import faulthandler
import os
import statistics
import sys
import time
from dataclasses import dataclass

import torch
import ttl

try:
    import ttnn
except ImportError as exc:
    raise SystemExit("ttnn is required for this benchmark") from exc

TILE = 32
DEFAULT_TRANSFERS = 64
DEFAULT_WARMUPS = 3
DEFAULT_ITERATIONS = 10
DEFAULT_STACK_TIMEOUT = 0


def _create_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--transfers", type=int, default=DEFAULT_TRANSFERS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--stack-timeout",
        type=int,
        default=DEFAULT_STACK_TIMEOUT,
        help="dump Python stacks after this many seconds; 0 disables stack dumps",
    )
    return parser


def _parse_static_args() -> argparse.Namespace:
    return _create_arg_parser(add_help=False).parse_known_args()[0]


STATIC_ARGS = _parse_static_args()
N_TRANSFERS = max(1, STATIC_ARGS.transfers)


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    first_ms: float
    mean_ms: float
    stdev_ms: float
    iterations: int

    @property
    def per_transfer_us(self) -> float:
        return self.mean_ms * 1000.0 / N_TRANSFERS


def _log(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@ttl.operation(grid=(2, 1))
def computed_unicast(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for _transfer_idx in range(N_TRANSFERS):

            def send(pipe):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()

            net.if_src(send)

            def recv(pipe):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()
                with recv_dfb.wait() as recv_blk:
                    ttl.copy(recv_blk, out[0, 0]).wait()

            net.if_dst(recv)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1), options="--no-ttl-pipe-computed-addresses")
def receiver_published_unicast(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for _transfer_idx in range(N_TRANSFERS):

            def send(pipe_arg):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe_arg).wait()

            net.if_src(send)

            def recv(pipe_arg):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe_arg, recv_blk).wait()
                with recv_dfb.wait() as recv_blk:
                    ttl.copy(recv_blk, out[0, 0]).wait()

            net.if_dst(recv)

    @ttl.datamovement()
    def dm_brisc():
        pass


def _time_call(device, label, operation, inp, out, warmups, iterations):
    _log(f"{label}: first compile+execute begin")
    first_start = time.perf_counter()
    operation(inp, out)
    _log(f"{label}: first call returned; synchronizing device")
    ttnn.synchronize_device(device)
    first_ms = (time.perf_counter() - first_start) * 1000.0
    _log(f"{label}: first compile+execute complete in {first_ms:.3f} ms")

    for _warmup_idx in range(warmups):
        _log(f"{label}: warmup {_warmup_idx + 1}/{warmups} begin")
        operation(inp, out)
        _log(f"{label}: warmup {_warmup_idx + 1}/{warmups} returned; synchronizing")
        ttnn.synchronize_device(device)
        _log(f"{label}: warmup {_warmup_idx + 1}/{warmups} complete")

    samples_ms = []
    for _iter_idx in range(iterations):
        _log(f"{label}: iteration {_iter_idx + 1}/{iterations} begin")
        start = time.perf_counter()
        operation(inp, out)
        _log(f"{label}: iteration {_iter_idx + 1}/{iterations} returned; synchronizing")
        ttnn.synchronize_device(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        _log(
            f"{label}: iteration {_iter_idx + 1}/{iterations} complete "
            f"in {samples_ms[-1]:.3f} ms"
        )

    mean_ms = statistics.fmean(samples_ms)
    stdev_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    print(
        f"{label}: first_compile_execute_ms={first_ms:.3f} "
        f"execute_mean_ms={mean_ms:.3f} execute_stdev_ms={stdev_ms:.3f} "
        f"per_transfer_us={mean_ms * 1000.0 / N_TRANSFERS:.3f} "
        f"iterations={iterations}"
    )
    return BenchmarkResult(label, first_ms, mean_ms, stdev_ms, iterations)


def _print_comparison(computed_result, receiver_published_result):
    delta_ms = receiver_published_result.mean_ms - computed_result.mean_ms
    delta_us = (
        receiver_published_result.per_transfer_us - computed_result.per_transfer_us
    )
    ratio = receiver_published_result.mean_ms / computed_result.mean_ms
    time_reduction_percent = 100.0 * delta_ms / receiver_published_result.mean_ms
    speedup_percent = 100.0 * (ratio - 1.0)
    faster_label = "computed_unicast" if delta_ms > 0 else "receiver_published_unicast"

    print(
        "comparison: "
        f"computed_mean_ms={computed_result.mean_ms:.3f} "
        f"receiver_published_mean_ms={receiver_published_result.mean_ms:.3f} "
        f"delta_ms={delta_ms:.3f} "
        f"delta_per_transfer_us={delta_us:.3f} "
        f"computed_per_transfer_us={computed_result.per_transfer_us:.3f} "
        f"receiver_published_per_transfer_us="
        f"{receiver_published_result.per_transfer_us:.3f} "
        f"receiver_published_over_computed={ratio:.4f} "
        f"computed_time_reduction_percent={time_reduction_percent:.2f} "
        f"computed_speedup_percent={speedup_percent:.2f} "
        f"faster={faster_label}"
    )


def _make_tensor(shape, device):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    args = _create_arg_parser().parse_args()

    if args.transfers != N_TRANSFERS:
        raise SystemExit(
            "--transfers must be provided before script import; run the script "
            f"again with --transfers={args.transfers}"
        )

    if args.stack_timeout > 0:
        faulthandler.dump_traceback_later(
            args.stack_timeout, repeat=True, file=sys.stderr
        )

    os.environ.pop("TTLANG_COMPILE_ONLY", None)
    _log(
        "benchmark begin "
        f"transfers={args.transfers} warmups={args.warmups} "
        f"iterations={args.iterations}"
    )
    _log("opening device")
    device = ttnn.open_device(device_id=0)
    try:
        _log("allocating tensors")
        inp = _make_tensor((TILE, TILE), device)
        computed_out = _make_tensor((TILE, TILE), device)
        receiver_published_out = _make_tensor((TILE, TILE), device)
        _log("tensors allocated")
        computed_result = _time_call(
            device,
            "computed_unicast",
            computed_unicast,
            inp,
            computed_out,
            args.warmups,
            args.iterations,
        )
        receiver_published_result = _time_call(
            device,
            "receiver_published_unicast",
            receiver_published_unicast,
            inp,
            receiver_published_out,
            args.warmups,
            args.iterations,
        )
        _print_comparison(computed_result, receiver_published_result)
    finally:
        _log("closing device")
        ttnn.close_device(device)
        if args.stack_timeout > 0:
            faulthandler.cancel_dump_traceback_later()
        _log("benchmark complete")


if __name__ == "__main__":
    main()
