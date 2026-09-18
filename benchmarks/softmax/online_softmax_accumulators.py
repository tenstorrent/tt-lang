# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-core online softmax benchmark for dependent accumulators."""

from pathlib import Path
from typing import NamedTuple

import torch
import ttnn
import ttl

from benchmarks.common import (
    assert_pcc,
    create_benchmark_arg_parser,
    measure_pcc,
    time_runs,
    to_device,
    write_csv,
)

TILE = ttnn.TILE_SIZE
PNHt = 1
SCORE_CHUNK_TILES = 1
VALUE_TILES = 1
N_CHUNKS = 2
Q_ROWS = PNHt * TILE
SEQ = SCORE_CHUNK_TILES * N_CHUNKS * TILE
VALUE_DIM = VALUE_TILES * TILE
PCC_THRESHOLD = 0.99
OUTPUT_CSV = Path("/tmp/ttlang_online_softmax_accumulators.csv")
DESCRIPTION = (
    "Benchmark a single-core online softmax update with dependent m/l/o "
    "accumulators."
)
SSA_VARIANT = "ssa"
STAGED_VARIANT = "staged"

FIELDS = (
    "label",
    "variant",
    "seq",
    "q_rows",
    "value_dim",
    "score_chunk_tiles",
    "n_chunks",
    "warmup",
    "runs",
    "ttlang_ms",
    "pcc",
    "max_abs",
    "mean_abs",
)


class BenchmarkVariant(NamedTuple):
    operation: object
    label: str
    name: str


@ttl.operation(grid=(1, 1), fp32_dest_acc_en=False)
def online_softmax_ssa(scores, values, final_out):
    score_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, SCORE_CHUNK_TILES), block_count=2
    )
    value_dfb = ttl.make_dataflow_buffer_like(
        values, shape=(SCORE_CHUNK_TILES, VALUE_TILES), block_count=2
    )
    score_replay_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, SCORE_CHUNK_TILES), block_count=2
    )
    chunk_max_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, 1), block_count=2
    )
    chunk_sum_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, 1), block_count=2
    )

    m_state_dfb = ttl.make_dataflow_buffer_like(scores, shape=(PNHt, 1), block_count=2)
    l_state_dfb = ttl.make_dataflow_buffer_like(scores, shape=(PNHt, 1), block_count=2)
    o_state_dfb = ttl.make_dataflow_buffer_like(
        values, shape=(PNHt, VALUE_TILES), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        final_out, shape=(PNHt, VALUE_TILES), block_count=2
    )

    @ttl.compute()
    def compute():
        m_initial = m_state_dfb.reserve()
        m_initial.store(ttl.block.fill(-1e30, shape=(PNHt, 1)))
        l_initial = l_state_dfb.reserve()
        l_initial.store(ttl.block.fill(0, shape=(PNHt, 1)))
        o_initial = o_state_dfb.reserve()
        o_initial.store(ttl.block.fill(0, shape=(PNHt, VALUE_TILES)))

        for _ in range(N_CHUNKS):
            score_block = score_dfb.wait()
            value_block = value_dfb.wait()

            chunk_max_block = chunk_max_dfb.reserve()
            chunk_max_block.store(ttl.math.reduce_max(score_block, dims=[1]))
            score_replay = score_replay_dfb.reserve()
            score_replay.store(score_block)

            m_old = m_state_dfb.wait()
            chunk_max = chunk_max_dfb.wait()
            m_new = ttl.math.max(m_old, chunk_max)
            alpha = ttl.exp(ttl.sub(m_old, m_new))
            m_next = m_state_dfb.reserve()
            m_next.store(m_new)

            m_bcast = ttl.block.broadcast(
                m_new, dims=[1], shape=(PNHt, SCORE_CHUNK_TILES)
            )
            score_reloaded = score_replay_dfb.wait()
            exp_scores = ttl.exp(ttl.sub(score_reloaded, m_bcast))
            chunk_sum_block = chunk_sum_dfb.reserve()
            chunk_sum_block.store(ttl.math.reduce_sum(exp_scores, dims=[1]))

            l_old = l_state_dfb.wait()
            chunk_sum = chunk_sum_dfb.wait()
            l_next = l_state_dfb.reserve()
            l_next.store(alpha * l_old + chunk_sum)

            o_old = o_state_dfb.wait()
            alpha_bcast = ttl.block.broadcast(
                alpha, dims=[1], shape=(PNHt, VALUE_TILES)
            )
            o_corr = alpha_bcast * o_old
            partial_value = exp_scores @ value_block
            o_next = o_state_dfb.reserve()
            o_next.store(o_corr + partial_value)

        l_final = l_state_dfb.wait()
        o_final = o_state_dfb.wait()
        l_bcast = ttl.block.broadcast(l_final, dims=[1], shape=(PNHt, VALUE_TILES))
        out_block = out_dfb.reserve()
        out_block.store(o_final * ttl.math.recip(l_bcast))

    @ttl.datamovement()
    def reader():
        for chunk_index in range(N_CHUNKS):
            chunk_start = chunk_index * SCORE_CHUNK_TILES
            with score_dfb.reserve() as score_block:
                ttl.copy(
                    scores[0:PNHt, chunk_start : chunk_start + SCORE_CHUNK_TILES],
                    score_block,
                ).wait()
            with value_dfb.reserve() as value_block:
                ttl.copy(
                    values[
                        chunk_start : chunk_start + SCORE_CHUNK_TILES, 0:VALUE_TILES
                    ],
                    value_block,
                ).wait()

    @ttl.datamovement()
    def writer():
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, final_out[0:PNHt, 0:VALUE_TILES]).wait()


@ttl.operation(grid=(1, 1), fp32_dest_acc_en=False)
def online_softmax_staged(scores, values, final_out):
    score_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, SCORE_CHUNK_TILES), block_count=2
    )
    value_dfb = ttl.make_dataflow_buffer_like(
        values, shape=(SCORE_CHUNK_TILES, VALUE_TILES), block_count=2
    )
    score_replay_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, SCORE_CHUNK_TILES), block_count=2
    )
    exp_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, SCORE_CHUNK_TILES), block_count=2
    )
    chunk_max_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, 1), block_count=2
    )
    chunk_sum_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, 1), block_count=2
    )
    m_new_dfb = ttl.make_dataflow_buffer_like(scores, shape=(PNHt, 1), block_count=2)
    alpha_dfb = ttl.make_dataflow_buffer_like(scores, shape=(PNHt, 1), block_count=2)
    m_bcast_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(PNHt, SCORE_CHUNK_TILES), block_count=2
    )
    alpha_bcast_dfb = ttl.make_dataflow_buffer_like(
        values, shape=(PNHt, VALUE_TILES), block_count=2
    )
    o_corr_dfb = ttl.make_dataflow_buffer_like(
        values, shape=(PNHt, VALUE_TILES), block_count=2
    )
    partial_value_dfb = ttl.make_dataflow_buffer_like(
        values, shape=(PNHt, VALUE_TILES), block_count=2
    )
    l_bcast_dfb = ttl.make_dataflow_buffer_like(
        values, shape=(PNHt, VALUE_TILES), block_count=2
    )

    m_state_dfb = ttl.make_dataflow_buffer_like(scores, shape=(PNHt, 1), block_count=2)
    l_state_dfb = ttl.make_dataflow_buffer_like(scores, shape=(PNHt, 1), block_count=2)
    o_state_dfb = ttl.make_dataflow_buffer_like(
        values, shape=(PNHt, VALUE_TILES), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        final_out, shape=(PNHt, VALUE_TILES), block_count=2
    )

    @ttl.compute()
    def compute():
        with m_state_dfb.reserve() as m_initial:
            m_initial.store(ttl.block.fill(-1e30, shape=(PNHt, 1)))
        with l_state_dfb.reserve() as l_initial:
            l_initial.store(ttl.block.fill(0, shape=(PNHt, 1)))
        with o_state_dfb.reserve() as o_initial:
            o_initial.store(ttl.block.fill(0, shape=(PNHt, VALUE_TILES)))

        for _ in range(N_CHUNKS):
            with score_dfb.wait() as score_block:
                with chunk_max_dfb.reserve() as chunk_max_block:
                    chunk_max_block.store(ttl.math.reduce_max(score_block, dims=[1]))
                with score_replay_dfb.reserve() as score_replay:
                    score_replay.store(score_block)

            with m_state_dfb.wait() as m_old:
                with chunk_max_dfb.wait() as chunk_max:
                    with m_new_dfb.reserve() as m_new_block:
                        m_new_block.store(ttl.math.max(m_old, chunk_max))
                with m_new_dfb.wait() as m_new:
                    with alpha_dfb.reserve() as alpha:
                        alpha.store(ttl.exp(ttl.sub(m_old, m_new)))
                    with m_bcast_dfb.reserve() as m_bcast:
                        m_bcast.store(
                            ttl.block.broadcast(
                                m_new, dims=[1], shape=(PNHt, SCORE_CHUNK_TILES)
                            )
                        )
                    with m_state_dfb.reserve() as m_next:
                        m_next.store(m_new)

            with (
                score_replay_dfb.wait() as score_reloaded,
                m_bcast_dfb.wait() as m_bcast,
                exp_dfb.reserve() as exp_block,
            ):
                exp_block.store(ttl.exp(ttl.sub(score_reloaded, m_bcast)))

            with exp_dfb.wait() as exp_scores:
                with chunk_sum_dfb.reserve() as chunk_sum:
                    chunk_sum.store(ttl.math.reduce_sum(exp_scores, dims=[1]))
                with exp_dfb.reserve() as exp_replay:
                    exp_replay.store(exp_scores)

            with (
                alpha_dfb.wait() as alpha,
                l_state_dfb.wait() as l_old,
                chunk_sum_dfb.wait() as chunk_sum,
            ):
                with l_state_dfb.reserve() as l_next:
                    l_next.store(alpha * l_old + chunk_sum)
                with alpha_bcast_dfb.reserve() as alpha_bcast:
                    alpha_bcast.store(
                        ttl.block.broadcast(alpha, dims=[1], shape=(PNHt, VALUE_TILES))
                    )

            with (
                alpha_bcast_dfb.wait() as alpha_bcast,
                o_state_dfb.wait() as o_old,
                o_corr_dfb.reserve() as o_corr,
            ):
                o_corr.store(alpha_bcast * o_old)

            with (
                exp_dfb.wait() as exp_scores,
                value_dfb.wait() as value_block,
                partial_value_dfb.reserve() as partial_value,
            ):
                partial_value.store(exp_scores @ value_block)

            with (
                o_corr_dfb.wait() as o_corr,
                partial_value_dfb.wait() as partial_value,
                o_state_dfb.reserve() as o_next,
            ):
                o_next.store(o_corr + partial_value)

        with l_state_dfb.wait() as l_final:
            with l_bcast_dfb.reserve() as l_bcast:
                l_bcast.store(
                    ttl.block.broadcast(l_final, dims=[1], shape=(PNHt, VALUE_TILES))
                )
        with (
            o_state_dfb.wait() as o_final,
            l_bcast_dfb.wait() as l_bcast,
            out_dfb.reserve() as out_block,
        ):
            out_block.store(o_final * ttl.math.recip(l_bcast))

    @ttl.datamovement()
    def reader():
        for chunk_index in range(N_CHUNKS):
            chunk_start = chunk_index * SCORE_CHUNK_TILES
            with score_dfb.reserve() as score_block:
                ttl.copy(
                    scores[0:PNHt, chunk_start : chunk_start + SCORE_CHUNK_TILES],
                    score_block,
                ).wait()
            with value_dfb.reserve() as value_block:
                ttl.copy(
                    values[
                        chunk_start : chunk_start + SCORE_CHUNK_TILES, 0:VALUE_TILES
                    ],
                    value_block,
                ).wait()

    @ttl.datamovement()
    def writer():
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, final_out[0:PNHt, 0:VALUE_TILES]).wait()


def _get_variant(name):
    if name == SSA_VARIANT:
        return BenchmarkVariant(
            online_softmax_ssa,
            "online_softmax_accumulators_ssa",
            SSA_VARIANT,
        )
    if name == STAGED_VARIANT:
        return BenchmarkVariant(
            online_softmax_staged,
            "online_softmax_accumulators_staged",
            STAGED_VARIANT,
        )
    raise ValueError(f"Unsupported benchmark variant: {name}")


def _error_stats(golden, actual):
    abs_diff = (actual.float() - golden.float()).abs()
    return abs_diff.max().item(), abs_diff.mean().item()


def _make_inputs(seed):
    torch.manual_seed(seed)
    scores = torch.randn(Q_ROWS, SEQ, dtype=torch.bfloat16) * 0.1
    values = torch.randn(SEQ, VALUE_DIM, dtype=torch.bfloat16) * 0.1
    expected = torch.softmax(scores.float(), dim=1) @ values.float()
    return scores, values, expected.to(torch.bfloat16)


def main():
    parser = create_benchmark_arg_parser(DESCRIPTION, default_csv=OUTPUT_CSV)
    parser.add_argument(
        "--variant",
        choices=(SSA_VARIANT, STAGED_VARIANT),
        default=SSA_VARIANT,
        help="Benchmark variant to run.",
    )
    args = parser.parse_args()
    variant = _get_variant(args.variant)

    device = ttnn.open_device(device_id=args.device_id)
    try:
        scores_torch, values_torch, expected = _make_inputs(args.seed)
        scores_dram = to_device(device, scores_torch)
        values_dram = to_device(device, values_torch)
        final_dram = to_device(
            device,
            torch.zeros(Q_ROWS, VALUE_DIM, dtype=torch.bfloat16),
        )

        def run_once():
            variant.operation(scores_dram, values_dram, final_dram)

        if args.compile_only:
            print(f"Running {variant.label} compile-only...")
            run_once()
            ttnn.synchronize_device(device)
            print("COMPILE_ONLY PASS")
            return

        print(
            f"Running {variant.label} warmup={args.warmup} runs={args.runs}...",
            flush=True,
        )
        seconds = time_runs(run_once, device, warmup=args.warmup, runs=args.runs)
        output = ttnn.to_torch(final_dram).reshape(Q_ROWS, VALUE_DIM).to(torch.bfloat16)
        pcc_value = measure_pcc(expected, output)
        max_abs, mean_abs = _error_stats(expected, output)
        row = {
            "label": variant.label,
            "variant": variant.name,
            "seq": SEQ,
            "q_rows": Q_ROWS,
            "value_dim": VALUE_DIM,
            "score_chunk_tiles": SCORE_CHUNK_TILES,
            "n_chunks": N_CHUNKS,
            "warmup": args.warmup,
            "runs": args.runs,
            "ttlang_ms": round(seconds * 1000, 4),
            "pcc": round(pcc_value, 6),
            "max_abs": round(max_abs, 6),
            "mean_abs": round(mean_abs, 6),
        }

        print(
            f"{variant.label}  "
            f"ttlang={row['ttlang_ms']:.4f}ms  "
            f"PCC={row['pcc']:.6f}  "
            f"max_abs={row['max_abs']:.6f}  "
            f"mean_abs={row['mean_abs']:.6f}",
            flush=True,
        )
        print(f"  out[0,:4] = {output[0,:4].tolist()}")
        print(f"  ref[0,:4] = {expected[0,:4].tolist()}")
        if not args.no_csv:
            write_csv(args.csv, FIELDS, row)
            print(f"wrote 1 row to {args.csv}")
        assert_pcc(expected.float(), output.float(), threshold=PCC_THRESHOLD)
        print("BENCH PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
