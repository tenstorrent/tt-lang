# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
    PerfRunType,
    StableSort,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.sfpu_domains import sfpu_unary_ops
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    TemplateParameter,
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

_OPS_WITHOUT_DEST_ACC = {
    MathOperation.Abs,
    # Acosh/Asinh now select their log1p polynomial precision from the dest-accum
    # (is_fp32_dest_acc_en) flag, so both modes are exercised.
    MathOperation.Celu,
    MathOperation.Cos,
    MathOperation.Elu,
    MathOperation.Exp2,
    MathOperation.Exp,
    MathOperation.Fill,
    MathOperation.Gelu,
    MathOperation.GeluTanh,
    MathOperation.Hardsigmoid,
    MathOperation.Log,
    MathOperation.Neg,
    MathOperation.Silu,
    MathOperation.Sin,
    MathOperation.Square,
    MathOperation.Threshold,
    MathOperation.ReluMax,
    MathOperation.ReluMin,
}

_OPS_WITH_FAST_MODE = {
    MathOperation.Exp,
    MathOperation.Rsqrt,
    MathOperation.Sqrt,
}

_OPS_WITH_STABLE_SORT = {
    MathOperation.TopKLocalSort,
    MathOperation.TopKMerge,
    MathOperation.TopKRebuild,
}


def _get_dest_acc_modes(mathop):
    if mathop in _OPS_WITHOUT_DEST_ACC:
        return [DestAccumulation.No]
    return [DestAccumulation.Yes, DestAccumulation.No]


def _get_fast_modes(mathop):
    if mathop in _OPS_WITH_FAST_MODE:
        return [FastMode.Yes, FastMode.No]
    return [FastMode.No]


def _get_stable_sort_modes(mathop):
    if mathop in _OPS_WITH_STABLE_SORT:
        return [StableSort.Yes, StableSort.No]
    return [StableSort.No]


# Every op with a unary SFPU kernel, taken from the same registry the correctness sweep
# in test_sfpu_unary.py drives, so an op cannot be added there and silently skip perf.
# _UNARY_OPS_NOT_SWEPT is deliberately *not* subtracted: those ops (the topk halves) are
# exempt from the correctness sweep precisely because they are perf-only, so they belong
# here. Sorted so the parametrize ids are stable across runs.
# The SFPU registry operations that have a TTKernel op, and so can appear in a
# tt-lang kernel. An LLK operation with no TTKernel counterpart cannot be reached
# by the compiler, so measuring it produces a row the cost table has nowhere to
# put -- these are exactly the mathops gen_cost_table.py reports as unmapped.
#
# Derived by matching each MathOperation name against the mnemonics in
# include/ttlang/Dialect/TTKernel/IR/TTKernelOps.td (CamelCase -> snake_case plus
# "_tile", with the irregulars spelled out). Regenerate when the dialect gains
# SFPU ops; the assert below catches the reverse, a registry rename.
#
# TopKLocalSort/Merge/Rebuild are excluded despite having TTKernel ops: nothing
# in tt-lang constructs them, and they are the only carriers of stable_sort.
_TTLANG_REACHABLE_OPS = frozenset(
    {
        MathOperation.Abs,
        MathOperation.Acos,
        MathOperation.Asin,
        MathOperation.Atan,
        MathOperation.Ceil,
        MathOperation.Clamp,
        MathOperation.Cos,
        MathOperation.EqualZero,
        MathOperation.Erf,
        MathOperation.Erfc,
        MathOperation.Exp,
        MathOperation.Exp2,
        MathOperation.Expm1,
        MathOperation.Fill,
        MathOperation.Floor,
        MathOperation.Frac,
        MathOperation.Gelu,
        MathOperation.GreaterThanEqualZero,
        MathOperation.GreaterThanZero,
        MathOperation.Hardsigmoid,
        MathOperation.LessThanEqualZero,
        MathOperation.LessThanZero,
        MathOperation.Log,
        MathOperation.Log1p,
        MathOperation.Neg,
        MathOperation.NotEqualZero,
        MathOperation.Reciprocal,
        MathOperation.Rsqrt,
        MathOperation.Selu,
        MathOperation.Sigmoid,
        MathOperation.Sign,
        MathOperation.Signbit,
        MathOperation.Silu,
        MathOperation.Sin,
        MathOperation.Softsign,
        MathOperation.Sqrt,
        MathOperation.Square,
        MathOperation.Tan,
        MathOperation.Tanh,
        MathOperation.Trunc,
    }
)

_MISSING_FROM_REGISTRY = sorted(
    op.name for op in _TTLANG_REACHABLE_OPS - sfpu_unary_ops()
)
assert not _MISSING_FROM_REGISTRY, (
    "named as tt-lang-reachable but absent from the SFPU registry, so the "
    f"mapping is stale: {_MISSING_FROM_REGISTRY}"
)

PERF_SWEEP_OPS = sorted(
    sfpu_unary_ops() & _TTLANG_REACHABLE_OPS, key=lambda op: op.name
)

# Five PerfRunTypes per variant, so all 97 registry ops against all 16 format pairs is
# ~30k ELF builds and profiled runs on llk_perf_tests.yaml's five shards, against ~6.4k
# before the reroute -- and it buys little, since an SFPU kernel's math cost is its
# instruction sequence while the format pair moves unpack/pack cycles, which these ops
# already characterise. So every op is still swept (with its own dest_acc / fast_mode /
# stable_sort / approx_mode), but only the pre-reroute set carries the full 16-pair matrix.
_FULL_FORMAT_MATRIX_OPS = frozenset(
    {
        MathOperation.Reciprocal,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
        MathOperation.Silu,
        MathOperation.Gelu,
        MathOperation.GeluTanh,
        MathOperation.Exp,
        MathOperation.TopKLocalSort,
        MathOperation.TopKMerge,
        MathOperation.TopKRebuild,
    }
    # Intersected with the reachable set for the same reason PERF_SWEEP_OPS is:
    # GeluTanh has no TTKernel op and the TopK ops are never constructed by
    # tt-lang, so they are not swept here and the assert below -- upstream's
    # guard that a declared full-matrix op is actually measured -- would fire.
) & _TTLANG_REACHABLE_OPS

# Float16 (true fp16) is absent: ttnn has no FLOAT16 dtype, and tt-lang maps the
# name "float16" onto BFLOAT16 (ttl/dtype_utils.py:188, "hardware implements f16
# as bf16"), so no tt-lang kernel presents a Float16 CB.
_FULL_FORMATS = [
    DataFormat.Float32,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
]

# Float16_b in and out: the SFPU's native 16-bit exponent-B format, so the measurement is
# the kernel's own cost with no unpack/pack conversion folded in.
#
# Float32 is here for attribution rather than for coverage. This kernel's math loop
# copies SrcA into Dest before the SFPU op unless the unpacker wrote Dest directly,
# so the MATH column covers two operations and cannot be charged to either -- the
# consumer records it as a lump and drops it. `unpack_to_dest` is what removes the
# copy, and it is set exactly when the input is 32-bit and dest_acc is on (see the
# derivation in the test body). With Float16_b alone, only the ten full-matrix ops
# ever reached that configuration, leaving the other thirty with no clean math
# measurement at all.
_REPRESENTATIVE_FORMAT = [DataFormat.Float16_b, DataFormat.Float32]

_FULL_FORMAT_PAIRS = input_output_formats(_FULL_FORMATS)
_REPRESENTATIVE_FORMAT_PAIRS = input_output_formats(_REPRESENTATIVE_FORMAT)

# An op named here but no longer in the sweep would silently stop being measured on the
# full matrix, which is the one regression this split can cause.
_UNSWEPT_FULL_MATRIX_OPS = sorted(
    op.name for op in _FULL_FORMAT_MATRIX_OPS - set(PERF_SWEEP_OPS)
)
assert not _UNSWEPT_FULL_MATRIX_OPS, (
    "these ops are declared as carrying the full format matrix but are not in "
    f"PERF_SWEEP_OPS: {_UNSWEPT_FULL_MATRIX_OPS}"
)


def _get_approx_modes(mathop):
    if mathop == MathOperation.Exp:
        return [ApproximationMode.Yes, ApproximationMode.No]
    return [ApproximationMode.No]


def _get_formats(mathop):
    if mathop in _FULL_FORMAT_MATRIX_OPS:
        return _FULL_FORMAT_PAIRS
    return _REPRESENTATIVE_FORMAT_PAIRS


@dataclass
class MEASURE_OP_INIT(TemplateParameter):
    """Which half of the init block sits inside the measured INIT zone.

    True brackets copy_tile_init's own call with the kernel-wide hw_configure
    hoisted before it; False brackets the hw_configure with the op init hoisted
    after. Both halves run either way and in the same order, so the two variants
    execute identical work and differ only in where the measurement starts.

    Two zones are all the harness supports -- ``read_perf_zone_names_from_elf``
    hardcodes ``[INIT, TILE_LOOP]`` by position -- so splitting the init into two
    measurements means two variants rather than two zones.
    """

    measure_op_init: bool = False

    def convert_to_cpp(self) -> str:
        value = "true" if self.measure_op_init else "false"
        return f"constexpr bool MEASURE_OP_INIT = {value};"


@dataclass
class MEASURE_DATACOPY_ONLY(TemplateParameter):
    """Whether the math loop drops the SFPU call and measures the datacopy alone.

    The subtrahend for recovering an SFPU operation's math cost at formats where
    it cannot be isolated directly. Both variants run the same datacopy, in the
    same kernel and build, over the same tiles, so subtracting one zone from the
    other cancels the pipeline fill instead of leaving it behind.
    """

    measure_datacopy_only: bool = False

    def convert_to_cpp(self) -> str:
        value = "true" if self.measure_datacopy_only else "false"
        return f"constexpr bool MEASURE_DATACOPY_ONLY = {value};"


@pytest.mark.perf
@parametrize(
    formats=lambda mathop: _get_formats(mathop),
    # exp_tile is the one unary op whose TTKernel definition carries an `approx`
    # attribute (ttl.exp(approx=...)), so it is the only one where a tt-lang
    # kernel can select the mode. Elsewhere the attribute does not exist and
    # sweeping it duplicates every row under an unmatchable second key.
    approx_mode=lambda mathop: _get_approx_modes(mathop),
    mathop=PERF_SWEEP_OPS,
    dest_acc=lambda mathop: _get_dest_acc_modes(mathop),
    # 2, matching every source here. N = tile_cnt * loop_factor is what a per-tile
    # figure is averaged over, so loop_factor decides how comparable one
    # benchmark's numbers are with another's; at 16x2 all six report over 32
    # tiles. Higher was tried -- copy_tile hangs at 16, where the repeated
    # blocks saturate a hardware credit rather than miscount one, since the same
    # kernel is correct at 2.
    loop_factor=[
        2,
    ],
    # 8, not 32. This is the SFPU kernel's inner trip count, and tt-metal's own
    # call sites compile 8 in 87 of 88 places; the ttkernel dialect documents the
    # same default inline on exp_tile. Measuring 32 describes a kernel tt-lang
    # never emits.
    iterations=[
        8,
    ],
    # Pinned: no ttkernel op carries a fast-mode attribute, and stable_sort
    # applies only to the TopK ops, which are excluded above.
    fast_mode=[FastMode.No],
    stable_sort=[StableSort.No],
    # 8 and 64 tiles. A per-tile figure is SETUP_TIME / N + PER_TILE, and at N=8
    # the pipeline fill dominates: this kernel's datacopy alone fits 262/N + 16,
    # so at N=8 the fill is three times the steady-state rate. The consumer keeps
    # the largest N (Benchmark.steady), and 8 is retained only so the fill can be
    # seen rather than assumed.
    # 16 tiles: a tt-lang DST region holds 16, so this is the block size a real
    # kernel runs and the one the per-tile figure should describe. Larger N was
    # tried to reach steady state, but the estimator charges per-op costs to
    # kernels shaped like this one, so the fill a 16-tile block genuinely pays is
    # part of the cost, not contamination to be swept away.
    input_dimensions=[
        [128, 128],  # tile_cnt: 16
    ],
    # Which half of the init the measured zone brackets; see the source. Two
    # variants of identical work, differing only in where the bracket sits, is
    # what makes the operation's own init attributable instead of lumped with
    # the common init.
    measure_op_init=[False, True],
    # The datacopy-only variant is the subtrahend; see the class above. Swept
    # across every format, not just the ones where the SFPU op cannot be isolated,
    # so that where a clean measurement does exist the subtraction can be checked
    # against it rather than trusted.
    measure_datacopy_only=[False, True],
)
def test_perf_eltwise_unary_sfpu(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    fast_mode,
    stable_sort,
    input_dimensions,
    measure_op_init,
    measure_datacopy_only,
):
    # Calculate tile count from input dimensions
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    # A 32-bit (fp32) input with dest_acc ON unpacks straight into the 32-bit Dest
    # register. With dest_acc OFF it goes through the source registers (converted to 16-bit)
    # and is copied into Dest for the SFPU op.
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = PerfConfig(
        "sources/ttlang_eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MEASURE_OP_INIT(measure_op_init),
            MEASURE_DATACOPY_ONLY(measure_datacopy_only),
            
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            FAST_MODE(fast_mode),
            STABLE_SORT(stable_sort),
            CLAMP_NEGATIVE(False),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count_A,
            tile_count_B=tile_count_B,
            tile_count_res=tile_count_A,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
