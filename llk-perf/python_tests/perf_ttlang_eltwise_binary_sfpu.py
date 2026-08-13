# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


def get_dest_accum_modes(formats):
    if formats.input_format.is_32_bit() and formats.input_format.is_integer():
        return [DestAccumulation.No]
    return [DestAccumulation.Yes, DestAccumulation.No]


_OPS_WITH_APPROX_MODE = {
    MathOperation.SfpuElwpow,
}


def _get_approx_modes(mathop):
    if mathop in _OPS_WITH_APPROX_MODE:
        return [ApproximationMode.Yes, ApproximationMode.No]
    return [ApproximationMode.No]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    # pow_binary_tiles is the only binary SFPU op whose ttkernel definition
    # carries an `approx` attribute, so it is the only one a tt-lang kernel can
    # select. Elsewhere the mode is metal's default and sweeping it duplicates
    # every row under an unmatchable second key.
    approx_mode=lambda mathop: _get_approx_modes(mathop),
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwdiv,
        MathOperation.SfpuElwpow,
    ],
    dest_acc=[
        DestAccumulation.Yes,
        DestAccumulation.No,
    ],
    # 2, matching every source here. N = tile_cnt * loop_factor is what a per-tile
    # figure is averaged over, so loop_factor decides how comparable one
    # benchmark's numbers are with another's; at 16x2 all six report over 32
    # tiles. Higher was tried -- copy_tile hangs at 16, where the repeated
    # blocks saturate a hardware credit rather than miscount one, since the same
    # kernel is correct at 2.
    loop_factor=[
        2,
    ],
    # 8, not 32: the SFPU inner trip count tt-metal compiles at 87 of its 88
    # call sites, and the default the ttkernel dialect documents on exp_tile.
    iterations=[
        8,
    ],
    input_dimensions=[
        [128, 128],  # tile_cnt: 16
    ],  # Specifying different input sizes to cover different tile counts
)
def test_perf_eltwise_binary_sfpu_float(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/ttlang_eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    # pow_binary_tiles is the only binary SFPU op whose ttkernel definition
    # carries an `approx` attribute, so it is the only one a tt-lang kernel can
    # select. Elsewhere the mode is metal's default and sweeping it duplicates
    # every row under an unmatchable second key.
    approx_mode=lambda mathop: _get_approx_modes(mathop),
    mathop=[
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwLogicalRightShift,
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
    ],
    dest_acc=lambda formats: get_dest_accum_modes(formats),
    # 2, matching every source here. N = tile_cnt * loop_factor is what a per-tile
    # figure is averaged over, so loop_factor decides how comparable one
    # benchmark's numbers are with another's; at 16x2 all six report over 32
    # tiles. Higher was tried -- copy_tile hangs at 16, where the repeated
    # blocks saturate a hardware credit rather than miscount one, since the same
    # kernel is correct at 2.
    loop_factor=[
        2,
    ],
    # 8, not 32: the SFPU inner trip count tt-metal compiles at 87 of its 88
    # call sites, and the default the ttkernel dialect documents on exp_tile.
    iterations=[
        8,
    ],
    input_dimensions=[
        [128, 128],  # tile_cnt: 16
    ],
)
def test_perf_eltwise_binary_sfpu_int(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/ttlang_eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.UInt32,
        ],
        same=True,
    ),
    # pow_binary_tiles is the only binary SFPU op whose ttkernel definition
    # carries an `approx` attribute, so it is the only one a tt-lang kernel can
    # select. Elsewhere the mode is metal's default and sweeping it duplicates
    # every row under an unmatchable second key.
    approx_mode=lambda mathop: _get_approx_modes(mathop),
    mathop=[
        MathOperation.SfpuAddTopRow,
    ],
    dest_acc=lambda formats: get_dest_accum_modes(formats),
    # 2, matching every source here. N = tile_cnt * loop_factor is what a per-tile
    # figure is averaged over, so loop_factor decides how comparable one
    # benchmark's numbers are with another's; at 16x2 all six report over 32
    # tiles. Higher was tried -- copy_tile hangs at 16, where the repeated
    # blocks saturate a hardware credit rather than miscount one, since the same
    # kernel is correct at 2.
    loop_factor=[
        2,
    ],
    # 8, not 32: the SFPU inner trip count tt-metal compiles at 87 of its 88
    # call sites, and the default the ttkernel dialect documents on exp_tile.
    iterations=[
        8,
    ],
    input_dimensions=[
        [128, 128],  # tile_cnt: 16
    ],
)
def test_perf_eltwise_binary_sfpu_add_top_row(
    perf_report,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    loop_factor,
    iterations,
    input_dimensions,
):
    chip_arch = get_chip_architecture()

    # Skip DestAccumulation.No on Blackhole for SfpuAddTopRow
    if chip_arch == ChipArchitecture.BLACKHOLE and dest_acc == DestAccumulation.No:
        pytest.skip(
            "DestAccumulation.No is not supported for SfpuAddTopRow on Blackhole"
        )

    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.Yes:
        pytest.skip("SfpuAddTopRow does not support Float32 with DestAccumulation.Yes")

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No
    )

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/ttlang_eltwise_binary_sfpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
