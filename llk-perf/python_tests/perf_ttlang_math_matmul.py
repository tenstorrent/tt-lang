# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import chain, product

import pytest
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    StochasticRounding,
)
from helpers.matmul_sweep import sweep_matmul, sweep_tiny_tiles_matmul
from helpers.param_config import input_output_formats
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    PARTIAL_FACE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Bfp8_b,
        DataFormat.Float16_b,
        DataFormat.Float32,
    ]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]
STOCHASTIC_ROUNDING_MODES = [StochasticRounding.No]
MATH_FIDELITIES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]

# `sweep_matmul` emits each configuration twice for math_matmul, at dst_index 0
# and at the highest valid index. Upstream keeps both as a correctness edge case;
# for perf they are the same work at a different register offset, and the
# consumer's key has no dst_index field, so the two land on one key and are
# averaged. Keeping index 0 alone halves the sweep and loses nothing the cost
# table can express.
MATMUL_COMBINATIONS = [
    config
    for config in sweep_matmul(
        MATMUL_FORMATS,
        DEST_ACC_MODES,
        STOCHASTIC_ROUNDING_MODES,
        DEST_SYNC_MODES,
        math_matmul=True,
    )
    if config.dst_index == 0
    # A 32-bit input needs a 32-bit Dest. `sweep_matmul` guards this with
    # is_dest_acc_needed, which inspects only the output format, so a
    # Float32 -> Bfp8_b pair is emitted with dest_acc=No and the kernel cannot be
    # built -- 960 of 4032 combinations here, all of which failed with a missing
    # ELF. Upstream never hit it because it runs only the tiny-tile sweep.
    and not (
        config.formats.input_format.is_32_bit()
        and config.dest_acc == DestAccumulation.No
    )
]

TINY_TILES_MATMUL_COMBINATIONS = sweep_tiny_tiles_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
    math_matmul=True,
)

# Full 32x32 tiles at throttle level 0.
#
# Upstream runs only TINY_TILES_MATMUL_COMBINATIONS, the regular ones having been
# commented out for CI disk space. For tt-lang that keeps the wrong half: a
# tt-lang tile is 32x32 (four faces), so every row the tiny-tile sweep produces
# is keyed `faces != 4` and unreachable, and tiny tiles are not reachable from
# tt-lang compute at all.
#
# Throttle stays 0: the word appears nowhere in the ttkernel dialect or tt-lang's
# lowering, so levels 1-5 would key rows nothing can match.
ALL_TEST_PARAMS = [
    (fidelity, combinations, 0)
    for fidelity, combinations in product(MATH_FIDELITIES, MATMUL_COMBINATIONS)
]


@pytest.mark.perf
@pytest.mark.parametrize("math_fidelity,matmul_config,throttle", ALL_TEST_PARAMS)
def test_perf_math_matmul(
    math_fidelity,
    matmul_config,
    throttle,
    perf_report,
):
    """
    Performance test for matmul operations.

    Includes both regular matmul (full 32x32 tiles) and tiny tiles matmul
    (input 0 with rows: 1, 2, 4, 8, 16 and columns: 32, input 1 always 32x32).
    """
    formats = matmul_config.formats
    in0_dimensions = matmul_config.tile_dimensions.in0_dimensions
    in1_dimensions = matmul_config.tile_dimensions.in1_dimensions
    transpose = matmul_config.face_layout_config.unpack_transpose_faces
    num_faces_in0 = matmul_config.face_layout_config.num_faces_in0
    num_faces_in1 = matmul_config.face_layout_config.num_faces_in1
    num_faces = matmul_config.face_layout_config.num_faces

    if is_dest_acc_needed(formats) and matmul_config.dest_acc == DestAccumulation.No:
        pytest.skip("Dest accumulation must be enabled for this format")

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    variant_tile_count = (
        matmul_config.tile_dimensions.rt_dim
        * matmul_config.tile_dimensions.ct_dim
        * matmul_config.tile_dimensions.kt_dim
    )

    configuration = PerfConfig(
        "sources/math_matmul_perf.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(matmul_config.dest_sync),
            THROTTLE_LEVEL(throttle),
        ],
        runtimes=[
            DEST_INDEX(matmul_config.dst_index),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            TILE_COUNT(variant_tile_count),
            NUM_FACES(
                num_faces, num_faces_in0, num_faces_in1
            ),  # In0 -> Input A, In1 -> Input B
            PARTIAL_FACE(  # In0 -> Input A, In1 -> Input B
                partial_a=matmul_config.face_layout_config.partial_face_in0,
                partial_face_pack=matmul_config.face_layout_config.partial_face_pack,
                partial_b=matmul_config.face_layout_config.partial_face_in1,
                partial_face_math=matmul_config.face_layout_config.partial_face_math,
            ),
            CRK_TILE_DIMM(
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.kt_dim,
            ),
            IN_TILE_DIMS(
                matmul_config.tile_dimensions.in0_tile_r_dim,
                matmul_config.tile_dimensions.in0_tile_c_dim,
                matmul_config.tile_dimensions.in1_tile_r_dim,
                matmul_config.tile_dimensions.in1_tile_c_dim,
            ),
            LOOP_FACTOR(1024),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=matmul_config.tile_dimensions.tile_cnt_in0,
            tile_count_B=matmul_config.tile_dimensions.tile_cnt_in1,
            tile_count_res=matmul_config.tile_dimensions.output_tile_cnt,
        ),
        dest_acc=matmul_config.dest_acc,
        # Formats as compile-time constants, which is what tt-lang generates: CB
        # data formats come from the tile types in the IR and are baked into the
        # emitted kernel. Under runtime formats they are loaded from the params
        # struct instead, and a one-shot init zone is where that shows -- the
        # unary SFPU init read 474 cycles that way against 356 this way, which
        # was the whole of its disagreement with the binary source.
        compile_time_formats=True,
    )

    configuration.run(perf_report)
