# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathOperation,
    PerfRunType,
    ReduceDimension,
    ReducePool,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    TemplateParameter,
    LOOP_FACTOR,
    MATH_OP,
    REDUCE_POOL_TYPE,
    TILE_COUNT,
)

REDUCE_MATHOP = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


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


@pytest.mark.perf
@parametrize(
    # Float16 (true fp16) dropped: ttnn has no FLOAT16 dtype and tt-lang maps the
    # name "float16" onto BFLOAT16, so no tt-lang kernel presents a Float16 CB.
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=[DestAccumulation.No],
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    # Average dropped: the DSL exposes reduce_sum and reduce_max only
    # (ttl/operators.py:858-871), and ttkernel.reduce_tile carries the pool type
    # as an attribute the compiler sets from that surface, so no tt-lang kernel
    # can contain a reduce_avg.
    pool_type=[ReducePool.Max, ReducePool.Sum],
    # Which half of the init the measured zone brackets; see the source. Two
    # variants running identical work, differing only in where the bracket sits,
    # is what makes reduce_init attributable instead of lumped with the common
    # init as every unsplit source leaves it.
    measure_op_init=[False, True],
)
def test_perf_reduce(
    perf_report,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    measure_op_init,
):

    tile_count = 16
    configuration = PerfConfig(
        "sources/ttlang_reduce_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=REDUCE_MATHOP[reduce_dim]),
            REDUCE_POOL_TYPE(pool_type),
            MEASURE_OP_INIT(measure_op_init),
        ],
        # 16 tiles x loop factor 2, matching every other sweep here: a per-tile
        # figure is averaged over tile_cnt * loop_factor, so a benchmark reporting
        # over a different N is not comparable with the rest of the table.
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(2)],
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
        unpack_to_dest=False,
        dest_acc=dest_acc,
        # Formats as compile-time constants, which is what tt-lang generates and
        # what every other sweep here uses; see perf_ttlang_eltwise_unary_sfpu.py.
        compile_time_formats=True,
    )

    configuration.run(perf_report)
