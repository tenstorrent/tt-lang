# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.format_config import DataFormat
from helpers.llk_params import (
    MathFidelity,
    MathOperation,
    PerfRunType,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    TemplateParameter,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    TILE_COUNT,
)


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
    # tt-lang's reachable formats. Float16 (true fp16) is excluded: ttnn has no
    # FLOAT16 dtype, and tt-lang maps the name "float16" onto BFLOAT16
    # (ttl/dtype_utils.py:188), so no tt-lang kernel presents a Float16 CB.
    # Float32 is added: it was absent, yet tt-lang runs f32 eltwise
    # (test/python/simple_add_f32.py), so every f32 add/sub/mul fell back to a
    # placeholder for want of a measurement.
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16_b, DataFormat.Float32]
    ),
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    # 16 tiles: a tt-lang DST region holds 16, so this is the block size a real
    # kernel runs and the one the per-tile figure should describe. Larger N was
    # tried to reach steady state, but the estimator charges per-op costs to
    # kernels shaped like this one, so the fill a 16-tile block genuinely pays is
    # part of the cost, not contamination to be swept away.
    tile_count=[16],
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(
        formats, mathop, PERF_RUN=True
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    # Which half of the init the measured zone brackets; see the source. Two
    # variants of identical work, differing only in where the bracket sits, is
    # what makes the operation's own init attributable instead of lumped with
    # the common init.
    measure_op_init=[False, True],
)
def test_perf_eltwise_binary_fpu(
    perf_report,
    formats,
    mathop,
    tile_count,
    math_fidelity,
    dest_acc,
    measure_op_init,
):
    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    configuration = PerfConfig(
        "sources/ttlang_eltwise_binary_fpu_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[MEASURE_OP_INIT(measure_op_init), MATH_FIDELITY(math_fidelity), MATH_OP(mathop=mathop)],
        # 16, matching every other source. N = tile_cnt * loop_factor is what a
        # per-tile figure is averaged over, so a benchmark's loop_factor decides
        # how comparable its numbers are with another's. At 16x16 all six sources
        # report over the same 256 tiles.
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
        dest_acc=dest_acc,
        # Formats as compile-time constants, which is what tt-lang generates: CB
        # data formats come from the tile types in the IR and are baked into the
        # emitted kernel. Under runtime formats they are loaded from the params
        # struct instead, and a one-shot init zone is where that shows -- the
        # unary SFPU init read 474 cycles that way against 356 this way, which
        # was the whole of its disagreement with the binary source.
        compile_time_formats=True,
    )

    configuration.run(perf_report)
