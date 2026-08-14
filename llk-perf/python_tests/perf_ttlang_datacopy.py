# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep for the datacopy behind tt-metal's ``copy_tile``.

Two things this measures that nothing else does.

**The MATH half of copy_tile, on its own.** The SFPU perf sources run the same
A2D datacopy, but only ahead of an SFPU operation in the same MATH loop, so
their MATH column covers two operations and is attributable to neither. Here
the datacopy is the only work on that thread.

**Both unpack_to_dest modes, with the datacopy present in each.** The SFPU
sources elide the datacopy entirely under ``unpack_to_dest``; ``copy_tile``
does not -- it always issues both calls and passes the mode down to each, so
under the mode the MATH half becomes synchronization rather than disappearing.
That configuration is what a real kernel runs and is currently unmeasured.

``tile_count`` is swept rather than pinned. Each measurement is
``SETUP_TIME / N + PER_TILE``, so a single N leaves the pipeline fill folded
into the per-tile figure -- 44.06 cycles/tile on UNPACK at N=16 against 40.73 at
N=64. Sweeping N is what makes the contamination visible; a consumer can either
fit the two terms apart, the way ``perf_math_matmul`` fits against ``c_dimm``,
or just take the largest N as the steady-state rate.
"""

from dataclasses import dataclass

import pytest
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DataCopyType,
    DestAccumulation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DATA_COPY_TYPE,
    LOOP_FACTOR,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    TILE_COUNT,
    TemplateParameter,
)


@dataclass(frozen=True)
class OpShape:
    """One operation, as the knob settings that select it."""

    name: str
    broadcast_type: BroadcastType
    data_copy_type: DataCopyType
    transpose: Transpose

    def __str__(self) -> str:
        return self.name


_ALL_OP_SHAPES = [
    OpShape("copy_tile", BroadcastType.None_, DataCopyType.A2D, Transpose.No),
    OpShape("unary_bcast_col", BroadcastType.Column, DataCopyType.B2D, Transpose.No),
    OpShape("unary_bcast_row", BroadcastType.Row, DataCopyType.B2D, Transpose.No),
    OpShape("transpose_wh_tile", BroadcastType.None_, DataCopyType.A2D, Transpose.Yes),
]

# Only copy_tile is swept. The others are kept above because the kernel supports
# them and they are one line from being re-enabled, but they do not run on this
# harness yet.
#
# The blocker is the isolate modes rather than the shapes. MATH_ISOLATE returns
# the unpack thread early and fakes the handshake in its place
# (`_perf_unpack_loop_set_valid`), and that fake was written for the SrcA path a
# plain datacopy uses. A broadcast leaves the tile in SrcB and pairs with B2D, so
# math waits on a bank the fake never fills -- the same shape of problem as
# copy_tile's math half under `unpack_to_dest`, unmeasurable here for the same
# reason.
#
# Confirmed on device: all 42 copy_tile variants pass and the first broadcast
# variant hangs. Three real bugs were found and fixed getting that far -- the
# MATH_ISOLATE SrcB handshake, the 32-bit broadcast route (which must use
# unpack-to-dest, since SrcB is 19 bits wide), and the broadcast type missing
# from the math init -- and none was the blocker. No upstream perf source runs a
# unary broadcast datacopy, so there is no reference for what the fake handshake
# should do; only functional tests, which run both threads and never need one.
_OP_SHAPES = [shape for shape in _ALL_OP_SHAPES if shape.name == "copy_tile"]


def _resolve_route(op_shape, formats, unpack_to_dest):
    """The copy type and unpack mode a shape actually runs with.

    A broadcast leaves the tile in SrcB, which is why those shapes pair with B2D
    -- but SrcB is only 19 bits wide, so a 32-bit broadcast cannot go that way at
    all. tt-metal handles this by forcing unpack-to-dest and A2D for 32-bit
    formats (bcast.h:44-62), and a kernel that ignores it hangs: the datacopy
    waits on a SrcB bank the unpacker can never fill.

    Returned rather than fixed in the table because it depends on the format,
    exactly as it does in the compute API.
    """
    if op_shape.broadcast_type == BroadcastType.None_:
        return op_shape.data_copy_type, unpack_to_dest
    if formats.input_format.is_32_bit():
        return DataCopyType.A2D, True
    return op_shape.data_copy_type, unpack_to_dest


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

# unpack_to_dest only changes the route for 32-bit input into a 32-bit Dest;
# for anything else tt-metal ignores the mode, so sweeping it there would
# produce duplicate rows under two different keys.
_UNPACK_TO_DEST_INPUT_FORMATS = {DataFormat.Float32}


def _get_unpack_to_dest_modes(formats, dest_acc):
    if (
        formats.input_format in _UNPACK_TO_DEST_INPUT_FORMATS
        and dest_acc == DestAccumulation.Yes
    ):
        return [False, True]
    return [False]


@pytest.mark.perf
@parametrize(
    # tt-lang's three reachable formats. Float16 (true fp16) is excluded: ttnn
    # has no FLOAT16 dtype -- BFLOAT16, BFLOAT4_B, BFLOAT8_B, FLOAT32, FP8_E4M3,
    # INT32, UINT8/16/32 -- and tt-lang's own dtype mapping sends the name
    # "float16" to BFLOAT16 ("hardware implements f16 as bf16",
    # ttl/dtype_utils.py:188), so no tt-lang kernel can present a Float16 CB.
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    unpack_to_dest=lambda formats, dest_acc: _get_unpack_to_dest_modes(
        formats, dest_acc
    ),
    # 16 tiles: a tt-lang DST region holds 16, so this is the block size a real
    # kernel runs and the one the per-tile figure should describe. Larger N was
    # tried to reach steady state, but the estimator charges per-op costs to
    # kernels shaped like this one, so the fill a 16-tile block genuinely pays is
    # part of the cost, not contamination to be swept away.
    tile_count=[16],
    # The three operations this source covers, as one axis rather than three
    # sweeps. On Blackhole they are the same two LLK calls with one knob moved,
    # so a single kernel measures all three and the consumer reads the shape to
    # decide which operation a row belongs to.
    #
    #   copy_tile          NONE / A2D / no transpose
    #   unary_bcast        Col or Row / B2D   (bcast.h:94-125, and B2D because a
    #                      broadcast leaves the tile in SrcB)
    #   transpose_wh_tile  NONE / A2D, with the unpacker configured to transpose
    #                      in the init -- transpose.h:107-125 issues exactly
    #                      copy_tile's two calls, so the tile loop is identical
    #                      and only the init differs
    op_shape=_OP_SHAPES,
    measure_op_init=[False, True],
)
def test_perf_datacopy(
    perf_report,
    formats,
    dest_acc,
    unpack_to_dest,
    tile_count,
    op_shape,
    measure_op_init,
):
    data_copy_type, unpack_to_dest = _resolve_route(op_shape, formats, unpack_to_dest)

    configuration = PerfConfig(
        "sources/ttlang_datacopy_perf.cpp",
        formats,
        # A2D is the copy copy_tile performs. Passed as a template rather than
        # hardcoded in the kernel: every other perf source supplies at least one
        # template parameter, and omitting them entirely made the generated
        # build.h carry a duplicate `constexpr std::uint32_t TILE_CNT`.
        templates=[
            DATA_COPY_TYPE(data_copy_type),
            BROADCAST_TYPE(op_shape.broadcast_type),
            MEASURE_OP_INIT(measure_op_init),
        ],
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        # Faces are not swept: the kernel uses the derived TILE_NUM_FACES, so a
        # full tile throughout. Partial-face copies would be a separate sweep,
        # and nothing consuming this data models subtiles yet.
        # 16, matching every other source. N = tile_cnt * loop_factor is what a
        # per-tile figure is averaged over, so a benchmark's loop_factor decides
        # how comparable its numbers are with another's. At 16x16 all six sources
        # report over the same 256 tiles.
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(2),
            UNPACK_TRANS_FACES(op_shape.transpose),
            UNPACK_TRANS_WITHIN_FACE(op_shape.transpose),
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
        # Formats as compile-time constants, which is what tt-lang generates: CB
        # data formats come from the tile types in the IR and are baked into the
        # emitted kernel. Under runtime formats they are loaded from the params
        # struct instead, and a one-shot init zone is where that shows -- the
        # unary SFPU init read 474 cycles that way against 356 this way, which
        # was the whole of its disagreement with the binary source.
        compile_time_formats=True,
    )

    configuration.run(perf_report)
