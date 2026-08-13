#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the CostEstimator work-cost table from LLK perf CSVs.

Reads ``perf_data/<name>/<name>.post.csv``, fetched from the nightly tt-llk perf
run, and emits a hardcoded C++ table of per-engine work costs, consumed by
lib/OpCost.

Usage::

    scripts/gen_cost_table.py --report              # coverage and diagnostics
    scripts/gen_cost_table.py -o lib/OpCost/CostTableBlackhole.inc
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib
import re
import subprocess
import sys
from collections import defaultdict
from typing import Optional

from lane_work import LANE_WORK, PER_CALL, PER_TILE, Cost

REPO = pathlib.Path(__file__).resolve().parent.parent
PERF_DATA = REPO / "perf_data"
TTKERNEL_OPS_TD = REPO / "include/ttlang/Dialect/TTKernel/IR/TTKernelOps.td"

# Architecture the nightly run measured.  Costs are not transferable across
# architectures, so this is part of the generated table's identity rather than an
# assumption left implicit.
ARCH = "blackhole"

LANES = ("unpack", "math", "pack")

# Prefix marking a lane whose loop body contains more than one operation.  The
# op names follow, joined by "+", so a lump keeps the identity of what was
# measured: it stays visible and recoverable by subtracting a separately measured
# component, but is never attributed to a single op.
LUMP = "lump:"


# ---------------------------------------------------------------------------
# mathop -> ttkernel op name
# ---------------------------------------------------------------------------

# Keys are the CSV's `mathop` values with the `MathOperation.` prefix stripped.
# A missing entry is reported, not guessed: an LLK operation with no ttkernel op
# cannot appear in a tt-lang kernel, so its rows carry no cost anyone could look
# up.  llk-perf/ narrows the sweeps to the reachable set, so an unmapped mathop
# appearing here now means the two have drifted apart and one needs updating.
MATHOP_TO_OP = {
    # FPU binary, eltwise_binary_fpu_perf.cpp
    "Elwadd": "add_tiles",
    "Elwsub": "sub_tiles",
    "Elwmul": "mul_tiles",
    # SFPU binary, eltwise_binary_sfpu_perf.cpp
    "SfpuElwadd": "add_binary_tile",
    "SfpuElwsub": "sub_binary_tile",
    "SfpuElwmul": "mul_binary_tile",
    "SfpuElwdiv": "div_binary_tile",
    "SfpuElwpow": "power_tile",
    "SfpuElwLeftShift": "binary_left_shift_tile",
    "SfpuElwRightShift": "binary_right_shift_tile",
    "SfpuElwLogicalRightShift": "binary_logical_right_shift_tile",
    # SFPU unary, eltwise_unary_sfpu_perf.cpp
    "Exp": "exp_tile",
    "Gelu": "gelu_tile",
    "Log": "log_tile",
    "Sqrt": "sqrt_tile",
    "Rsqrt": "rsqrt_tile",
    "Reciprocal": "recip_tile",
    "Square": "square_tile",
    "Silu": "silu_tile",
    "Softsign": "softsign_tile",
    "Abs": "abs_tile",
    "Acos": "acos_tile",
    "Asin": "asin_tile",
    "Atan": "atan_tile",
    "Ceil": "ceil_tile",
    "Clamp": "clamp_tile",
    "Cos": "cos_tile",
    "EqualZero": "eqz_tile",
    "Erf": "erf_tile",
    "Erfc": "erfc_tile",
    "Exp2": "exp2_tile",
    "Expm1": "expm1_tile",
    "Floor": "floor_tile",
    "Frac": "frac_tile",
    "GreaterThanEqualZero": "gez_tile",
    "GreaterThanZero": "gtz_tile",
    "Hardsigmoid": "hardsigmoid_tile",
    "LessThanEqualZero": "lez_tile",
    "LessThanZero": "ltz_tile",
    "Log1p": "log1p_tile",
    "Neg": "negative_tile",
    "NotEqualZero": "nez_tile",
    "Selu": "selu_tile",
    "Sigmoid": "sigmoid_tile",
    "Sign": "sign_tile",
    "Signbit": "signbit_tile",
    "Sin": "sin_tile",
    "Tan": "tan_tile",
    "Tanh": "tanh_tile",
    "Trunc": "trunc_tile",
    "TopKLocalSort": "topk_local_sort",
    "TopKMerge": "topk_merge",
    "TopKRebuild": "topk_rebuild",
    # reduce_perf.cpp.  All three name the reduce dimension rather than a distinct
    # operation -- the operation is `reduce_tile` either way, and the dimension is
    # a key field, so they share one mapping.
    "ReduceRow": "reduce_tile",
    "ReduceColumn": "reduce_tile",
    "ReduceScalar": "reduce_tile",
    # eltwise_unary_typecast_perf.cpp
    "Typecast": "typecast_tile",
    # eltwise_unary_sfpu_perf.cpp, int32 sweep.  Every row there is
    # unpack_to_dest=True, so the math loop skips the datacopy and the column is
    # the SFPU op alone -- clean single-op attribution, unlike the bf16 sweep.
    "AbsInt32": "abs_tile_int32",
    "AddInt32": "add_int_tile",
    "BitwiseNot": "bitwise_not_tile",
    "Fill": "fill_tile",
    "LogicalNot": "logical_not_tile",
    "SubInt32": "sub_int_tile",
}


def _mathop(row: dict) -> Optional[str]:
    """ttkernel op for a row's `mathop`, or None when unmapped."""
    return MATHOP_TO_OP.get(row.get("mathop", "").split(".")[-1])


def _one(row: dict) -> Optional[list[str]]:
    """Single owner taken from the row's `mathop`."""
    op = _mathop(row)
    return [op] if op else None


def _sfpu_math(row: dict) -> Optional[list[str]]:
    """Owners of the MATH lane in an SFPU benchmark.

    The math loop copies SrcA into DST before the SFPU operation whenever the
    unpacker did not write DST directly, so the measured column covers two ops.
    With `unpack_to_dest` the copy is skipped and the column is the SFPU op
    alone.  eltwise_unary_sfpu_perf.cpp:200-218.
    """
    op = _mathop(row)
    if op is None:
        return None
    return [op] if row.get("unpack_to_dest") == "True" else ["copy_tile", op]


def _copy_tile_math(row: dict) -> Optional[list[str]]:
    """Owner of the MATH tile loop in the copy_tile benchmark, where there is one.

    ``MATH_ISOLATE`` elides the datacopy under ``unpack_to_dest``
    (copy_tile_perf.cpp:225-235): the isolate returns the unpack thread early,
    and ``math_unpack_to_dest_math_ready()`` spins on semaphores that thread
    would have posted, so running it deadlocks the device.  The zone is then an
    empty loop and its column is loop overhead -- 2.56 cycles/tile at N=16 and
    0.64 at N=64, which is one fixed cost divided by N and no per-tile rate at
    all.  Attributing that to `copy_tile` would put a measured 0.63 where the
    real number is unknown, which is worse than the placeholder it replaces.

    So the mode's math cost is still unmeasured, and this is the one place the
    benchmark does not cover what it was written to cover.  Recovering it needs
    an isolate that keeps the handshake, not a different attribution.
    """
    return None if row.get("unpack_to_dest") == "True" else ["copy_tile"]


def _copy_tile_init(row: dict) -> list[str]:
    """Owner of the UNPACK or MATH init lane in the copy_tile benchmark.

    ``measure_op_init`` is not a configuration -- both variants execute the same
    calls in the same order -- but a selector for which half of the init the
    measured zone brackets, so it names the owner rather than joining the key:

    * True brackets ``_llk_unpack_A_init_`` / ``_llk_math_eltwise_unary_datacopy_init_``,
      which is exactly what ``copy_tile_init`` issues.
    * False brackets ``_llk_unpack_hw_configure_`` and ``_llk_math_pack_sync_init_ +
      _llk_math_hw_configure_``, which is exactly what ``compute_kernel_hw_startup``
      issues on those two threads (compute_kernel_hw_startup.h:81-84).

    Splitting the zone in two is what makes either half attributable: measured
    together they would be a lump, which is the state every other benchmark's
    init zone is still in.  copy_tile_perf.cpp:97-129, 186-212.
    """
    return (
        ["copy_tile_init"]
        if row.get("measure_op_init") == "True"
        else ["compute_kernel_hw_startup"]
    )


def _init_math(common: str):
    """Owners of the MATH lane of an init zone: the common init plus the op's own.

    The math zone brackets ``pack_sync_init + hw_configure + <op>_init``
    (eltwise_binary_fpu_perf.cpp:97-99, and the same shape in the SFPU sources),
    so it covers two ttkernel ops and is a lump like any other -- recorded, never
    attributed.  The unpack and pack zones are the common init alone (:44-53,
    :165-167) and stay attributed to it.

    This cannot be detected from the numbers.  A benchmark sweeping ops whose
    inits cost the same hides the second op inside a constant: the FPU sweep's
    three mathops all route through ``binary_tiles_init`` and show 0.0% spread,
    while the SFPU sweep's eighteen show 42%.  Both are contaminated; only one
    looks it.  The kernel source is the authority here, not the variance.
    """

    def owners(row) -> Optional[list[str]]:
        op = _mathop(row)
        if op:
            return [common, f"{op}_init"]
        # The zone still contains an op-specific init even when the mathop has no
        # ttkernel op to name it with.  Naming the contaminant is what makes a
        # lump recoverable later; not naming it does not make the row clean, so
        # the raw mathop stands in rather than the row falling back to a
        # single-owner attribution.
        raw = row.get("mathop", "").split(".")[-1]
        return [common, f"{raw}_init"] if raw else None

    return owners


# ---------------------------------------------------------------------------
# Benchmark attribution
# ---------------------------------------------------------------------------
#
# Attribution is the whole problem this script exists to get right.  A CSV row is
# *not* one operation's cost vector.  Each benchmark builds three independent
# kernels behind `#ifdef LLK_TRISC_UNPACK` / `_MATH` / `_PACK`, and an
# `*_ISOLATE` run measures one of them with the other two returning early.  So a
# row is a per-lane decomposition of the benchmark's loop body, and each column
# belongs to whichever ttkernel op owns that lane -- often a different op per
# column.  `matmul_block` is an FPU op: it owns UNPACK and MATH, and the PACK
# column of the matmul benchmark belongs to `pack_tile`.
#
# Reading a row as one op's cost vector is how you conclude that an FPU op does
# packer work.  So the owners below were read out of the kernel sources, are
# cited line by line, and a lane whose loop body runs more than one op is
# recorded as a lump instead of being attributed.
#
# What the CSVs do and do not contain:
#
#   * `*_ISOLATE` strips the cross-thread credit handshakes -- the harness fakes
#     dvalid (`_perf_unpack_loop_set_valid` / `_perf_math_loop_clear_valid`) and
#     drops `wait_for_dest_available` / `packer_wait_for_math_done`.  That makes
#     it pure work time, which is what the estimator wants, since it derives the
#     stalls itself from the credit counters.
#   * `L1_TO_L1` keeps the handshakes, so it is a validation target, not an input.
#   * No perf source isolates a handshake, and none of the 18 touches a circular
#     buffer.  The DST lifecycle ops and the CB ops therefore have no data here
#     and are modelled as pure synchronization: zero work, resource effect only.


@dataclasses.dataclass(frozen=True)
class Benchmark:
    """How to read one benchmark's CSV.

    `tile_loop` and `init` map a lane to the ttkernel ops owning that lane's
    zone: a list of op names, or a callable taking the CSV row where the op
    varies per row.  None means the lane's loop body is empty in that benchmark.
    """

    source: str
    """tt-llk kernel source the attribution was read from."""

    tile_loop: dict[str, object]
    init: dict[str, object]

    faces: int
    """Faces per tile the kernel configures.  Part of the key: the same LLK
    measured at 2 vs 4 faces differs by ~50%, so a 2-face number must never
    satisfy a 4-face lookup."""

    keys: tuple[str, ...] = ()
    """Extra CSV columns that form part of an entry's identity, beyond the
    universal format/dest_acc/fidelity/dst_sync ones.  A knob the sweep varies
    but the key omits silently averages unlike measurements together, so
    anything a benchmark sweeps belongs here."""

    dim: Optional[str] = None
    """Column to fit cost against, for benchmarks that sweep a block dimension.
    None means a single per-tile number."""

    steady: Optional[str] = None
    """Column swept only to expose the pipeline fill, of which the largest value
    is kept and the rest discarded.

    Each measurement is ``SETUP_TIME / N + PER_TILE``: an execution unit's
    pipeline takes a fixed time to fill, and the harness divides the zone by the
    tile count, so a small N carries the fill folded into the per-tile figure --
    for copy_tile's UNPACK lane, 44.06 cycles/tile at N=16 against 40.73 at N=64.
    They are the same rate measured with different amounts of contamination, not
    two configurations, so they must not become two keys, and averaging them
    (which is what leaving the column out of `keys` does) yields a number
    matching neither.

    Keeping the largest N is the cheap approximation to the steady-state rate:
    at N=64 the residual fill is under 1% of the fitted asymptote.  `dim` is the
    exact alternative -- it fits the fill and the rate apart -- but it emits a
    non-zero fixed term, which the estimator's lookup currently rejects, and
    tt-lang does not model per-lane pipelining yet."""

    note: str = ""


# Knobs the SFPU benchmarks sweep.
#
# `iterations` stays: it scales the SFPU loop directly, and `exp_tile` is one of
# the few ttkernel ops that can carry the attribute, so a kernel really can ask
# for a value other than the default the sweep measures.
#
# `fast_mode` and `stable_sort` are gone.  No ttkernel op can express either, so
# tt-lang always compiles metal's default and the sweep now pins both to it.  A
# knob measured at exactly one value is not part of an entry's identity -- it
# cannot separate two rows, and emitting it only gives the lookup something else
# to reject.  Provenance for what was pinned lives in llk-perf/README.md.
SFPU_KEYS = (
    "unpack_to_dest",
    "approx_mode",
    "iterations",
)


def _lanes(unpack=None, math=None, pack=None) -> dict:
    return {"unpack": unpack, "math": math, "pack": pack}


# INIT is one lumped measurement per lane covering every init call in the zone,
# so it is attributed to the group's primary op rather than split among them.
BENCHMARKS: dict[str, Benchmark] = {
    # UNPACK loop `_llk_unpack_AB_` (:71), MATH `_llk_math_eltwise_binary_`
    # (:121), PACK `_llk_pack_` (:184).  Full tiles: num_faces 4 (:51-52),
    # TILE_WIDTH*TILE_HEIGHT (:165).  INIT zones: hw_configure + AB_init
    # (:44-53); pack_sync_init + hw_configure + eltwise_binary_init (:97-99);
    # pack hw_configure + pack_init + dest_init (:165-167).
    "perf_ttlang_eltwise_binary_fpu": Benchmark(
        source="eltwise_binary_fpu_perf.cpp",
        tile_loop=_lanes(unpack=_one, math=_one, pack=["pack_tile"]),
        init=_lanes(
            unpack=["binary_op_init_common"],
            math=_init_math("binary_op_init_common"),
            pack=["binary_op_init_common"],
        ),
        faces=4,
        keys=("unpack_to_dest",),
        steady="tile_cnt",
    ),
    # UNPACK loop `_llk_unpack_A_` (:153), MATH `_llk_math_eltwise_unary_datacopy_`
    # (:231/:267/:282), PACK `_llk_pack_` (:331/:344).  Full tiles: TILE_NUM_FACES,
    # TILE_WIDTH*TILE_HEIGHT (:312).  The INIT zones are split on the unpack and
    # math threads (see _copy_tile_init); the pack thread's is not, because
    # copy_tile_init has no pack half at all -- its three calls (:312-314) are
    # compute_kernel_hw_startup's pack half in the same order, under either
    # measure_op_init.  That the two variants then measure the same thing and
    # agree to 0.3% (297 vs 298 cycles) is the control on the split.
    "perf_ttlang_copy_tile": Benchmark(
        source="copy_tile_perf.cpp",
        tile_loop=_lanes(
            unpack=["copy_tile"], math=_copy_tile_math, pack=["pack_tile"]
        ),
        init=_lanes(
            unpack=_copy_tile_init,
            math=_copy_tile_init,
            pack=["compute_kernel_hw_startup"],
        ),
        faces=4,
        # The only knob swept.  No SFPU operation runs here, so unlike every
        # other source measuring a datacopy this one carries no approx_mode or
        # iterations in its key -- which is what makes its rows reachable from
        # the estimator at all.
        keys=("unpack_to_dest",),
        steady="tile_cnt",
    ),
    # UNPACK loop `_llk_unpack_AB_reduce_` (:78), MATH `_llk_math_reduce_` (:141).
    # Every zone but those two is a lump:
    #
    #   INIT, all three threads -- hw_configure plus the thread's half of
    #     `reduce_init` (:49-58, :112-114, :190-193), the same shape as every
    #     other source's init.
    #   PACK TILE_LOOP -- `_llk_pack_reduce_mask_clear_` sits inside the bracket
    #     alongside the pack loop (:242), so the zone covers `pack_tile` and part
    #     of `reduce_uninit`. Dropping it costs nothing: `pack_tile` is measured
    #     cleanly by four other benchmarks.
    #
    # `mathop` here names the reduce dimension rather than an operation -- the
    # operation is always `reduce_tile` -- so it joins the key instead of
    # selecting the owner.
    "perf_ttlang_reduce": Benchmark(
        source="ttlang_reduce_perf.cpp",
        tile_loop=_lanes(unpack=["reduce_tile"], math=["reduce_tile"], pack=None),
        init=_lanes(*(["compute_kernel_hw_startup", "reduce_init"],) * 3),
        faces=4,
        keys=("unpack_to_dest", "mathop", "reduce_pool_type"),
    ),
    # UNPACK `_llk_unpack_AB_matmul_` (:99), MATH `_llk_math_matmul_` (:174),
    # PACK `_llk_pack_` over CT*RT (:244).
    "perf_ttlang_math_matmul": Benchmark(
        source="math_matmul_perf.cpp",
        tile_loop=_lanes(
            unpack=["matmul_block"], math=["matmul_block"], pack=["pack_tile"]
        ),
        init=_lanes(*(["mm_block_init"],) * 3),
        # Fallback only: this benchmark reports `num_faces` per row, so the key
        # takes the row's value and this is used only if the column disappears.
        faces=2,
        # `r_dimm` and `k_dimm` are block dimensions the cost genuinely depends
        # on, and the two transpose columns select a different unpack path.  All
        # four were constant in the tiny-tile data this benchmark used to
        # produce -- 1, 1, No, No -- so leaving them out of the key cost nothing
        # and looked correct.  The full-tile sweep varies every one of them
        # (r 1-8, k 1-4, transpose both ways), at which point their absence
        # would silently average unlike measurements onto one key.  `c_dimm` is
        # the exception: it stays out because `dim` fits against it.
        keys=(
            "unpack_to_dest",
            "throttle_level",
            "in0_r_dim",
            "partial_face_math",
            "r_dimm",
            "k_dimm",
            "unpack_transpose_faces",
            "unpack_transpose_within_face",
        ),
        dim="c_dimm",
    ),
    # UNPACK `_llk_unpack_A_` (:85) is copy_tile's unpack half; MATH is datacopy
    # plus the SFPU op (:207-218); PACK `_llk_pack_` (:310).
    "perf_ttlang_eltwise_unary_sfpu": Benchmark(
        source="eltwise_unary_sfpu_perf.cpp",
        tile_loop=_lanes(unpack=["copy_tile"], math=_sfpu_math, pack=["pack_tile"]),
        init=_lanes(
            unpack=["init_sfpu"],
            math=_init_math("init_sfpu"),
            pack=["init_sfpu"],
        ),
        faces=4,
        keys=SFPU_KEYS,
    ),
    "perf_ttlang_eltwise_unary_sfpu_int32": Benchmark(
        source="eltwise_unary_sfpu_perf.cpp",
        tile_loop=_lanes(unpack=["copy_tile"], math=_sfpu_math, pack=["pack_tile"]),
        init=_lanes(
            unpack=["init_sfpu"],
            math=_init_math("init_sfpu"),
            pack=["init_sfpu"],
        ),
        faces=4,
        keys=SFPU_KEYS,
    ),
    "perf_ttlang_eltwise_binary_sfpu": Benchmark(
        source="eltwise_binary_sfpu_perf.cpp",
        tile_loop=_lanes(unpack=["copy_tile"], math=_sfpu_math, pack=["pack_tile"]),
        init=_lanes(
            unpack=["init_sfpu"],
            math=_init_math("init_sfpu"),
            pack=["init_sfpu"],
        ),
        faces=4,
        keys=SFPU_KEYS,
    ),
    "perf_ttlang_eltwise_typecast": Benchmark(
        source="eltwise_unary_typecast_perf.cpp",
        tile_loop=_lanes(unpack=["copy_tile"], math=_sfpu_math, pack=["pack_tile"]),
        init=_lanes(
            unpack=["init_sfpu"],
            math=_init_math("init_sfpu"),
            pack=["init_sfpu"],
        ),
        faces=4,
        keys=SFPU_KEYS,
    ),
}

# Present in perf_data but deliberately unattributed.  Guessing an owner would
# put a number in the table that nothing verified.
UNATTRIBUTED = {
    "perf_fast_tilize_full": "tilize_block path not read; has an UNINIT zone the "
    "two-zone INIT/TILE_LOOP model does not cover",
    "perf_fast_untilize": "untilize_block path not read; same UNINIT zone",
    "perf_fast_untilize_baseline_compare": "baseline comparison run, not a "
    "primary measurement",
    "perf_eltwise_bcast_col_custom": "broadcast variant; needs the unary_bcast "
    "lowering checked before its column can be assigned an op",
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Key:
    op: str
    lane: str
    in_format: str
    out_format: str
    dest_acc: bool
    fidelity: str
    dst_sync: str
    faces: int
    variant: tuple[tuple[str, str], ...]
    """Benchmark-specific knobs, from Benchmark.keys."""


@dataclasses.dataclass
class Samples:
    """Measurements collected for one key, with where they came from."""

    values: list[float] = dataclasses.field(default_factory=list)
    dims: list[float] = dataclasses.field(default_factory=list)
    sources: set[str] = dataclasses.field(default_factory=set)
    tiles: list[float] = dataclasses.field(default_factory=list)
    """Effective N each value was measured over, for `keep_steadiest`.

    N is `tile_cnt * loop_factor`, not `tile_cnt`: the harness repeats the whole
    tile loop `loop_factor` times and normalises by the total, so the fill is
    amortised over both.  Reading `tile_cnt` alone gets the ordering backwards --
    copy_tile runs 64x1 while the SFPU sweeps run 8x16, so the sweep that looks
    eight times smaller is actually twice as close to steady state."""

    def keep_steadiest(self) -> None:
        """Drop every sample not taken at the largest tile count seen.

        A per-tile figure is SETUP_TIME / N + PER_TILE, so two benchmarks
        reporting the same operation at different N are not reporting the same
        quantity.  `pack_tile` is measured by four sources: copy_tile and the FPU
        sweep at N=64, the two SFPU sweeps at N=8, where the pipeline fill is
        spread over eight tiles instead of sixty-four.  They disagreed by 6-10%,
        which tripped the cross-source check and cost the operation its only
        measurement -- a real number lost to an artefact of test shape.

        `Benchmark.steady` does this within one benchmark that sweeps N; this is
        the same rule across benchmarks that each pin a different one.
        """
        if not self.tiles or len(set(self.tiles)) == 1:
            return
        best = max(self.tiles)
        keep = [i for i, t in enumerate(self.tiles) if t == best]
        self.values = [self.values[i] for i in keep]
        if self.dims:
            self.dims = [self.dims[i] for i in keep]
        self.tiles = [best] * len(keep)

    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    def spread(self) -> float:
        """Max relative deviation from the mean, as a fraction."""
        m = self.mean()
        return max(abs(v - m) for v in self.values) / m if m else 0.0

    def by_dim(self) -> tuple[list[float], list[float]]:
        """Total cycles against block dimension, averaged per distinct dim."""
        acc: dict[float, list[float]] = defaultdict(list)
        for v, d in zip(self.values, self.dims):
            acc[d].append(v * d)
        xs = sorted(acc)
        return xs, [sum(acc[x]) / len(acc[x]) for x in xs]


# Math operations whose per-tile cost depends on math fidelity.
#
# Fidelity is a count of multiply passes, and the compute API decides per op
# whether to pass it through: mul_tiles forwards MATH_FIDELITY to
# llk_math_eltwise_binary (eltwise_binary.h:184) while add_tiles pins
# MathFidelity::LoFi in the template argument (eltwise_binary.h:212), as does
# sub_tiles.  So an addition runs its math at LoFi no matter what the kernel is
# compiled with, and the sweep measuring Elwadd/Elwsub at LoFi only is complete
# coverage rather than a gap.
#
# An op not listed here gets an empty fidelity in the key, so a kernel at any
# fidelity matches it.  Without that, an addition compiled at the ttnn default of
# HiFi4 would find no entry at all.
FIDELITY_SENSITIVE_MATHOPS = {"Elwmul"}


def _fidelity(row: dict, bench: Benchmark, lane: str) -> str:
    """Fidelity for the key, empty where the measurements show no dependence.

    Three cases:

    * Only the MATH lane can depend on it at all.  Across the FPU sweep's four
      fidelities UNPACK holds at 42.69-42.75 and PACK at 29.31-29.69 cycles/tile,
      differences well under a percent, while MATH moves by 4x.
    * An INIT zone always keeps it.  add_tiles_init goes through
      binary_tiles_init, which forwards MATH_FIDELITY to
      llk_math_eltwise_binary_init (eltwise_binary.h:77), and the measurements
      follow: 139/140/140/143 cycles for HiFi2/HiFi3/HiFi4/LoFi, identical across
      Elwadd, Elwmul and Elwsub.  So the init tracks the kernel's fidelity even
      for an op whose own math does not.
    * A TILE_LOOP entry keeps it only for the ops that forward it; see
      FIDELITY_SENSITIVE_MATHOPS.
    """
    if lane != "math":
        return ""
    if row.get("marker") == "INIT":
        return row.get("math_fidelity", "").split(".")[-1]
    mathop = row.get("mathop", "").split(".")[-1]
    if mathop and mathop not in FIDELITY_SENSITIVE_MATHOPS and not bench.dim:
        return ""
    return row.get("math_fidelity", "").split(".")[-1]


# Knobs that belong to one lane, and must not be stamped onto the others.
#
# `Benchmark.keys` is declared per benchmark, so without this every lane of an
# SFPU sweep carries the math thread's knobs.  That is not merely redundant: the
# SFPU and FPU sweeps both measure `pack_tile` on the pack lane at the same
# config, and if one set of rows carries `approx_mode`/`iterations` and the other
# does not, a kernel matches BOTH -- two entries, two different numbers, which
# the lookup reports as a disagreement and answers with a placeholder.  A better
# measurement made an op unmeasured.
#
# The split is what the measurements say: across the SFPU sweeps these knobs move
# unpack and pack by 0.1-0.4% and math by up to 526%.  `unpack_to_dest` is absent
# here deliberately -- it is a real route change on unpack and on copy_tile's
# math, so it applies to every lane.
KNOB_LANES = {
    "approx_mode": {"math"},
    "iterations": {"math"},
}


def _effective_n(row: dict) -> float:
    """Tiles a per-tile figure was averaged over: `tile_cnt * loop_factor`."""
    return float(row.get("tile_cnt") or 0) * float(row.get("loop_factor") or 1)


def _knob_value(raw: str) -> str:
    """One knob's value, normalised.

    The CSVs spell the same idea several ways -- `unpack_to_dest` is True/False
    while `approx_mode` is Yes/No -- and enums arrive fully qualified
    (`MathOperation.ReduceRow`).  The consumer matches these as plain strings, so
    a caller would otherwise have to know which spelling each knob uses.  Booleans
    collapse to true/false and enums to their bare case name; anything else is
    passed through, which covers numbers like `iterations`.
    """
    value = raw.split(".")[-1]
    if value in ("True", "Yes"):
        return "true"
    if value in ("False", "No"):
        return "false"
    return value


def _key(row: dict, bench: Benchmark, op: str, lane: str) -> Key:
    return Key(
        op=op,
        lane=lane,
        in_format=row["formats.input_A"],
        out_format=row["formats.output"],
        dest_acc=row.get("dest_acc", "").endswith("Yes"),
        fidelity=_fidelity(row, bench, lane),
        dst_sync=row.get("dest_sync", "").split(".")[-1],
        # A benchmark that reports its face count per row is believed over the
        # per-benchmark constant, which is a fallback for the sources that
        # configure faces in the kernel and never emit a column.  Taking the
        # row's value is what lets one sweep carry both full and partial tiles
        # without the two blending.
        faces=int(row.get("num_faces") or bench.faces),
        variant=tuple(
            (k, _knob_value(row.get(k, "")))
            for k in bench.keys
            if k in row and lane in KNOB_LANES.get(k, {lane})
        ),
    )


def _owners(spec, row) -> Optional[list[str]]:
    return spec(row) if callable(spec) else spec


class Diagnostics:
    def __init__(self) -> None:
        self.unmapped: set[str] = set()
        self.lumped: dict[tuple[str, ...], int] = defaultdict(int)
        self.rows_read = 0
        self.rows_used = 0
        self.disagreements: list[str] = []
        self.dropped_lumps = 0
        self.emitted_rows = 0
        self.emitted_ops = 0
        self.steady: list[str] = []
        self.conflicted: set = set()
        self.dropped_conflicts = 0

    def report(self, out) -> None:
        print(f"rows read: {self.rows_read}, used: {self.rows_used}", file=out)
        if self.emitted_ops:
            print(
                f"emitted: {self.emitted_ops} ops, {self.emitted_rows} measured "
                f"rows, {self.dropped_lumps} lumped and "
                f"{self.dropped_conflicts} conflicted keys dropped",
                file=out,
            )
        if self.unmapped:
            print(
                f"unmapped mathops ({len(self.unmapped)}): "
                + ", ".join(sorted(self.unmapped)),
                file=out,
            )
        for msg in self.steady:
            print(f"steady state: {msg}", file=out)
        for ops, n in sorted(self.lumped.items()):
            print(f"lumped lane ({n} rows): {' + '.join(ops)}", file=out)
        for msg in self.disagreements:
            print(f"disagreement: {msg}", file=out)


def _steadiest(rows: list[dict], bench: Benchmark, name: str, diag: Diagnostics):
    """Keep, per configuration, only the rows at the largest `Benchmark.steady`.

    Grouped rather than filtered against a global maximum so a sweep that runs
    some configurations at fewer values keeps its best row for each instead of
    losing them all to a maximum they never reach.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            tuple(
                v
                for k, v in sorted(row.items())
                if k != bench.steady and not k.startswith(("mean(", "TEXT_SIZE("))
            )
        ].append(row)

    # Grouped on the declared column, but compared on effective N -- see
    # _effective_n.  A benchmark that sweeps tile_cnt at a fixed loop_factor gets
    # the same answer either way; one that varied both would not.
    kept, dropped, values = [], 0, set()
    for group in groups.values():
        best = max(_effective_n(r) for r in group)
        values.add(best)
        for row in group:
            if _effective_n(row) == best:
                kept.append(row)
            else:
                dropped += 1

    if dropped:
        seen = ", ".join(f"{v:g}" for v in sorted(values))
        diag.steady.append(
            f"{name}: kept N={seen}, dropped {dropped} lower row(s)"
        )
    return kept


def extract(diag: Diagnostics):
    """Read every attributed benchmark into {marker: {Key: Samples}}."""
    tables = {"TILE_LOOP": defaultdict(Samples), "INIT": defaultdict(Samples)}

    for name, bench in sorted(BENCHMARKS.items()):
        path = PERF_DATA / name / f"{name}.post.csv"
        if not path.exists():
            print(f"warning: {path} missing, skipping", file=sys.stderr)
            continue

        rows = list(csv.DictReader(path.open()))
        if bench.steady:
            rows = _steadiest(rows, bench, name, diag)

        for row in rows:
            marker = row["marker"]
            if marker not in tables:
                continue
            diag.rows_read += 1
            spec = bench.tile_loop if marker == "TILE_LOOP" else bench.init
            used = False

            for lane in LANES:
                raw = row.get(f"mean({lane.upper()}_ISOLATE)")
                if not raw:
                    continue
                owners = _owners(spec[lane], row)
                if not owners:
                    if row.get("mathop") and _mathop(row) is None:
                        diag.unmapped.add(row["mathop"].split(".")[-1])
                    continue

                if len(owners) == 1:
                    op = owners[0]
                else:
                    op = LUMP + "+".join(owners)
                    diag.lumped[tuple(owners)] += 1

                s = tables[marker][_key(row, bench, op, lane)]
                s.values.append(float(raw))
                s.sources.add(bench.source)
                s.tiles.append(_effective_n(row))
                if bench.dim:
                    s.dims.append(float(row[bench.dim]))
                used = True

            if used:
                diag.rows_used += 1

    # A key measured by two benchmarks must agree; if it does not, the two are
    # not measuring the same thing and the key cannot tell them apart.  This is
    # the check that caught pack_tile at 20.00 (2-face matmul) against 29.94
    # (4-face eltwise).
    #
    # Such a key is dropped, not averaged.  The mean of two numbers that disagree
    # by 83% is a measurement of nothing, and it would be reported as `meas` --
    # more confident than the placeholder it replaced and less correct.  The same
    # reasoning as the lumps: a number that cannot be attributed is not data.
    #
    # What this currently catches is not an attribution error but a provenance
    # one.  `perf_data/` mixes CSVs from the nightly runner with CSVs re-run
    # locally, and the two environments do not agree on INIT zones: running the
    # *same* benchmark both ways gives 212 vs 453 cycles for the FPU unpack init
    # and 257 vs 470 for the SFPU one, uniformly across every operation, while
    # their TILE_LOOP rates agree to within a few percent (42.7 vs 42.6 on unpack,
    # 29.3 vs 28.2 on pack).  A one-shot zone is sensitive to instruction-cache
    # and layout state that a warm loop averages away, so an init measurement is
    # only comparable against others taken in the same build.  Until every
    # benchmark is re-run in one environment, an init row's number belongs to its
    # runner as much as to the operation.
    for marker, table in tables.items():
        for s in table.values():
            if marker == "TILE_LOOP":
                s.keep_steadiest()

    for marker, table in tables.items():
        for key, s in table.items():
            if len(s.sources) > 1 and s.spread() > 0.05:
                diag.disagreements.append(
                    f"{marker} {key.op}/{key.lane} {key.in_format}->"
                    f"{key.out_format}: {s.spread():.0%} spread across "
                    + ", ".join(sorted(s.sources))
                )
                diag.conflicted.add((marker, key))
    return tables


def fit_affine(xs: list[float], ys: list[float]):
    """Least squares y = a*x + b, returning (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return (ys[0] / xs[0] if xs and xs[0] else 0.0), 0.0, 1.0
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return (sy / sx if sx else 0.0), 0.0, 1.0
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    resid = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - sy / n) ** 2 for y in ys)
    return a, b, (1 - resid / total if total else 1.0)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

HEADER = """// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Generated by scripts/gen_cost_table.py. Do not edit.
//
// Regenerate with:
//   llk-perf/run_sweep.sh
//   scripts/gen_cost_table.py -o lib/OpCost/CostTable{arch_camel}.inc
//
// Measured on {arch} by the sweeps in llk-perf/, which are narrowed to the
// configurations tt-lang can generate -- not the nightly tt-llk run, whose
// sweeps cover configurations no tt-lang kernel produces. Costs are not
// transferable across architectures.
//
// Source data: {rows} CSV rows from {sources} benchmark(s), against tt-metal
// {metal_sha}. The CSVs are not in the tree: they are large, regenerable from
// llk-perf/, and only meaningful against the revision that produced them. That
// revision is recorded here because it is the one thing about them that cannot
// be recovered later -- the same benchmark across two tt-llk revisions differs
// by up to 2x on init zones. Coverage of this table is pinned by
// test/ttlang/OpCost/coverage.mlir.
//
// Included into a scope that defines `MeasuredCost`, `EngineCost` and `OpCost`;
// this file is data only. See include/ttlang/OpCost/OpCost.h.
//
// Each entry is one lane of one benchmark's loop body, attributed to the
// ttkernel operation that owns that lane. `*_ISOLATE` measurements exclude the
// cross-thread credit handshakes, which the scheduler derives instead. The DST
// lifecycle ops (tile_regs_*) and the circular-buffer ops have no entries: no
// perf source isolates a handshake, and none of them touches a circular buffer,
// so they are modelled as pure synchronization with zero work.
//
// `op` "{lump}" marks a lane whose loop body ran more than one operation. Those
// are recorded rather than attributed, and are recoverable by subtracting a
// separately measured component.

"""


# Lane spec slots the CSVs can measure.  A benchmark builds three kernels behind
# `#ifdef LLK_TRISC_UNPACK` / `_MATH` / `_PACK`, so the data covers the three
# TRISCs and never the data-movement RISCs.
MEASURED_LANES = ("unpack", "math", "pack")

# Every lane slot of a table entry, in `CostEstimator::Lane` order as far as the
# slots go.  `dm` is one slot covering both data-movement RISCs; see lane_work.
SPEC_LANES = ("dm", "unpack", "math", "pack")

OP_DEF = re.compile(r'^def\s+TTKernel_\w+\s*:\s*TTKernel_\w+<\s*"([^"]+)"', re.M)


def metal_revision() -> str:
    """tt-metal revision the perf CSVs were produced against.

    Recorded in the generated table because the CSVs themselves are not in the
    tree, and a measurement is only comparable with others from the same
    revision: running the same benchmark across two of them moved the FPU unpack
    init from 212 cycles to 453.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO / "third-party/tt-metal"), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def dialect_ops() -> set[str]:
    """Every operation mnemonic the TTKernel dialect defines.

    Read from the ODS rather than taken on trust, so the generated table cannot
    silently fall behind the dialect: an op added without a lane spec fails
    generation instead of surfacing later as an unexplained estimate failure on
    somebody's kernel.
    """
    return set(OP_DEF.findall(TTKERNEL_OPS_TD.read_text()))


def _sorted(table):
    return sorted(table, key=lambda k: (k.op, k.lane, k.in_format, k.out_format))


def _measured_row(key: Key, unit: str, cost: float, fixed: float) -> str:
    """One measured row.

    Carries neither `op` nor `lane`: a row is reached only through the lane slot
    that slices it, so both are implied by where it sits.
    """
    return (
        f'    {{"{key.in_format}", "{key.out_format}", '
        f"{'true' if key.dest_acc else 'false'}, "
        f'"{key.fidelity}", "{key.dst_sync}", {key.faces}, '
        f'"{";".join(f"{k}={v}" for k, v in key.variant)}", '
        f"Unit::{unit}, {cost:.2f}, {fixed:.2f}}},"
    )


def _lane_slot(slot, span) -> str:
    """One lane of a table entry: `{}` when the op does not run there."""
    if slot is None:
        assert span is None, "measured rows for a lane the spec says is unused"
        return "{}"
    cost = slot if isinstance(slot, Cost) else Cost(slot, PER_CALL)
    first, count = span or (0, 0)
    return f"EngineCost{{{cost.cycles}, Unit::{cost.unit}, {first}, {count}}}"


def collect_measured(tables, diag):
    """Measured rows grouped by (op, lane), lumps dropped.

    A lumped lane covers two operations in one number, so it belongs to neither
    and can never answer a lookup keyed on one op.  Dropping it here is what
    keeps ~1000 unreachable rows out of the compiled table; the diagnostics still
    name every one, and they stay recoverable from the CSVs by subtracting a
    separately measured component.
    """
    grouped = defaultdict(list)
    for marker, unit in (("TILE_LOOP", PER_TILE), ("INIT", PER_CALL)):
        for key in _sorted(tables[marker]):
            if key.op.startswith(LUMP):
                diag.dropped_lumps += 1
                continue
            if (marker, key) in diag.conflicted:
                diag.dropped_conflicts += 1
                continue
            s = tables[marker][key]
            if s.dims:
                cost, fixed, r2 = fit_affine(*s.by_dim())
                note = f"r2={r2:.4f}, fitted vs block dim"
            else:
                cost, fixed = s.mean(), 0.0
                note = f"n={len(s.values)}"
                if s.spread() > 0.02:
                    note += f", spread={s.spread():.1%}"
            grouped[(key.op, key.lane)].append((key, unit, cost, fixed, note))
    return grouped


def check_coverage(grouped) -> None:
    """Refuse to emit a table that disagrees with the dialect or with itself."""
    ops = dialect_ops()
    missing = sorted(ops - set(LANE_WORK))
    if missing:
        raise SystemExit(
            f"lane_work.py covers no lanes for {len(missing)} ttkernel op(s), so "
            f"the table would be incomplete: {', '.join(missing)}"
        )
    stale = sorted(set(LANE_WORK) - ops)
    if stale:
        raise SystemExit(
            f"lane_work.py names {len(stale)} op(s) the dialect does not define: "
            f"{', '.join(stale)}"
        )
    for op, lane in sorted(grouped):
        if op not in LANE_WORK:
            raise SystemExit(f"measured rows for unknown op {op}")
        if getattr(LANE_WORK[op], lane) is None:
            raise SystemExit(
                f"{op} has measured {lane} rows but lane_work.py says it does "
                f"not run on that lane; one of the two is wrong"
            )


def emit(tables, diag, out) -> None:
    grouped = collect_measured(tables, diag)
    check_coverage(grouped)

    out.write(
        HEADER.format(
            arch=ARCH,
            arch_camel=ARCH.capitalize(),
            lump=LUMP,
            rows=diag.rows_read,
            sources=len(BENCHMARKS),
            metal_sha=metal_revision(),
        )
    )

    out.write(
        "// Measured rows, grouped so each (op, lane) run is contiguous. Reached\n"
        "// only through the lane slot that slices it.\n"
        "static const MeasuredCost kMeasured[] = {\n"
    )
    spans, at = {}, 0
    for op_lane in sorted(grouped):
        rows = grouped[op_lane]
        spans[op_lane] = (at, len(rows))
        at += len(rows)
        for key, unit, cost, fixed, note in rows:
            out.write(
                _measured_row(key, unit, cost, fixed)
                + f"  // {key.op}/{key.lane}, {note}\n"
            )
    out.write("};\n\n")

    out.write(
        "// The cost table. One entry per ttkernel operation; a lane with no\n"
        "// value is a lane the operation does not run on, and an entry with no\n"
        "// lane at all runs nowhere.\n"
        "static const OpCost kCostTable[] = {\n"
    )
    for op in sorted(dialect_ops()):
        spec = LANE_WORK[op]
        slots = [
            _lane_slot(getattr(spec, lane), spans.get((op, lane)))
            for lane in SPEC_LANES
        ]
        out.write(f'    {{"{op}", ' + ", ".join(slots) + "},\n")
    out.write("};\n")
    diag.emitted_rows = at
    diag.emitted_ops = len(LANE_WORK)


def report(tables, diag, out) -> None:
    diag.report(out)
    for marker in ("TILE_LOOP", "INIT"):
        table = tables[marker]
        lanes_by_op = defaultdict(set)
        for key in table:
            lanes_by_op[key.op].add(key.lane)
        print(f"\n{marker}: {len(table)} entries, {len(lanes_by_op)} ops", file=out)
        for op in sorted(lanes_by_op):
            lanes = ",".join(l for l in LANES if l in lanes_by_op[op])
            n = sum(1 for k in table if k.op == op)
            print(f"    {op:24s} {lanes:20s} {n:5d} entries", file=out)
    print("\ncaveats on attributed benchmarks:", file=out)
    for name, bench in sorted(BENCHMARKS.items()):
        if bench.note:
            print(f"    {name}: {bench.note}", file=out)
    print("\nunattributed benchmarks:", file=out)
    for name, why in sorted(UNATTRIBUTED.items()):
        print(f"    {name}: {why}", file=out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", type=pathlib.Path)
    p.add_argument("--report", action="store_true", help="print coverage instead")
    args = p.parse_args()

    diag = Diagnostics()
    tables = extract(diag)

    if args.report:
        report(tables, diag, sys.stdout)
        return 0

    if args.output:
        with args.output.open("w") as f:
            emit(tables, diag, f)
    else:
        emit(tables, diag, sys.stdout)
    diag.report(sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
