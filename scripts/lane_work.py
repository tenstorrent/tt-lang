#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-lane work for every TTKernel operation.

One of the two inputs to ``gen_cost_table.py``; the other is the LLK perf data.
This half answers *which lanes an operation runs on*, which is a compile-time
fact rather than a measurement: a compute kernel is one source file compiled
three times with ``-DTRISC_UNPACK`` / ``-DTRISC_MATH`` / ``-DTRISC_PACK``, and
the ``UNPACK()`` / ``MATH()`` / ``PACK()`` macros in
``api/compute/common_globals.h`` erase every call not belonging to the thread
being compiled.  So the wrapper around a call decides which TRISC runs it, and
that is read from the headers, never measured.

``dm`` is one slot rather than one per data-movement RISC because an operation
does not choose which core it runs on -- ``ttl.noc_index`` on the enclosing
function does -- so NCRISC and BRISC can never differ here.  If one is ever
measured to differ from the other, that belongs in a measured row, which is
already keyed by lane.

The cycle counts here are the other half of each slot, and they are
**PLACEHOLDERS, NOT MEASUREMENTS**.  They are ordered sensibly against each
other -- a semaphore poll is cheaper than a tile of unpack, which is cheaper
than a tile of FPU work -- and nothing else.  A placeholder is only ever used
when the perf data has nothing for the operation's configuration; where a
measurement exists it wins, and the report says per placement which it used.

An operation absent from ``LANE_WORK`` is a generation error, not an unknown:
``gen_cost_table.py`` reads the op list out of ``TTKernelOps.td`` and refuses to
emit a table that does not cover the dialect.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Union

PER_CALL = "PerCall"
PER_TILE = "PerTile"


@dataclasses.dataclass(frozen=True)
class Cost:
    cycles: int
    unit: str = PER_CALL


def per_tile(cycles: int) -> Cost:
    """A cost charged once per tile the operation processes.

    Only meaningful for operations taking a tile count from an operand;
    ``pack_tile_block`` packs ``ntiles`` tiles in a loop, so a flat cost is
    wrong for any block bigger than one tile.
    """
    return Cost(cycles, PER_TILE)


Slot = Optional[Union[int, Cost]]


@dataclasses.dataclass(frozen=True)
class Lanes:
    dm: Slot = None
    unpack: Slot = None
    math: Slot = None
    pack: Slot = None


def lanes(dm: Slot = None, unpack: Slot = None, math: Slot = None,
          pack: Slot = None) -> Lanes:
    """Work on each lane; ``None`` means the op does not run there."""
    return Lanes(dm, unpack, math, pack)


def sfpu(cost: Slot = 300) -> Lanes:
    """An SFPU tile operation: MATH only.

    Every SFPU call in the compute API sits under ``MATH()``; the ones going
    through ``SFPU_BINARY_CALL`` / ``SFPU_BINARY_INIT_FN``
    (``llk_math_eltwise_binary_sfpu_macros.h:49,:81``) expand under it too, so
    the lane is unambiguous either way.
    """
    return Lanes(math=cost)


def sfpu_init(cost: Slot = 40) -> Lanes:
    """The short init for an SFPU operation: MATH only, same reasoning."""
    return Lanes(math=cost)


def fpu(unpack: Slot = 200, math: Slot = 300) -> Lanes:
    """An FPU tile operation: UNPACK fills SrcA/SrcB, MATH drains it.

    The packer work of such a kernel belongs to ``pack_tile``, not here -- which
    is why no FPU op carries a pack slot.  ``TTKernelFPUOpTrait`` encodes the
    same fact for the Src credit model, and the estimator asserts the two agree.
    """
    return Lanes(unpack=unpack, math=math)


def fpu_init(cost: Slot = 40) -> Lanes:
    """The short init for an FPU operation: UNPACK and MATH, no handshake."""
    return Lanes(unpack=cost, math=cost)


FREE = Lanes()
"""Runs on no lane.

Distinct from being absent, which is a generation error.  These are the
operations lowering to an expression, a variable or a compile-time constant
rather than to a call: address arithmetic, handle construction, casts.
"""


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
#
# Grouped by the header the lane membership was read from.  A group sharing one
# justification is written against a shared constructor, so the justification is
# stated once and every operation covered by it stays visible.

LANE_WORK: dict[str, Lanes] = {}


def _add(spec: Lanes, *ops: str) -> None:
    for op in ops:
        assert op not in LANE_WORK, f"duplicate lane spec for {op}"
        LANE_WORK[op] = spec


# -- Circular buffers, api/dataflow/circular_buffer.h:31-69 -----------------
# Under COMPILE_FOR_TRISC the four methods are wrapped PACK/PACK/UNPACK/UNPACK;
# otherwise they call the plain dataflow functions.
#
# These and the DST lifecycle below are unmeasured: no benchmark in the LLK perf
# suite isolates a handshake, and none of them touches a circular buffer.  What
# matters about them is not the call anyway, it is the credit they move -- the
# waiting is derived by the scheduler -- so they carry the unmeasured value of
# one and let their resource effect do the work.
_add(lanes(dm=1, unpack=1), "cb_wait_front", "cb_pop_front")
_add(lanes(dm=1, pack=1), "cb_reserve_back", "cb_push_back")

# -- DST lifecycle, api/compute/reg_api.h:45-89 -----------------------------
_add(lanes(math=1), "tile_regs_acquire", "tile_regs_commit")
_add(lanes(pack=1), "tile_regs_wait", "tile_regs_release")

# -- Common inits, api/compute/{eltwise_binary,eltwise_unary,matmul}.h ------
# The one-per-kernel inits configuring all three engines.  Their MATH half calls
# llk_math_pack_sync_init, which is what makes them a DST pipeline reset.
_add(lanes(unpack=100, math=100, pack=140), "binary_op_init_common")
_add(lanes(unpack=100, math=140, pack=140), "unary_op_init_common", "init_sfpu",
     "mm_init", "mm_block_init")
_add(lanes(unpack=60, math=100, pack=140), "compute_kernel_hw_startup")

# -- FPU tile ops, api/compute/{eltwise_binary,matmul,reduce,bcast}.h -------
_add(fpu(), "add_tiles", "sub_tiles", "mul_tiles", "binary_dest_reuse_tiles",
     "reduce_tile", "unary_bcast")
_add(fpu(math=400), "matmul_tiles", "matmul_block", "experimental.matmul_block")
_add(fpu_init(), "add_tiles_init", "sub_tiles_init", "mul_tiles_init",
     "binary_dest_reuse_tiles_init", "mm_init_short", "mm_block_init_short",
     "unary_bcast_init")
_add(lanes(unpack=40, math=40, pack=200), "reduce_init")
_add(lanes(math=300, pack=200), "reduce_uninit")

# -- Datacopy, api/compute/tile_move_copy.h ---------------------------------
# Not FPU ops, but they feed SrcA/SrcB exactly the same way, which is what
# getResourceEffect's non-FPU Src coupling list records.
_add(fpu(math=150), "copy_tile", "copy_block_matmul_partials")
_add(fpu(math=300), "transpose_wh_tile")
_add(fpu_init(), "copy_tile_init", "transpose_wh_init")

# -- Pack, api/compute/pack.h and pack_untilize.h ---------------------------
# pack_tile (:86-94) packs one tile; pack_tile_block (:128-135) is the same
# packer work hoisted out of a loop over ntiles, so its cost is per tile.  Both
# forms occur: ttkernel-combine-pack-tiles fuses a run of pack_tile into one
# pack_tile_block, but only when the CB indices step by one from zero, so a
# subblocked compute keeps separate pack_tile ops after the first round.
_add(lanes(pack=200), "pack_tile", "pack_reconfig_l1_acc",
     "pack_reconfig_data_format")
# Flat, not per_tile, until something consumes the unit.  pack_tile_block packs
# `ntiles` tiles in a loop, so a per-call cost is wrong for any block bigger than
# one tile -- but the estimator charges every cost once per call, so declaring
# this per-tile today would charge one tile's worth for the whole block, which is
# wrong in the other direction and by more.  See the scaling TODO.
_add(lanes(pack=260), "pack_tile_block")
_add(lanes(unpack=40, math=40), "pack_untilize_init")
_add(lanes(pack=480), "pack_untilize_uninit")

# -- Tilize / untilize, api/compute/{tilize,untilize}.h ---------------------
# tilize_block and untilize_block run the DST handshake themselves: their MATH
# halves call llk_math_wait_for_dest_available and llk_math_dest_section_done,
# and their PACK halves the packer's matching pair.  getResourceEffect keys the
# DST effects on the tile_regs_* ops only, so those internal acquires are not
# modelled -- a pre-existing gap this comment carries.
_add(lanes(unpack=200, math=190, pack=240), "tilize_block",
     "experimental.tilize_block", "experimental.pack_untilize_block")
_add(lanes(unpack=200, math=40, pack=240), "untilize_block",
     "experimental.untilize_block")
_add(lanes(unpack=200, pack=40), "tilize_uninit", "untilize_uninit")
_add(fpu_init(), "tilize_init", "untilize_init")

# -- SFPU tile ops, api/compute/**.h ----------------------------------------
_add(sfpu(350), "where_tile")
_add(sfpu(), *"""
abs_tile abs_tile_int32 acos_tile add_binary_tile add_int_tile add_unary_tile
add_unary_tile_int32 asin_tile atan2_binary_tile atan_tile binary_left_shift_tile
binary_logical_right_shift_tile binary_max_int32_tile binary_max_tile
binary_min_int32_tile binary_min_tile binary_right_shift_tile
bitwise_and_binary_tile bitwise_not_tile bitwise_or_binary_tile
bitwise_xor_binary_tile ceil_tile clamp_tile clamp_tile_int32 copy_dest_values
cos_tile div_binary_tile div_unary_tile eq_binary_tile eqz_tile eqz_tile_int32
erf_tile erfc_tile exp2_tile exp_tile expm1_tile fill_tile fill_tile_int
floor_tile frac_tile ge_binary_tile gelu_tile gez_tile gez_tile_int32
gt_binary_tile gtz_tile gtz_tile_int32 hardsigmoid_tile invoke_sfpi
le_binary_tile lez_tile lez_tile_int32 log1p_tile log_tile logical_not_tile
lt_binary_tile ltz_tile ltz_tile_int32 mul_binary_tile mul_int_tile
mul_unary_tile ne_binary_tile negative_tile negative_tile_int32 nez_tile
nez_tile_int32 power_binary_tile power_tile rand_tile recip_tile relu_tile
relu_tile_int32 rsqrt_tile selu_tile sfpu_reduce sigmoid_tile sign_tile
signbit_tile silu_tile sin_tile softsign_tile sqrt_tile square_tile
sub_binary_tile sub_int_tile sub_unary_tile sub_unary_tile_int32 tan_tile
tanh_tile topk_local_sort topk_merge topk_rebuild trunc_tile typecast_tile
""".split())

# -- SFPU inits ------------------------------------------------------------
_add(sfpu_init(), *"""
abs_tile_init acos_tile_init add_binary_tile_init add_int_tile_init
asin_tile_init atan2_binary_tile_init atan_tile_init binary_bitwise_tile_init
binary_max_int32_tile_init binary_max_tile_init binary_min_int32_tile_init
binary_min_tile_init binary_shift_tile_init binop_with_scalar_tile_init
bitwise_not_tile_init clamp_tile_init copy_dest_values_init cos_tile_init
div_binary_tile_init eq_binary_tile_init eqz_tile_init erf_tile_init
erfc_tile_init exp2_tile_init exp_tile_init expm1_tile_init fill_tile_init
ge_binary_tile_init gelu_tile_init gez_tile_init gt_binary_tile_init
gtz_tile_init hardsigmoid_tile_init le_binary_tile_init lez_tile_init
log1p_tile_init log_tile_init logical_not_tile_init lt_binary_tile_init
ltz_tile_init mul_binary_tile_init mul_int_tile_init ne_binary_tile_init
negative_tile_init nez_tile_init power_binary_tile_init power_tile_init
rand_tile_init recip_tile_init relu_tile_init rounding_op_tile_init
rsqrt_tile_init selu_tile_init sfpu_reduce_init sigmoid_tile_init
sign_tile_init signbit_tile_init silu_tile_init sin_tile_init
softsign_tile_init sqrt_tile_init square_tile_init sub_binary_tile_init
sub_int_tile_init tan_tile_init tanh_tile_init topk_tile_init
typecast_tile_init where_tile_init
""".split())

# -- Tile writes driven by the SFPU, api/compute/experimental --------------
_add(sfpu(), "experimental.fill_arange_tile", "experimental.write_col_mask_tile",
     "experimental.write_row_mask_tile")

# -- NoC transfers, api/dataflow/noc.h -------------------------------------
# A barrier's cost is the transfer it waits on, not the call itself.
_add(lanes(dm=60), "noc_async_read", "noc_async_read_tile",
     "noc_async_read_one_packet_with_state", "noc_async_write",
     "noc_async_write_tile", "noc_async_write_multicast",
     "noc_async_write_multicast_loopback_src",
     "noc_async_write_multicast_one_packet",
     "noc_async_write_one_packet_with_state",
     "noc_async_write_one_packet_with_trid")
_add(lanes(dm=30), "noc_async_read_barrier", "noc_async_read_barrier_with_trid",
     "noc_async_write_barrier", "noc_async_write_barrier_with_trid",
     "noc_async_atomic_barrier")
_add(lanes(dm=20), "noc_inline_dw_write", "remote_sram_write_u32",
     "noc_semaphore_inc", "noc_semaphore_inc_multicast",
     "noc_semaphore_set_multicast", "noc_semaphore_set_multicast_loopback_src")
_add(lanes(dm=10), "noc_semaphore_set", "noc_async_read_one_packet_set_state",
     "noc_async_write_one_packet_set_state", "experimental.semaphore_wait",
     "experimental.semaphore_wait_min", "load_from_l1", "store_to_l1")

# -- Fabric, api/dataflow/experimental --------------------------------------
_add(lanes(dm=100), "experimental.setup_fabric_connections",
     "experimental.create_fabric_connection_manager",
     "experimental.close_fabric_connections")
_add(lanes(dm=60), "experimental.fabric_fast_write_any_len",
     "experimental.fabric_mcast_fast_write_any_len")
_add(lanes(dm=20), "experimental.fabric_sem_inc",
     "experimental.fabric_mcast_sem_inc")

# -- Pointers and arguments, api/dataflow/{dataflow_api,circular_buffer}.h --
_add(lanes(dm=20), "TensorAccessor")
_add(lanes(dm=10), "get_arg_val", "get_common_arg_val")
_add(lanes(dm=5), "get_read_ptr", "get_write_ptr", "get_noc_addr",
     "get_noc_addr_from_bank_id", "get_noc_multicast_addr",
     "tensor_accessor.get_bank_and_offset", "tensor_accessor.get_noc_addr",
     "tensor_accessor.get_shard_noc_addr", "tensor_accessor.is_local_addr",
     "tensor_accessor.is_local_bank", "tensor_accessor.is_local_page",
     "tensor_accessor.is_local_shard")

# -- UNPACK-side synchronization -------------------------------------------
_add(lanes(unpack=10), "experimental.unpack_stall_on_pack")

# -- Known to be free -------------------------------------------------------
# Lower to an expression, a variable or a compile-time constant rather than to a
# call: compile-time args, casts, coordinate and identity registers, and the
# handle-construction ops.  Free is how the table says "costs nothing", as
# opposed to being absent, which the generator rejects.
_add(FREE, *"""
TensorAccessorArgs get_compile_time_arg_val get_dataformat get_dfb_id
get_semaphore get_tile_size mem_zeros_base mem_zeros_size my_logical_x_
my_logical_y_ my_x my_y bitcast reinterpret_cast cast_to_l1_addr unreachable
bfloat16_greater float32_greater experimental.constant_table_lookup
experimental.convert_logical_x_to_translated
experimental.convert_logical_y_to_translated
experimental.get_device_id_from_logical_mesh_position
experimental.get_my_device_id experimental.get_my_logical_mesh_position
""".split())

# -- Uncostable escape hatches ---------------------------------------------
# `opaque_call` calls a user-supplied C/C++ function and `dprint` formats into a
# host-drained buffer.  Neither has a cost anyone can look up, and both can
# appear in any kernel, so they carry a placeholder on every lane rather than
# being silently free.  A kernel leaning on either is one whose estimate should
# not be trusted, which the report's placeholder count already says.
_add(lanes(dm=50, unpack=50, math=50, pack=50), "opaque_call")
_add(lanes(dm=500, unpack=500, math=500, pack=500), "dprint")
