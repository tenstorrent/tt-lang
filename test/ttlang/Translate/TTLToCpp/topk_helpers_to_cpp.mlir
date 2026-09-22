// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttkernel-insert-inits %s -o %t.ttkernel.mlir
// RUN: FileCheck %s --input-file=%t.ttkernel.mlir --check-prefix=TTKERNEL
// RUN: ttlang-opt --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Summary: explicit TopK helper ops lower to the metal calls. Insert-inits
// emits topk_tile_init and does not add a second copy of the helpers. The
// global order sets helper polarity.

// Fused mode lowers the explicit fuse and defuse around the stage.
// TTKERNEL-LABEL: func.func @topk_fused_mode
// TTKERNEL: ttkernel.tile_regs_acquire
// TTKERNEL-NEXT: ttkernel.topk_tile_init
// TTKERNEL-SAME: fused = true
// TTKERNEL-NEXT: ttkernel.topk_fuse_tile
// TTKERNEL-SAME: largest = false
// TTKERNEL-NEXT: ttkernel.topk_local_sort
// TTKERNEL: ttkernel.topk_defuse_tile
// TTKERNEL-SAME: largest = false
// TTKERNEL-NEXT: ttkernel.tile_regs_release

// CPP: #include "api/compute/topk.h"
// CPP-LABEL: void kernel_main()
// CPP: tile_regs_acquire();
// CPP-NEXT: topk_tile_init<true>();
// CPP-NEXT: topk_fuse_tile<false>(
// CPP-NEXT: topk_local_sort<false, DST_ACCUM_MODE, true>(
// CPP: topk_defuse_tile<false>(
// CPP-NEXT: tile_regs_release();
func.func @topk_fused_mode() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %direction = arith.constant 0 : i32
  %end_phase = arith.constant 4 : i32
  %start_phase = arith.constant 0 : i32
  %num_tiles = arith.constant 1 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_fuse dst[%dst] {order = #ttl.topk_order<ascending>} : index
  ttl.tile_topk_local_sort dst[%dst] direction = %direction
      end_phase = %end_phase start_phase = %start_phase
      {fused = true, order = #ttl.topk_order<ascending>}
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_defuse dst[%dst] num_tiles = %num_tiles
      {order = #ttl.topk_order<ascending>} : (index, i32) -> ()
  ttl.tile_regs_release
  return
}

// -----

// Rank-stamped mode lowers the explicit stamp and strip. Both helpers carry
// tag_bits from the stages.
// TTKERNEL-LABEL: func.func @topk_rank_stamped_mode
// TTKERNEL: ttkernel.topk_tile_init
// TTKERNEL-SAME: rank_stamped = true
// TTKERNEL-SAME: tag_bits = 8
// TTKERNEL-NEXT: ttkernel.topk_stamp_local_positions
// TTKERNEL-SAME: tag_bits = 8
// TTKERNEL-NEXT: ttkernel.topk_local_sort
// TTKERNEL-NEXT: ttkernel.topk_merge
// TTKERNEL-NEXT: ttkernel.topk_strip_rank_tags
// TTKERNEL-SAME: tag_bits = 8

// CPP: topk_tile_init<false, true, 8>();
// CPP-NEXT: topk_stamp_local_positions<true, 8>(
// CPP-NEXT: topk_local_sort<false, DST_ACCUM_MODE, false, true>(
// CPP-NEXT: topk_merge<false, false, DST_ACCUM_MODE, false, true, TopkTieOrder::Unset, 8>(
// CPP-NEXT: topk_strip_rank_tags<8>(
func.func @topk_rank_stamped_mode() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %direction = arith.constant 0 : i32
  %end_phase = arith.constant 4 : i32
  %start_phase = arith.constant 0 : i32
  %iteration = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_stamp_local_positions dst[%dst]
      {order = #ttl.topk_order<descending>, tag_bits = 8 : i32} : index
  ttl.tile_topk_local_sort dst[%dst] direction = %direction
      end_phase = %end_phase start_phase = %start_phase
      {order = #ttl.topk_order<descending>,
       rank_stamped = true, tag_bits = 8 : i32}
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iteration k = %k
      {order = #ttl.topk_order<descending>,
       rank_stamped = true, tag_bits = 8 : i32}
      : (index, i32, i32) -> ()
  ttl.tile_topk_strip_rank_tags dst[%dst] {tag_bits = 8 : i32} : index
  ttl.tile_regs_release
  return
}

// -----

// Stable mode lowers the explicit negative-zero fold. Nothing is packed, so
// no unpack helper follows the stage. A descending order selects the
// descending tie order.
// TTKERNEL-LABEL: func.func @topk_stable_mode
// TTKERNEL: ttkernel.topk_tile_init()
// TTKERNEL-NEXT: ttkernel.topk_canonicalize_negzero_values
// TTKERNEL-NEXT: ttkernel.topk_local_sort
// TTKERNEL-SAME: {stable_sort = true, tie_order = #ttkernel.topk_tie_order<descending>}
// TTKERNEL-NEXT: ttkernel.tile_regs_release
// TTKERNEL-NOT: ttkernel.topk_defuse_tile
// TTKERNEL-NOT: ttkernel.topk_strip_rank_tags

// CPP: topk_tile_init();
// CPP-NEXT: topk_canonicalize_negzero_values(
// CPP-NEXT: topk_local_sort<true, DST_ACCUM_MODE, false, false, TopkTieOrder::Descending>(
func.func @topk_stable_mode() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %direction = arith.constant 0 : i32
  %end_phase = arith.constant 4 : i32
  %start_phase = arith.constant 0 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_canonicalize_negzero_values dst[%dst] : index
  ttl.tile_topk_local_sort dst[%dst] direction = %direction
      end_phase = %end_phase start_phase = %start_phase
      {order = #ttl.topk_order<descending>, stable_sort = true}
      : (index, i32, i32, i32) -> ()
  ttl.tile_regs_release
  return
}
