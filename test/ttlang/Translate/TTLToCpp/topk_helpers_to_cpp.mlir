// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttkernel-insert-inits %s -o %t.ttkernel.mlir
// RUN: FileCheck %s --input-file=%t.ttkernel.mlir --check-prefix=TTKERNEL
// RUN: ttlang-opt --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Summary: the TopK mode attributes alone select the slab helpers. The stage
// ops carry no helper calls, and TTKernelInsertInits emits topk_tile_init, the
// prepare helper, and the matching unpack helper around the stages of a block.

// Fused mode surrounds the stage with topk_fuse_tile and topk_defuse_tile.
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
  ttl.tile_regs_acquire
  ttl.tile_topk_local_sort dst[%dst] direction = %direction
      end_phase = %end_phase start_phase = %start_phase
      {fused = true, largest = false}
      : (index, i32, i32, i32) -> ()
  ttl.tile_regs_release
  return
}

// -----

// Rank-stamped mode stamps once before the first stage and strips once after
// the last, and both helpers inherit tag_bits from the stages.
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
  ttl.tile_topk_local_sort dst[%dst] direction = %direction
      end_phase = %end_phase start_phase = %start_phase
      {rank_stamped = true, tag_bits = 8 : i32}
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iteration k = %k
      {rank_stamped = true, tag_bits = 8 : i32}
      : (index, i32, i32) -> ()
  ttl.tile_regs_release
  return
}

// -----

// Stable mode only needs the negative-zero fold before the sort; nothing is
// packed, so no unpack helper follows the stage.
// TTKERNEL-LABEL: func.func @topk_stable_mode
// TTKERNEL: ttkernel.topk_tile_init()
// TTKERNEL-NEXT: ttkernel.topk_canonicalize_negzero_values
// TTKERNEL-NEXT: ttkernel.topk_local_sort
// TTKERNEL-NEXT: ttkernel.tile_regs_release
// TTKERNEL-NOT: ttkernel.topk_defuse_tile
// TTKERNEL-NOT: ttkernel.topk_strip_rank_tags

// CPP: topk_tile_init();
// CPP-NEXT: topk_canonicalize_negzero_values(
// CPP-NEXT: topk_local_sort<true, DST_ACCUM_MODE, false, false, TopkTieOrder::Ascending>(
func.func @topk_stable_mode() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %direction = arith.constant 0 : i32
  %end_phase = arith.constant 4 : i32
  %start_phase = arith.constant 0 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_local_sort dst[%dst] direction = %direction
      end_phase = %end_phase start_phase = %start_phase
      {stable_sort = true, tie_order = #ttl.topk_tie_order<ascending>}
      : (index, i32, i32, i32) -> ()
  ttl.tile_regs_release
  return
}
