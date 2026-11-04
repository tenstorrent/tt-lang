// RUN: ttlang-opt %s --ttkernel-insert-inits --split-input-file | FileCheck %s

// Summary: topk_tile_init follows the stage mode. Insert-inits does not invent
// fuse, defuse, stamp, strip, or canonicalize; those belong to the packed-key
// buffer lifetime.

// CHECK-LABEL: func.func @topk_two_destinations
// CHECK: ttkernel.topk_tile_init
// CHECK-SAME: fused = true
// CHECK-NEXT: ttkernel.topk_local_sort
// CHECK-NOT: ttkernel.topk_fuse_tile
// CHECK-NOT: ttkernel.topk_defuse_tile
// CHECK: ttkernel.tile_regs_release
func.func @topk_two_destinations() {
  %dst0 = arith.constant 0 : index
  %dst4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.topk_local_sort(%dst0, %c0_i32, %c4_i32, %c0_i32) {fused = true}
      : (index, i32, i32, i32) -> ()
  ttkernel.topk_local_sort(%dst4, %c0_i32, %c4_i32, %c0_i32) {fused = true}
      : (index, i32, i32, i32) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// CHECK-LABEL: func.func @topk_same_constant_destination
// CHECK: ttkernel.topk_tile_init
// CHECK-SAME: fused = true
// CHECK: ttkernel.topk_local_sort
// CHECK: ttkernel.topk_merge
// CHECK-NOT: ttkernel.topk_fuse_tile
// CHECK-NOT: ttkernel.topk_defuse_tile
// CHECK: ttkernel.tile_regs_release
func.func @topk_same_constant_destination() {
  %dst0 = arith.constant 0 : index
  %dst0_again = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.topk_local_sort(%dst0, %c0_i32, %c4_i32, %c0_i32) {fused = true}
      : (index, i32, i32, i32) -> ()
  ttkernel.topk_merge(%dst0_again, %c0_i32, %c32_i32) {fused = true}
      : (index, i32, i32) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// CHECK-LABEL: func.func @topk_nested_single_block
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.topk_tile_init
// CHECK-SAME: fused = true
// CHECK: scf.for
// CHECK: ttkernel.topk_local_sort
// CHECK: ttkernel.topk_merge
// CHECK-NOT: ttkernel.topk_fuse_tile
// CHECK-NOT: ttkernel.topk_defuse_tile
// CHECK: ttkernel.tile_regs_release
func.func @topk_nested_single_block() {
  %dst = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  ttkernel.tile_regs_acquire() : () -> ()
  scf.for %i = %lb to %ub step %step {
    ttkernel.topk_local_sort(%dst, %c0_i32, %c4_i32, %c0_i32) {fused = true}
        : (index, i32, i32, i32) -> ()
    ttkernel.topk_merge(%dst, %c0_i32, %c32_i32) {fused = true}
        : (index, i32, i32) -> ()
  }
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Fuse and a fused stage share one init. The helper is not inserted again.
// CHECK-LABEL: func.func @fuse_shares_fused_init
// CHECK: ttkernel.topk_tile_init
// CHECK-SAME: fused = true
// CHECK-NEXT: ttkernel.topk_fuse_tile
// CHECK-NEXT: ttkernel.topk_local_sort
// CHECK-NOT: ttkernel.topk_tile_init
func.func @fuse_shares_fused_init() {
  %dst = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.topk_fuse_tile(%dst) : (index) -> ()
  ttkernel.topk_local_sort(%dst, %c0_i32, %c4_i32, %c0_i32) {fused = true}
      : (index, i32, i32, i32) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Defuse selects the fused init on its own.
// CHECK-LABEL: func.func @defuse_selects_fused_init
// CHECK: ttkernel.topk_tile_init
// CHECK-SAME: fused = true
// CHECK-NEXT: ttkernel.topk_defuse_tile
func.func @defuse_selects_fused_init() {
  %dst = arith.constant 0 : index
  %num_tiles = arith.constant 1 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.topk_defuse_tile(%dst, %num_tiles) : (index, i32) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Stamp and strip with the same tag width share one rank-stamped init.
// CHECK-LABEL: func.func @stamp_and_strip_share_tag_bits
// CHECK: ttkernel.topk_tile_init
// CHECK-SAME: rank_stamped = true
// CHECK-SAME: tag_bits = 8 : i32
// CHECK-NEXT: ttkernel.topk_stamp_local_positions
// CHECK-NEXT: ttkernel.topk_strip_rank_tags
// CHECK-NOT: ttkernel.topk_tile_init
func.func @stamp_and_strip_share_tag_bits() {
  %dst = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.topk_stamp_local_positions(%dst) {tag_bits = 8 : i32} : (index) -> ()
  ttkernel.topk_strip_rank_tags(%dst) {tag_bits = 8 : i32} : (index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// A different tag width replaces the init.
// CHECK-LABEL: func.func @tag_bits_change_replaces_init
// CHECK: ttkernel.topk_tile_init
// CHECK-SAME: tag_bits = 8 : i32
// CHECK: ttkernel.topk_stamp_local_positions
// CHECK: ttkernel.topk_tile_init
// CHECK-SAME: tag_bits = 6 : i32
// CHECK-NEXT: ttkernel.topk_strip_rank_tags
func.func @tag_bits_change_replaces_init() {
  %dst = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.topk_stamp_local_positions(%dst) {tag_bits = 8 : i32} : (index) -> ()
  ttkernel.topk_strip_rank_tags(%dst) {tag_bits = 6 : i32} : (index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// Canonicalize selects the plain init. A following fused stage replaces it.
// CHECK-LABEL: func.func @canonicalize_then_fused_stage
// CHECK: ttkernel.topk_tile_init()
// CHECK-NEXT: ttkernel.topk_canonicalize_negzero_values
// CHECK: ttkernel.topk_tile_init
// CHECK-SAME: fused = true
// CHECK-NEXT: ttkernel.topk_local_sort
func.func @canonicalize_then_fused_stage() {
  %dst = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.topk_canonicalize_negzero_values(%dst) : (index) -> ()
  ttkernel.topk_local_sort(%dst, %c0_i32, %c4_i32, %c0_i32) {fused = true}
      : (index, i32, i32, i32) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// -----

// The uint16 pack move is not a stage, so it does not receive topk_tile_init.
// CHECK-LABEL: func.func @uint16_move_has_no_topk_init
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NEXT: ttkernel.topk_uint16_move_dest_tile_to_pack_half
// CHECK-NOT: ttkernel.topk_tile_init
func.func @uint16_move_has_no_topk_init() {
  %dst = arith.constant 0 : index
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.topk_uint16_move_dest_tile_to_pack_half(%dst) : (index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
