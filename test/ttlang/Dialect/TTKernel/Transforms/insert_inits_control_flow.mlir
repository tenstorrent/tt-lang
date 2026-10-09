// RUN: ttlang-opt %s --ttkernel-insert-inits --ttkernel-verify-hardware-config --split-input-file | FileCheck %s
// Summary: Per-op MATH init placement. A uniform scf region with no DST sync
// takes one init before the region. Mixed consumers, and any region that
// contains a DST sync or unsupported nested region, keep the init at the
// consumer. copy_tile_init stays at the copy. DST sync preserves a non-reduce
// configuration. A reduce configuration on any incoming path emits
// reduce_uninit before a sync or non-reduce consumer.

// CHECK-LABEL: func.func @inits_inside_if
// CHECK:       scf.if
// CHECK:         ttkernel.exp_tile_init
// CHECK-NEXT:    ttkernel.exp_tile(
// CHECK:       } else
// CHECK:         ttkernel.log_tile_init
// CHECK-NEXT:    ttkernel.log_tile
func.func @inits_inside_if(%cond: i1) {
  %c0 = arith.constant 0 : index
  scf.if %cond {
    ttkernel.exp_tile(%c0) : (index) -> ()
  } else {
    ttkernel.log_tile(%c0) : (index) -> ()
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @inits_inside_for
// CHECK-NOT:   ttkernel.exp_tile_init
// CHECK:       scf.for
// CHECK:         ttkernel.exp_tile_init
// CHECK-NEXT:    ttkernel.exp_tile(
// CHECK:         ttkernel.log_tile_init
// CHECK-NEXT:    ttkernel.log_tile
func.func @inits_inside_for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    ttkernel.exp_tile(%c0) : (index) -> ()
    ttkernel.log_tile(%c0) : (index) -> ()
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @sync_preserves_non_reduce
// CHECK:       ttkernel.exp_tile_init
// CHECK-NEXT:  ttkernel.exp_tile(
// CHECK:       ttkernel.tile_regs_release
// CHECK-NOT:   ttkernel.exp_tile_init
// CHECK:       ttkernel.exp_tile(
func.func @sync_preserves_non_reduce() {
  %c0 = arith.constant 0 : index
  ttkernel.exp_tile(%c0) : (index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.tile_regs_release() : () -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}

// -----

// CHECK-LABEL: func.func @uninit_before_sync
// CHECK:       ttkernel.reduce_init
// CHECK-NEXT:  ttkernel.reduce_tile
// CHECK-NEXT:  ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.tile_regs_commit
func.func @uninit_before_sync() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  func.return
}

// -----

// Cleanup before a sync resets the descriptor, so a later reduce needs a new
// init even when its descriptor matches the pre-sync reduce.
// CHECK-LABEL: func.func @sync_resets_reduce
// CHECK:       ttkernel.reduce_init
// CHECK-NEXT:  ttkernel.reduce_tile
// CHECK-NEXT:  ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.tile_regs_commit
// CHECK-NEXT:  ttkernel.reduce_init
// CHECK-NEXT:  ttkernel.reduce_tile
// CHECK-NEXT:  ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.tile_regs_commit
// CHECK-NEXT:  ttkernel.exp_tile_init
// CHECK-NEXT:  ttkernel.exp_tile(
func.func @sync_resets_reduce() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}

// -----

// A reduce init added after cleanup can itself require cleanup immediately
// before a non-reduce init.
// CHECK-LABEL: func.func @repeated_cleanup_before_non_reduce
// CHECK:       ttkernel.reduce_init
// CHECK-NEXT:  ttkernel.reduce_tile
// CHECK-NEXT:  ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.tile_regs_commit
// CHECK-NEXT:  ttkernel.reduce_init
// CHECK-NEXT:  ttkernel.reduce_tile
// CHECK-NEXT:  ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.exp_tile_init
// CHECK-NEXT:  ttkernel.exp_tile(
func.func @repeated_cleanup_before_non_reduce() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}

// -----

// CHECK-LABEL: func.func @uninit_before_loop
// CHECK:       ttkernel.reduce_init
// CHECK-NEXT:  ttkernel.reduce_tile
// CHECK-NEXT:  ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.exp_tile_init
// CHECK-NEXT:  scf.for
// CHECK:         ttkernel.exp_tile(
func.func @uninit_before_loop() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  scf.for %i = %c0 to %c2 step %c1 {
    ttkernel.exp_tile(%c0) : (index) -> ()
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @uninit_then_init
// CHECK:       ttkernel.reduce_init
// CHECK-NEXT:  ttkernel.reduce_tile
// CHECK-NEXT:  ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.exp_tile_init
// CHECK-NEXT:  ttkernel.exp_tile(
func.func @uninit_then_init() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}

// -----

// A reduce on one branch can make the configuration unknown at the merge, but
// the following sync still requires reduce_uninit.
// CHECK-LABEL: func.func @uninit_after_partial_reduce_branch
// CHECK:       ttkernel.reduce_init
// CHECK-NEXT:  scf.if
// CHECK:         ttkernel.reduce_tile
// CHECK:       ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.tile_regs_commit
func.func @uninit_after_partial_reduce_branch(%cond: i1) {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  scf.if %cond {
    ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  }
  ttkernel.tile_regs_commit() : () -> ()
  func.return
}

// -----

// A dynamic-trip loop may execute a reduce, so its zero-trip path must not
// discard the cleanup obligation at the following sync.
// CHECK-LABEL: func.func @uninit_after_dynamic_reduce_loop
// CHECK:       ttkernel.reduce_init
// CHECK-NEXT:  scf.for
// CHECK:         ttkernel.reduce_tile
// CHECK:       ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.tile_regs_commit
func.func @uninit_after_dynamic_reduce_loop(%n: index) {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  }
  ttkernel.tile_regs_commit() : () -> ()
  func.return
}

// -----

// Identical consumers in both branches hoist one init, not two.
// CHECK-LABEL: func.func @one_init_for_both_branches
// CHECK:       ttkernel.exp_tile_init
// CHECK-NEXT:  scf.if
// CHECK-NOT:   ttkernel.exp_tile_init
// CHECK:       ttkernel.exp_tile(
// CHECK:       ttkernel.exp_tile(
func.func @one_init_for_both_branches(%cond: i1) {
  %c0 = arith.constant 0 : index
  scf.if %cond {
    ttkernel.exp_tile(%c0) : (index) -> ()
  } else {
    ttkernel.exp_tile(%c0) : (index) -> ()
  }
  func.return
}

// -----

// A trip count of one runs the body on the incoming state. The body is
// uniform and has no sync, so the init is placed before the loop.
// CHECK-LABEL: func.func @trip_count_one
// CHECK:       ttkernel.exp_tile_init
// CHECK-NEXT:  scf.for
// CHECK-NOT:   ttkernel.exp_tile_init
// CHECK:       ttkernel.exp_tile(
func.func @trip_count_one() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    ttkernel.exp_tile(%c0) : (index) -> ()
  }
  func.return
}

// -----

// A trip count of zero does not visit the body, so the dead exp gets no init.
// CHECK-LABEL: func.func @trip_count_zero
// CHECK:       scf.for
// CHECK-NEXT:  ttkernel.exp_tile(
// CHECK:       ttkernel.exp_tile_init
// CHECK-NEXT:  ttkernel.exp_tile(
func.func @trip_count_zero() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c0 step %c1 {
    ttkernel.exp_tile(%c0) : (index) -> ()
  }
  ttkernel.exp_tile(%c0) : (index) -> ()
  func.return
}

// -----

// The two regions of scf.while need different inits, so each stays inside.
// CHECK-LABEL: func.func @inits_inside_while
// CHECK:       scf.while
// CHECK:         ttkernel.exp_tile_init
// CHECK-NEXT:    ttkernel.exp_tile(
// CHECK:       } do
// CHECK:         ttkernel.log_tile_init
// CHECK-NEXT:    ttkernel.log_tile
func.func @inits_inside_while(%cond: i1) {
  %c0 = arith.constant 0 : index
  scf.while : () -> () {
    ttkernel.exp_tile(%c0) : (index) -> ()
    scf.condition(%cond)
  } do {
    ttkernel.log_tile(%c0) : (index) -> ()
    scf.yield
  }
  func.return
}

// -----

// A DST sync inside the loop keeps the init inside the sync region.
// CHECK-LABEL: func.func @sync_blocks_hoist
// CHECK:       scf.for
// CHECK-NOT:   ttkernel.exp_tile_init
// CHECK:       ttkernel.tile_regs_acquire
// CHECK-NEXT:  ttkernel.exp_tile_init
// CHECK-NEXT:  ttkernel.exp_tile(
func.func @sync_blocks_hoist() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.exp_tile(%c0) : (index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.tile_regs_release() : () -> ()
  }
  func.return
}

// -----

// copy_tile_init is left at the copy so canonicalization can hoist it later.
// CHECK-LABEL: func.func @copy_init_stays_at_copy
// CHECK:       scf.for
// CHECK-NOT:   ttkernel.copy_tile_init
// CHECK:       ttkernel.copy_tile_init
// CHECK-NEXT:  ttkernel.copy_tile
func.func @copy_init_stays_at_copy() {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    ttkernel.copy_tile(%cb, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index) -> ()
  }
  func.return
}

// -----

// invoke_sfpi writes an unknown MATH configuration. Hoisting considers the
// whole if, so effect collection must accept the op and keep the init inside.
// CHECK-LABEL: func.func @invoke_sfpi_blocks_hoist
// CHECK-NOT:   ttkernel.exp_tile_init
// CHECK:       scf.if
// CHECK:         ttkernel.exp_tile_init
// CHECK-NEXT:    ttkernel.exp_tile(
// CHECK:         ttkernel.invoke_sfpi
// CHECK-NOT:   ttkernel.reduce_uninit
func.func @invoke_sfpi_blocks_hoist(%cond: i1) {
  %c0 = arith.constant 0 : index
  scf.if %cond {
    ttkernel.exp_tile(%c0) : (index) -> ()
    ttkernel.invoke_sfpi {
    }
  }
  func.return
}

// -----

// An unsupported nested region prevents hoisting without inventing reduce
// provenance when no reduce init is present.
// CHECK-LABEL: func.func @unmodeled_region_blocks_hoist
// CHECK:       scf.if
// CHECK:         scf.execute_region
// CHECK-NEXT:      ttkernel.exp_tile_init
// CHECK-NEXT:      ttkernel.exp_tile(
// CHECK:         ttkernel.exp_tile_init
// CHECK-NEXT:    ttkernel.exp_tile(
// CHECK-NOT:   ttkernel.reduce_uninit
func.func @unmodeled_region_blocks_hoist(%cond: i1) {
  %c0 = arith.constant 0 : index
  scf.if %cond {
    scf.execute_region {
      ttkernel.exp_tile(%c0) : (index) -> ()
      scf.yield
    }
    ttkernel.exp_tile(%c0) : (index) -> ()
  }
  func.return
}

// -----

// Reduce provenance produced inside an unsupported region survives its
// unknown descriptor boundary.
// CHECK-LABEL: func.func @unmodeled_region_preserves_reduce_provenance
// CHECK:       scf.execute_region
// CHECK:         ttkernel.reduce_init
// CHECK-NEXT:    ttkernel.reduce_tile
// CHECK:       ttkernel.reduce_uninit
// CHECK-NEXT:  ttkernel.tile_regs_commit
func.func @unmodeled_region_preserves_reduce_provenance() {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  scf.execute_region {
    ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) {ttl.reduce_output_cb_index = 2 : index} : (!ttkernel.cb<4, !ttcore.tile<32x32, f32>>, !ttkernel.cb<4, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
    scf.yield
  }
  ttkernel.tile_regs_commit() : () -> ()
  func.return
}
