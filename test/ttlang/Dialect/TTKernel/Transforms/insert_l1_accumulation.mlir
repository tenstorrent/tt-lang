// Verifies ttkernel-insert-l1-accumulation: pack_reconfig_l1_acc guards are
// inserted around reduction loops. The enable call happens once after the
// first iteration's last pack (iv == lb), and disable guards bracket the
// accumulation scope.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttkernel-insert-l1-accumulation)' --split-input-file | FileCheck %s
// Idempotency: running twice produces the same output.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttkernel-insert-l1-accumulation, ttkernel-insert-l1-accumulation)' --split-input-file | FileCheck %s

// Basic L1 acc loop: enable after first iteration, disable before/after loop.

// CHECK-LABEL: func.func @basic_l1_acc_loop
// CHECK: ttkernel.pack_reconfig_l1_acc(%{{.*}}) : (i32)
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.tile_regs_acquire
// CHECK:   ttkernel.pack_tile
// CHECK:   ttkernel.tile_regs_release
// CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if %[[CMP]]
// CHECK:     %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK:     ttkernel.pack_reconfig_l1_acc(%[[ENABLE]]) : (i32)
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%{{.*}}) : (i32)
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @basic_l1_acc_loop() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// Reduction loop fallback (ttl.reduction_loop attribute) with sum reduce.

// CHECK-LABEL: func.func @reduction_loop_fallback
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for
// CHECK:   arith.cmpi eq
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @reduction_loop_fallback() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_in = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %cb_scaler = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %cb_out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.reduce_tile(%cb_in, %cb_scaler, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb_out, %c0, true) : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.reduction_loop}
  return
}

// -----

// Max reduce loops should NOT get L1 accumulation guards.

// CHECK-LABEL: func.func @max_reduce_no_l1_acc
// CHECK-NOT: pack_reconfig_l1_acc
func.func @max_reduce_no_l1_acc() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_in = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %cb_scaler = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %cb_out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.reduce_tile(%cb_in, %cb_scaler, %c0, %c0, %c0, <reduce_max>, <reduce_dim_col>) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb_out, %c0, true) : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.reduction_loop}
  return
}

// -----

// No reduction loop attribute: no transformation.

// CHECK-LABEL: func.func @no_reduction_loop
// CHECK-NOT: pack_reconfig_l1_acc
func.func @no_reduction_loop() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  }
  return
}

// -----

// Subblocked loop: multiple acquire/release pairs per iteration inside nested
// loops. The enable guard should appear once after the outermost subblock loop
// (containing the last release), not after each individual release.

// CHECK-LABEL: func.func @subblocked_loop
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   scf.for
// CHECK:     ttkernel.tile_regs_acquire
// CHECK:     ttkernel.tile_regs_release
// CHECK:   }
// CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if %[[CMP]]
// CHECK:     %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK:     ttkernel.pack_reconfig_l1_acc(%[[ENABLE]]) : (i32)
// CHECK: }
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @subblocked_loop() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    scf.for %sb = %c0 to %c2 step %c1 {
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.matmul_block(%cb, %cb, %c0, %c0, %c0, %c0_i32, %c1_i32, %c1_i32, %c1_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
    }
  } {ttl.l1_acc_loop}
  return
}

// -----

// L1 acc loop with no tile_regs_acquire/release inside: no guards inserted.

// CHECK-LABEL: func.func @l1_acc_loop_no_sync
// CHECK-NOT: pack_reconfig_l1_acc
func.func @l1_acc_loop_no_sync() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  } {ttl.l1_acc_loop}
  return
}

// -----

// L1 acc loop inside an unannotated outer loop (the realistic pattern:
// outer M/N iteration loop wraps the inner K reduction loop). The disable
// guards bracket the inner K loop, not the outer loop. Each outer
// iteration gets a fresh disable-before -> K loop -> disable-after cycle.

// CHECK-LABEL: func.func @l1_acc_inside_outer_loop
// CHECK: scf.for
// CHECK:   ttkernel.pack_reconfig_l1_acc
// CHECK:   scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:     ttkernel.tile_regs_acquire
// CHECK:     ttkernel.pack_tile
// CHECK:     ttkernel.tile_regs_release
// CHECK:     %[[CMP:.*]] = arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:     scf.if %[[CMP]]
// CHECK:       ttkernel.pack_reconfig_l1_acc
// CHECK:   }
// CHECK:   ttkernel.cb_push_back
// CHECK:   ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @l1_acc_inside_outer_loop() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  scf.for %outer = %c0 to %c2 step %c1 {
    scf.for %inner = %c0 to %c4 step %c1 {
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
    } {ttl.l1_acc_loop}
    ttkernel.cb_push_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  }
  return
}

// -----

// Multiple consecutive cb_push_back ops after the loop (multi-output compute).
// The disable guard should go after the last push.

// CHECK-LABEL: func.func @multi_push_after_loop
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for
// CHECK:   arith.cmpi eq
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @multi_push_after_loop() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb0, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c0, %cb1, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb0, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_push_back(%cb1, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// Nested l1_acc loops: reserve is outside both loops, so both are annotated
// and all iterations accumulate into the same CB slot. Disable guards
// bracket the outermost loop; enable fires once after the first inner
// iteration of the first outer iteration.

// CHECK-LABEL: func.func @nested_l1_acc_loops
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for
// CHECK:   scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:     ttkernel.tile_regs_acquire
// CHECK:     ttkernel.tile_regs_release
// CHECK:     arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:     scf.if
// CHECK:       ttkernel.pack_reconfig_l1_acc
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @nested_l1_acc_loops() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  scf.for %outer = %c0 to %c2 step %c1 {
    scf.for %inner = %c0 to %c4 step %c1 {
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
    } {ttl.l1_acc_loop}
  } {ttl.l1_acc_loop}
  return
}

// -----

// Nested reduction loops (multi-dim reduce): all iterations contribute to
// a single accumulated result. Same structure as nested l1_acc loops.

// CHECK-LABEL: func.func @nested_reduction_loops
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for
// CHECK:   scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:     ttkernel.tile_regs_acquire
// CHECK:     ttkernel.tile_regs_release
// CHECK:     arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:     scf.if
// CHECK:       ttkernel.pack_reconfig_l1_acc
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @nested_reduction_loops() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_in = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %cb_scaler = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %cb_out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %row = %c0 to %c2 step %c1 {
    scf.for %col = %c0 to %c2 step %c1 {
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.reduce_tile(%cb_in, %cb_scaler, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_col>) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%c0, %cb_out, %c0, true) : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
    } {ttl.reduction_loop}
  } {ttl.reduction_loop}
  return
}

// -----

// Two consecutive L1 acc loops writing to the same CB.
// The reserve/push scope spans both loops. One disable pair brackets the
// entire scope; only the first loop gets the enable guard.

// CHECK-LABEL: func.func @consecutive_l1_acc_loops
// CHECK: ttkernel.cb_reserve_back
// Disable before first loop.
// CHECK: ttkernel.pack_reconfig_l1_acc
// First loop with enable guard.
// CHECK: scf.for %[[IV1:.*]] = %[[LB1:.*]] to
// CHECK:   ttkernel.tile_regs_acquire
// CHECK:   ttkernel.pack_tile
// CHECK:   ttkernel.tile_regs_release
// CHECK:   arith.cmpi eq, %[[IV1]], %[[LB1]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// No disable between the loops. Unconditional enable re-arms L1 acc
// after any init ops that may reset packer state.
// CHECK-NOT: pack_reconfig_l1_acc(%{{.*}}0
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for %[[IV2:.*]] = %[[LB2:.*]] to
// CHECK:   ttkernel.tile_regs_acquire
// CHECK:   ttkernel.pack_tile
// CHECK:   ttkernel.tile_regs_release
// CHECK:   arith.cmpi eq, %[[IV2]], %[[LB2]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// Push then disable.
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @consecutive_l1_acc_loops() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.for %iv1 = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  scf.for %iv2 = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// Single loop with two independent accumulating outputs.
// Both pack to different CBs but share one L1 acc enable/disable scope.

// CHECK-LABEL: func.func @two_outputs_one_loop
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.tile_regs_acquire
// CHECK:   ttkernel.pack_tile
// CHECK:   ttkernel.tile_regs_release
// CHECK:   ttkernel.tile_regs_acquire
// CHECK:   ttkernel.pack_tile
// CHECK:   ttkernel.tile_regs_release
// Enable after the last release (second output).
// CHECK:   arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// Two pushes then disable.
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @two_outputs_one_loop() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb0, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb1, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb0, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_push_back(%cb1, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// Idempotency: input already has pack_reconfig_l1_acc guards. Running
// the pass again should not insert duplicates.

// CHECK-LABEL: func.func @already_guarded
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for
// CHECK:   ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @already_guarded() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  ttkernel.pack_reconfig_l1_acc(%c0_i32) : (i32) -> ()
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    %cmp = arith.cmpi eq, %iv, %c0 : index
    scf.if %cmp {
      ttkernel.pack_reconfig_l1_acc(%c1_i32) : (i32) -> ()
    }
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.pack_reconfig_l1_acc(%c0_i32) : (i32) -> ()
  return
}

// -----

// Two consecutive annotated loops packing to DIFFERENT CBs.
// Each loop gets its own independent disable pair.

// CHECK-LABEL: func.func @different_cb_siblings
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for
// CHECK:   ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for
// CHECK:   ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @different_cb_siblings() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  scf.for %iv1 = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb0, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb0, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.for %iv2 = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb1, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb1, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// Consecutive annotated loops with init ops between them (the real-world
// pattern from the full pipeline). The scope must span past the init ops
// to include the push after the second loop.

// CHECK-LABEL: func.func @consecutive_with_init_between
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for
// CHECK:   ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK-NOT: pack_reconfig_l1_acc(%{{.*}}0
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK: scf.for
// CHECK:   ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @consecutive_with_init_between() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb_in0 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb_in1 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.for %iv1 = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.matmul_block(%cb_in0, %cb_in0, %c0, %c0, %c0, %c0_i32, %c1_i32, %c1_i32, %c1_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  // init_short between the two loops (InsertInits emits init_short when
  // sibling loops share an output CB, to avoid clobbering PACK config).
  "ttkernel.mm_block_init_short"(%cb_in1, %cb_in1, %c0_i32, %c1_i32, %c1_i32, %c1_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32, i32, i32, i32) -> ()
  scf.for %iv2 = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.matmul_block(%cb_in1, %cb_in1, %c0, %c0, %c0, %c0_i32, %c1_i32, %c1_i32, %c1_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}
