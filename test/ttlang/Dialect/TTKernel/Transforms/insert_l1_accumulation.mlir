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
// No disable between the loops. Unconditional enable re-enables L1 acc
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

// -----

// A non-accumulating pack into the same CB precedes the L1 acc loop, so L1
// already holds a prior value when the loop starts. The pre-group reconfig
// must be ENABLE (1) instead of DISABLE (0) so iteration 0 accumulates onto
// that value, and the per-iteration conditional ENABLE on the root loop must
// be omitted.

// CHECK-LABEL: func.func @prior_value_then_l1_acc_loop
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.pack_tile
// CHECK: ttkernel.tile_regs_release
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[ENABLE]]) : (i32)
// CHECK: scf.for
// CHECK:   ttkernel.tile_regs_acquire
// CHECK:   ttkernel.pack_tile
// CHECK:   ttkernel.tile_regs_release
// CHECK-NOT: arith.cmpi
// CHECK-NOT: scf.if
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @prior_value_then_l1_acc_loop() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  // Non-accumulating pack (corresponds to user's `.store(...)` before the loop).
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  // L1 acc loop (corresponds to user's `+=` loop).
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

// A pack into a DIFFERENT CB (cb_other) leaves the loop's pack CB without a
// prior value. The standard DISABLE-before / per-iter ENABLE / DISABLE-after
// pattern must be preserved.

// CHECK-LABEL: func.func @prior_pack_different_cb_ignored
// CHECK: ttkernel.pack_tile(%{{.*}}, %[[OTHER:.*]],
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.pack_tile
// CHECK:   arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if
// CHECK:     %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK:     ttkernel.pack_reconfig_l1_acc(%[[ENABLE]]) : (i32)
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
func.func @prior_pack_different_cb_ignored() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_out = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb_other = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb_other, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_reserve_back(%cb_out, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  // Pack into cb_other, NOT cb_out — must not be treated as init for cb_out.
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb_other, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  ttkernel.cb_push_back(%cb_other, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb_out, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb_out, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// An intervening cb_reserve_back on the loop's pack CB resets the L1 slot,
// so a pack before that boundary does not provide a prior value to the loop.
// The standard DISABLE-before / per-iter ENABLE / DISABLE-after pattern must
// be preserved.

// CHECK-LABEL: func.func @cb_reserve_back_blocks_prior_value_detection
// CHECK: ttkernel.pack_tile
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_reserve_back
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.pack_tile
// CHECK:   arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
func.func @cb_reserve_back_blocks_prior_value_detection() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  ttkernel.cb_push_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  // New reservation on the same CB resets the L1 slot for the loop.
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
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

// A non-accumulating pack precedes a group of two consecutive L1 acc loops
// sharing the same pack CB. The root loop suppresses its per-iteration
// enable (the pre-group enable already covers it); the sibling keeps both
// its unconditional pre-loop enable and its per-iteration enable.

// CHECK-LABEL: func.func @prior_value_root_then_plain_acc_sibling
// CHECK: ttkernel.pack_tile
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: %[[ENABLE0:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[ENABLE0]]) : (i32)
// CHECK: scf.for
// CHECK:   ttkernel.pack_tile
// CHECK-NOT: arith.cmpi
// CHECK-NOT: scf.if
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: }
// CHECK: %[[ENABLE1:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[ENABLE1]]) : (i32)
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.pack_tile
// CHECK:   arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
func.func @prior_value_root_then_plain_acc_sibling() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
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

// The prior pack is wrapped in a non-annotated scf.for (mirrors the
// compiler-generated tile-loop wrapper that the real pipeline produces
// around the user's `.store(...)`). The pre-group reconfig must still be
// ENABLE because the wrapper runs with L1 acc disabled.

// CHECK-LABEL: func.func @prior_pack_inside_tile_loop_wrapper
// CHECK: scf.for
// CHECK:   ttkernel.pack_tile
// CHECK: } {ttl.tile_loop_stride
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[ENABLE]]) : (i32)
// CHECK: scf.for
// CHECK:   ttkernel.pack_tile
// CHECK-NOT: arith.cmpi
// CHECK-NOT: scf.if
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: } {ttl.l1_acc_loop}
// CHECK: ttkernel.cb_push_back
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @prior_pack_inside_tile_loop_wrapper() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.for %iv0 = %c0 to %c1 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.tile_loop_stride = 1 : index}
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

// A pack inside an scf.if conditionally writes to L1, so the pre-group
// reconfig cannot assume L1 holds a prior value. The conservative fallback
// in precededByNonAccumulatingPack treats any region-bearing op (other
// than scf.for that we recurse into) that packs to our CB as a boundary.

// CHECK-LABEL: func.func @prior_pack_inside_scf_if_not_treated_as_prior
// CHECK: scf.if
// CHECK:   ttkernel.pack_tile
// CHECK: }
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.pack_tile
// CHECK:   arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
func.func @prior_pack_inside_scf_if_not_treated_as_prior(%cond: i1) attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.if %cond {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  }
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

// Multi-output L1 acc loop with a prior pack for only one of its CBs. L1
// acc is enabled or disabled for the entire sync region — there is no
// per-CB switch. The standard pattern (overwrite + per-iter enable) must
// be preserved.

// CHECK-LABEL: func.func @multi_output_partial_prior_falls_back_to_standard
// CHECK: ttkernel.pack_tile
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.pack_tile
// CHECK:   ttkernel.pack_tile
// CHECK:   arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
func.func @multi_output_partial_prior_falls_back_to_standard() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_a = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb_b = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb_a, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_reserve_back(%cb_b, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  // Prior pack only for cb_a — cb_b has no prior value.
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb_a, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb_a, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c0, %cb_b, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb_a, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_push_back(%cb_b, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// Multi-output L1 acc loop where BOTH pack CBs have prior packs. All
// covered → enable pre-group, suppress per-iter enable on the root.

// CHECK-LABEL: func.func @multi_output_full_prior_enables_pre_group
// CHECK: ttkernel.pack_tile
// CHECK: ttkernel.pack_tile
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[ENABLE]]) : (i32)
// CHECK: scf.for
// CHECK:   ttkernel.pack_tile
// CHECK:   ttkernel.pack_tile
// CHECK-NOT: arith.cmpi
// CHECK-NOT: scf.if
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_push_back
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
func.func @multi_output_full_prior_enables_pre_group() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_a = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb_b = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb_a, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_reserve_back(%cb_b, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb_a, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.pack_tile(%c0, %cb_b, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb_a, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.pack_tile(%c0, %cb_b, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb_a, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_push_back(%cb_b, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// A reduction loop (ttl.reduction_loop) packing to cb_out is a boundary
// for a later L1-acc loop on the same CB: its packs run under its own
// enable scope, so they do not provide a prior value. Between the two, a
// cb_push_back + cb_reserve_back pair resets the L1 slot. The L1-acc
// loop must use the standard pattern.

// CHECK-LABEL: func.func @prior_reduction_loop_boundary
// CHECK: scf.for
// CHECK:   ttkernel.pack_tile
// CHECK: } {ttl.reduction_loop
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_reserve_back
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.pack_tile
// CHECK:   arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
func.func @prior_reduction_loop_boundary() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.for %iv0 = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.reduction_loop}
  ttkernel.cb_push_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.for %iv1 = %c0 to %c4 step %c1 {
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

// A pack inside an scf.while triggers the conservative fallback: the walk
// does not reason about the while's condition or iteration count, so any
// pack into the L1-acc loop's pack CB inside the while body makes it a
// boundary. The standard pattern must be preserved.

// CHECK-LABEL: func.func @prior_pack_inside_scf_while_not_treated_as_prior
// CHECK: scf.while
// CHECK:   ttkernel.pack_tile
// CHECK: }
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.pack_tile
// CHECK:   arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
func.func @prior_pack_inside_scf_while_not_treated_as_prior(%init: i32) attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  %final = scf.while (%arg = %init) : (i32) -> (i32) {
    %cond = arith.cmpi sgt, %arg, %zero : i32
    scf.condition(%cond) %arg : i32
  } do {
  ^bb0(%arg: i32):
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    %next = arith.subi %arg, %one : i32
    scf.yield %next : i32
  }
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

// A non-annotated scf.for that packs only to a different CB sits between
// a real prior pack into our CB and the L1-acc loop. The backward walk
// must skip over the unrelated for (it does not touch our CB) and still
// find the earlier prior pack.

// CHECK-LABEL: func.func @unrelated_scf_for_skipped_to_prior_pack
// CHECK: ttkernel.pack_tile(%{{.*}}, %[[CB_OUT:.*]],
// CHECK: scf.for
// CHECK:   ttkernel.pack_tile(%{{.*}}, %[[CB_OTHER:.*]],
// CHECK: } {ttl.tile_loop_stride
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[ENABLE]]) : (i32)
// CHECK: scf.for
// CHECK:   ttkernel.pack_tile(%{{.*}}, %[[CB_OUT]],
// CHECK-NOT: arith.cmpi
// CHECK-NOT: scf.if
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: } {ttl.l1_acc_loop}
// CHECK: ttkernel.cb_push_back
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @unrelated_scf_for_skipped_to_prior_pack() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_out = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb_other = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb_other, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_reserve_back(%cb_out, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  // Prior pack into cb_out (our CB).
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb_out, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  // Unrelated non-annotated scf.for packing cb_other — walk must skip over.
  scf.for %iv0 = %c0 to %c1 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb_other, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.tile_loop_stride = 1 : index}
  ttkernel.cb_push_back(%cb_other, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb_out, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb_out, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// An scf.if whose body packs only to a different CB sits between a real
// prior pack into our CB and the L1-acc loop. The conservative fallback
// must let the walk skip over the if (its body does not touch our CB)
// and still find the earlier prior pack.

// CHECK-LABEL: func.func @unrelated_scf_if_skipped_to_prior_pack
// CHECK: ttkernel.pack_tile(%{{.*}}, %[[CB_OUT:.*]],
// CHECK: scf.if
// CHECK:   ttkernel.pack_tile(%{{.*}}, %[[CB_OTHER:.*]],
// CHECK: }
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[ENABLE]]) : (i32)
// CHECK: scf.for
// CHECK:   ttkernel.pack_tile(%{{.*}}, %[[CB_OUT]],
// CHECK-NOT: arith.cmpi
// CHECK-NOT: scf.if
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c0_i32
// CHECK: } {ttl.l1_acc_loop}
// CHECK: ttkernel.cb_push_back
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc
func.func @unrelated_scf_if_skipped_to_prior_pack(%cond: i1) attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb_out = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %cb_other = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb_other, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.cb_reserve_back(%cb_out, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  // Prior pack into cb_out (our CB).
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %cb_out, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  // Unrelated scf.if packing cb_other — walk must skip over.
  scf.if %cond {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb_other, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  }
  ttkernel.cb_push_back(%cb_other, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  scf.for %iv = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb_out, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb_out, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}

// -----

// A non-annotated scf.for with lb == ub packs to the L1-acc loop's CB but
// never executes, so its pack does not reach L1. The walk must not credit
// such a for as a prior-value contributor; the lowering falls back to the
// no-prior-pack pattern (DISABLE before, per-iteration ENABLE, DISABLE
// after).

// CHECK-LABEL: func.func @zero_trip_wrapper_not_credited
// CHECK: scf.for %{{.*}} = %[[LBU:.*]] to %[[LBU]]
// CHECK:   ttkernel.pack_tile
// CHECK: } {ttl.tile_loop_stride
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%c1_i32
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]]) : (i32)
// CHECK: scf.for %[[IV:.*]] = %[[LB:.*]] to
// CHECK:   ttkernel.pack_tile
// CHECK:   arith.cmpi eq, %[[IV]], %[[LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: }
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
func.func @zero_trip_wrapper_not_credited() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  // Non-annotated scf.for with lb == ub: zero-trip, pack never runs.
  scf.for %iv0 = %c0 to %c0 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.tile_loop_stride = 1 : index}
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

// Two annotated L1-acc loops sharing one CB reservation with a bare
// non-annotated scf.for between them that packs nothing to the shared
// CB. `collectLoopGroups` must span the bare for and place both
// annotated loops in one group; the bare for is transparent.

// CHECK-LABEL: func.func @annotated_siblings_split_by_bare_for
// CHECK: ttkernel.cb_reserve_back
// CHECK: %[[DISABLE:.*]] = arith.constant 0 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
// CHECK: scf.for {{.*}} {
// CHECK:   ttkernel.pack_tile
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: } {ttl.l1_acc_loop}
// CHECK-NOT: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
// Bare non-annotated scf.for — transparent to the group.
// CHECK: scf.for
// CHECK: }
// CHECK: %[[ENABLE:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[ENABLE]])
// CHECK: scf.for {{.*}} {
// CHECK:   ttkernel.pack_tile
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc
// CHECK: } {ttl.l1_acc_loop}
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[DISABLE]])
func.func @annotated_siblings_split_by_bare_for() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  ttkernel.cb_reserve_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  // Annotated loop A: accumulates sum_{ivA} pack(cb) into L1.
  scf.for %ivA = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  // Bare non-annotated scf.for with no packs — splits A and B into
  // separate loop groups.
  scf.for %ivU = %c0 to %c1 step %c1 {
  }
  // Annotated loop B: should accumulate onto A's L1 value in the same
  // reservation, but overwrites it on iteration 0.
  scf.for %ivB = %c0 to %c4 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  } {ttl.l1_acc_loop}
  ttkernel.cb_push_back(%cb, %c4_i32) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, i32) -> ()
  return
}
