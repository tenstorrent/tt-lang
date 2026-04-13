// Verifies ttkernel-insert-l1-accumulation: pack_reconfig_l1_acc guards are
// inserted around reduction loops. The enable call happens once after the
// first iteration's last pack (iv == lb), and disable guards bracket the
// outermost loop.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttkernel-insert-l1-accumulation)' --split-input-file | FileCheck %s

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
