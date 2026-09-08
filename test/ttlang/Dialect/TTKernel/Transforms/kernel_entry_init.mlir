// Summary: One startup precedes control flow and CB waits. Region transitions
// reconfigure output formats and geometry without resetting DST synchronization.
// RUN: ttlang-opt %s --ttkernel-insert-inits --split-input-file | FileCheck %s --implicit-check-not=ttkernel.init_sfpu --implicit-check-not=ttkernel.binary_op_init_common --implicit-check-not=ttkernel.compute_kernel_hw_startup

// A conditional first region must not make hardware startup conditional. The
// following loop changes both input and output tile geometry and data format.
// CHECK-LABEL: func.func @conditional_then_loop
// CHECK-NEXT: %[[A:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-NEXT: %[[O:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-NEXT: ttkernel.compute_kernel_hw_startup(%[[A]], %[[A]], %[[O]])
// CHECK-DAG: %[[B:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK-DAG: %[[P:.*]] = ttkernel.get_compile_time_arg_val(3)
// CHECK: scf.if
// CHECK: ttkernel.cb_wait_front
// CHECK: ttkernel.pack_reconfig_data_format(%[[O]]) {tile_dim_reconfig = true}
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NEXT: ttkernel.reconfig_data_format(%[[A]], %[[A]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[A]])
// CHECK: ttkernel.tile_regs_release
// CHECK: scf.for
// CHECK: ttkernel.pack_reconfig_data_format(%[[P]]) {tile_dim_reconfig = true}
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NEXT: ttkernel.reconfig_data_format(%[[B]], %[[B]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[B]])
// CHECK: ttkernel.tile_regs_release
// CHECK-NOT: ttkernel.compute_kernel_hw_startup
// CHECK: return
func.func @conditional_then_loop(%condition: i1) {
  %a = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %o = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  %b = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<16x32, f32>>
  %p = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.cb<2, !ttcore.tile<16x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %one_i32 = arith.constant 1 : i32
  scf.if %condition {
    ttkernel.cb_wait_front(%a, %one_i32) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile(%a, %c0, %c0) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %o, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  }
  scf.for %i = %c0 to %c2 step %c1 {
    ttkernel.tile_regs_acquire() : () -> ()
    ttkernel.copy_tile(%b, %c0, %c0) : (!ttkernel.cb<2, !ttcore.tile<16x32, f32>>, index, index) -> ()
    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %p, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<16x32, f32>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
  }
  return
}

// -----

// Matmul maps its left input to SrcB and its right input to SrcA at startup
// and on every reconfiguration, including after an intervening copy operation.
// CHECK-LABEL: func.func @matmul_source_order
// CHECK-NEXT: ttkernel.compute_kernel_hw_startup(%arg1, %arg0, %arg2)
// CHECK: ttkernel.reconfig_data_format(%arg1, %arg0)
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NEXT: ttkernel.reconfig_data_format(%arg2, %arg2)
// CHECK-NEXT: ttkernel.copy_tile_init(%arg2)
// CHECK-NEXT: ttkernel.copy_tile
// CHECK-NEXT: ttkernel.reconfig_data_format(%arg1, %arg0)
// CHECK-NEXT: "ttkernel.mm_block_init_short"(%arg0, %arg1,
// CHECK-NEXT: ttkernel.matmul_block(%arg0, %arg1,
// CHECK-NOT: ttkernel.compute_kernel_hw_startup
// CHECK: return
func.func @matmul_source_order(%lhs: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %rhs: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %out: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.copy_tile(%out, %c0, %c0) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  ttkernel.matmul_block(%lhs, %rhs, %c0, %c0, %c0, %zero, %one, %one, %one) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  return
}
