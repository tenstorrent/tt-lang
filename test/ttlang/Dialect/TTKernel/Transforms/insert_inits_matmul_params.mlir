// mm_block_init_short forwards transpose and the block dimensions to the
// unpack and math inits, so matmuls that share CBs but differ in those
// operands need separate inits and cannot share one hoisted init. Constant
// operands compare by value.
// RUN: ttlang-opt %s --ttkernel-insert-inits | FileCheck %s

// Different ct_dim in one loop: each matmul is initialized inside the loop.
// CHECK-LABEL: func.func @loop_matmul_different_dims
// CHECK-DAG: %[[ONE:[^ ]+]] = arith.constant 1 : i32
// CHECK-DAG: %[[TWO:[^ ]+]] = arith.constant 2 : i32
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NEXT: scf.for
// CHECK-NEXT: "ttkernel.mm_block_init_short"({{.*}}, %[[ONE]], %[[ONE]], %[[ONE]])
// CHECK-NEXT: ttkernel.matmul_block
// CHECK-NEXT: "ttkernel.mm_block_init_short"({{.*}}, %[[TWO]], %[[ONE]], %[[ONE]])
// CHECK-NEXT: ttkernel.matmul_block
// CHECK-NEXT: }
// CHECK: ttkernel.tile_regs_release
func.func @loop_matmul_different_dims() {
  %in0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %in1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %transpose = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  scf.for %i = %c0 to %c4 step %c1 {
    ttkernel.matmul_block(%in0, %in1, %c0, %c0, %c0, %transpose, %one, %one, %one) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
    ttkernel.matmul_block(%in0, %in1, %c0, %c0, %c0, %transpose, %two, %one, %one) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
  }
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// Equal dimensions from distinct constant ops share one init hoisted above the
// loop.
// CHECK-LABEL: func.func @loop_matmul_equal_dims_distinct_constants
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NEXT: "ttkernel.mm_block_init_short"
// CHECK-NEXT: scf.for
// CHECK-NOT: ttkernel.mm_block_init_short
// CHECK: ttkernel.tile_regs_release
func.func @loop_matmul_equal_dims_distinct_constants() {
  %in0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %in1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %transpose = arith.constant 0 : i32
  %one_a = arith.constant 1 : i32
  %one_b = arith.constant 1 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  scf.for %i = %c0 to %c4 step %c1 {
    ttkernel.matmul_block(%in0, %in1, %c0, %c0, %c0, %transpose, %one_a, %one_a, %one_a) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
    ttkernel.matmul_block(%in0, %in1, %c0, %c0, %c0, %transpose, %one_b, %one_b, %one_b) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
  }
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}

// The loop's ct_dim is defined in the body, so its init stays in the loop
// instead of being hoisted above that definition.
// CHECK-LABEL: func.func @loop_matmul_loop_local_dims
// CHECK: ttkernel.tile_regs_acquire
// CHECK-NEXT: "ttkernel.mm_block_init_short"
// CHECK-NEXT: ttkernel.matmul_block
// CHECK-NEXT: scf.for
// CHECK-NEXT: %[[CT:[^ ]+]] = arith.constant 2 : i32
// CHECK-NEXT: "ttkernel.mm_block_init_short"({{.*}}, %[[CT]], {{.*}})
// CHECK-NEXT: ttkernel.matmul_block
// CHECK-NEXT: }
// CHECK: ttkernel.tile_regs_release
func.func @loop_matmul_loop_local_dims() {
  %in0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %in1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %out = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %transpose = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  ttkernel.tile_regs_acquire() : () -> ()
  ttkernel.matmul_block(%in0, %in1, %c0, %c0, %c0, %transpose, %one, %one, %one) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
  scf.for %i = %c0 to %c4 step %c1 {
    %two = arith.constant 2 : i32
    ttkernel.matmul_block(%in0, %in1, %c0, %c0, %c0, %transpose, %two, %one, %one) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index, i32, i32, i32, i32) -> ()
  }
  ttkernel.pack_tile(%c0, %out, %c0, false) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
  ttkernel.tile_regs_release() : () -> ()
  func.return
}
