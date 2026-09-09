// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0},func.func(ttl-assign-dst,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,canonicalize,cse)' | FileCheck %s

// Batched transposed matmul keeps independent input offsets and DST slots.
// A=[2,2,3], B=[2,2,3] (stored N,K), output=[2,2,2]. Each batch occupies six
// input tiles and four output tiles; K and N are both nontrivial.

// CHECK-LABEL: func.func @batched_transpose_offsets
// CHECK-DAG: %[[ONE_I32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[THREE_I32:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[THREE:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[FIVE:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[SIX:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[SEVEN:.*]] = arith.constant 7 : index
// CHECK: %[[LHS:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[RHS:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: "ttkernel.mm_block_init"(%[[LHS]], %[[RHS]], %[[OUT]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: ttkernel.tile_regs_acquire
// CHECK: "ttkernel.mm_block_init_short"(%[[LHS]], %[[RHS]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: scf.for %[[K0:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] {
// CHECK: ttkernel.matmul_block(%[[LHS]], %[[RHS]], %[[K0]], %[[K0]], %[[ZERO]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: %[[ROW1:.*]] = arith.addi %[[K0]], %[[THREE]] : index
// CHECK: ttkernel.matmul_block(%[[LHS]], %[[RHS]], %[[K0]], %[[ROW1]], %[[ONE]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: ttkernel.matmul_block(%[[LHS]], %[[RHS]], %[[ROW1]], %[[K0]], %[[TWO]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: ttkernel.matmul_block(%[[LHS]], %[[RHS]], %[[ROW1]], %[[ROW1]], %[[THREE]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: scf.for %[[K1:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] {
// CHECK: %[[BATCH1_ROW0:.*]] = affine.linearize_index [%[[ONE]], %[[K1]]] by (2, 6) : index
// CHECK: ttkernel.matmul_block(%[[LHS]], %[[RHS]], %[[BATCH1_ROW0]], %[[BATCH1_ROW0]], %[[FOUR]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: %[[LOCAL_ROW1:.*]] = arith.addi %[[K1]], %[[THREE]] : index
// CHECK: %[[BATCH1_ROW1:.*]] = affine.linearize_index [%[[ONE]], %[[LOCAL_ROW1]]] by (2, 6) : index
// CHECK: ttkernel.matmul_block(%[[LHS]], %[[RHS]], %[[BATCH1_ROW0]], %[[BATCH1_ROW1]], %[[FIVE]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: ttkernel.matmul_block(%[[LHS]], %[[RHS]], %[[BATCH1_ROW1]], %[[BATCH1_ROW0]], %[[SIX]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: ttkernel.matmul_block(%[[LHS]], %[[RHS]], %[[BATCH1_ROW1]], %[[BATCH1_ROW1]], %[[SEVEN]], %[[ONE_I32]], %[[ONE_I32]], %[[ONE_I32]], %[[THREE_I32]])
// CHECK: ttkernel.tile_regs_commit
// CHECK-NEXT: ttkernel.tile_regs_wait
// CHECK-COUNT-8: ttkernel.pack_tile
// CHECK-NEXT: ttkernel.tile_regs_release
func.func @batched_transpose_offsets(
    %arg0: tensor<2x2x3x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<2x2x3x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x2x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2, 2], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x3x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x3x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x2x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b {transpose_rhs} : tensor<2x2x3x!ttcore.tile<32x32, bf16>>, tensor<2x2x3x!ttcore.tile<32x32, bf16>> -> tensor<2x2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<2x2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x2x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<2x2x2x!ttcore.tile<32x32, bf16>>
}
