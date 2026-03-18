// Summary: End-to-end test for matmul_block lowering through the full pipeline.
// matmul_block handles K internally and writes M*N DST registers in one call.
// No per-tile iteration — the compute body is a single matmul_block + pack.

// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-insert-tile-regs-sync, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
// RUN:   --split-input-file | FileCheck %s

// =============================================================================
// Test 1: 1x1 bf16.
// =============================================================================

// CHECK-LABEL: func.func @matmul_1x1_bf16
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK-DAG: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
// CHECK:     "ttkernel.mm_block_init"(%[[CB0]], %[[CB1]], %[[CB2]], %[[C0_I32]], %[[C1_I32]], %[[C1_I32]], %[[C1_I32]])
// CHECK:     ttkernel.tile_regs_acquire
// CHECK-NEXT: "ttkernel.mm_block_init_short"(%[[CB0]], %[[CB1]], %[[C0_I32]], %[[C1_I32]], %[[C1_I32]], %[[C1_I32]])
// CHECK-NEXT: "ttkernel.experimental::matmul_block"(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]], %[[C0_I32]], %[[C1_I32]], %[[C1_I32]], %[[C1_I32]], %[[C1_I32]])
// CHECK-NEXT: ttkernel.tile_regs_commit
// CHECK-NEXT: ttkernel.tile_regs_wait
// CHECK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]]
// CHECK-NEXT: ttkernel.tile_regs_release
func.func @matmul_1x1_bf16(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// =============================================================================
// Test 2: [2,4] @ [4,3] -> [2,3]. Output 6 tiles, fits in DST (capacity 8).
// No loops — matmul_block handles K=4 internally.
// Block dims: rt=2, ct=3, kt=4, nt=3.
// CB tile indices are 0 (start of block, no per-tile iteration).
// =============================================================================

// CHECK-LABEL: func.func @matmul_2x4_4x3
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C2_I32:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[C3_I32:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[C4_I32:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<16, !ttcore.tile<32x32, bf16>>
// CHECK-DAG: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<24, !ttcore.tile<32x32, bf16>>
// CHECK-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, bf16>>
// mm_block_init with block dims rt=2, ct=3, kt=4.
// CHECK:     "ttkernel.mm_block_init"(%[[CB0]], %[[CB1]], %[[CB2]], %[[C0_I32]], %[[C3_I32]], %[[C2_I32]], %[[C4_I32]])
// CHECK:     ttkernel.tile_regs_acquire
// No loops — single matmul_block call.
// CHECK-NOT: scf.for
// CHECK:     "ttkernel.experimental::matmul_block"(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]], %[[C0_I32]], %[[C3_I32]], %[[C2_I32]], %[[C4_I32]], %[[C3_I32]])
// CHECK:     ttkernel.tile_regs_commit
// CHECK-NEXT: ttkernel.tile_regs_wait
// TODO: replace M*N individual pack_tile calls with pack_tile_block.
// CHECK:     ttkernel.pack_tile
// CHECK:     ttkernel.tile_regs_release
func.func @matmul_2x4_4x3(
    %arg0: tensor<2x4x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<4x3x!ttcore.tile<32x32, bf16>>) -> tensor<2x3x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<4x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x3x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<4x3x!ttcore.tile<32x32, bf16>> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<2x3x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<2x3x!ttcore.tile<32x32, bf16>>
}
