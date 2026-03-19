// Summary: End-to-end test for the flat matmul_block lowering pattern.
// The lower-matmul-block pass replaces the compute with:
// sync acquire -> matmul_block(kt=1) -> commit -> wait -> M*N pack_tiles -> release.
// CB lifecycle (wait/pop/reserve/push) comes from user code, not the pass.

// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute, ttl-assign-dst{enable-fpu-binary-ops=0}, ttl-insert-tile-regs-sync, ttl-lower-matmul-block, ttl-lower-to-loops, ttl-annotate-cb-associations), convert-ttl-to-ttkernel, ttkernel-insert-inits, canonicalize, cse)' \
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
// CHECK:      "ttkernel.mm_block_init"(%[[CB0]], %[[CB1]], %[[CB2]], %[[C0_I32]], %[[C1_I32]], %[[C1_I32]], %[[C1_I32]])
// CHECK:      ttkernel.tile_regs_acquire
// CHECK-NEXT: "ttkernel.mm_block_init_short"(%[[CB0]], %[[CB1]], %[[C0_I32]], %[[C1_I32]], %[[C1_I32]], %[[C1_I32]])
// CHECK-NEXT: "ttkernel.experimental::matmul_block"(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]], %[[C0_I32]], %[[C1_I32]], %[[C1_I32]], %[[C1_I32]], %[[C1_I32]])
// CHECK-NEXT: ttkernel.tile_regs_commit
// CHECK-NEXT: ttkernel.tile_regs_wait
// CHECK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]]
// CHECK-NEXT: ttkernel.tile_regs_release
// No K-loop (single matmul_block call).
// CHECK-NOT:  scf.for
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
// Test 2: [2,4] @ [4,3] -> [2,3]. Single matmul_block, 6 pack_tiles, no loops.
// =============================================================================

// CHECK-LABEL: func.func @matmul_2x4_4x3
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C2_I32:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[C3_I32:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<16, !ttcore.tile<32x32, bf16>>
// CHECK-DAG: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<24, !ttcore.tile<32x32, bf16>>
// CHECK-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, bf16>>
//
// mm_block_init: ct=3, rt=2, kt=1.
// CHECK:      "ttkernel.mm_block_init"(%[[CB0]], %[[CB1]], %[[CB2]], %[[C0_I32]], %[[C3_I32]], %[[C2_I32]], %[[C1_I32]])
// CHECK:      ttkernel.tile_regs_acquire
// CHECK-NEXT: "ttkernel.mm_block_init_short"(%[[CB0]], %[[CB1]], %[[C0_I32]], %[[C3_I32]], %[[C2_I32]], %[[C1_I32]])
// CHECK-NEXT: "ttkernel.experimental::matmul_block"(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]], %[[C0_I32]], %[[C3_I32]], %[[C2_I32]], %[[C1_I32]], %[[C3_I32]])
// CHECK-NEXT: ttkernel.tile_regs_commit
// CHECK-NEXT: ttkernel.tile_regs_wait
// 6 pack_tiles: DST[0..5] -> CB2[0..5].
// CHECK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]]
// CHECK-NEXT: ttkernel.pack_tile(%[[C1]], %[[CB2]], %[[C1]]
// CHECK-NEXT: ttkernel.pack_tile(%[[C2]], %[[CB2]], %[[C2]]
// CHECK-NEXT: ttkernel.pack_tile(%[[C3]], %[[CB2]], %[[C3]]
// CHECK-NEXT: ttkernel.pack_tile(%[[C4]], %[[CB2]], %[[C4]]
// CHECK-NEXT: ttkernel.pack_tile(%[[C5]], %[[CB2]], %[[C5]]
// CHECK-NEXT: ttkernel.tile_regs_release
// No loops.
// CHECK-NOT:  scf.for
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

// -----

// =============================================================================
// Test 3: 2x2 f32. DST capacity is 4 for f32, so 2x2=4 fits exactly.
// =============================================================================

// CHECK-LABEL: func.func @matmul_2x2_f32
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C2_I32:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[CB0:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
// CHECK-DAG: %[[CB1:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f32>>
// CHECK-DAG: %[[CB2:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
// CHECK:      "ttkernel.mm_block_init"(%[[CB0]], %[[CB1]], %[[CB2]], %[[C0_I32]], %[[C2_I32]], %[[C2_I32]], %[[C1_I32]])
// CHECK:      ttkernel.tile_regs_acquire
// CHECK-NEXT: "ttkernel.mm_block_init_short"(%[[CB0]], %[[CB1]], %[[C0_I32]], %[[C2_I32]], %[[C2_I32]], %[[C1_I32]])
// CHECK-NEXT: "ttkernel.experimental::matmul_block"(%[[CB0]], %[[CB1]], %[[C0]], %[[C0]], %[[C0]], %[[C0_I32]], %[[C2_I32]], %[[C2_I32]], %[[C1_I32]], %[[C2_I32]])
// CHECK-NEXT: ttkernel.tile_regs_commit
// CHECK-NEXT: ttkernel.tile_regs_wait
// CHECK-NEXT: ttkernel.pack_tile(%[[C0]], %[[CB2]], %[[C0]]
// CHECK-NEXT: ttkernel.pack_tile(%[[C1]], %[[CB2]], %[[C1]]
// CHECK-NEXT: ttkernel.pack_tile(%[[C2]], %[[CB2]], %[[C2]]
// CHECK-NEXT: ttkernel.pack_tile(%[[C3]], %[[CB2]], %[[C3]]
// CHECK-NEXT: ttkernel.tile_regs_release
// CHECK-NOT:  scf.for
func.func @matmul_2x2_f32(
    %arg0: tensor<2x1x!ttcore.tile<32x32, f32>>,
    %arg1: tensor<1x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %mm = ttl.matmul %a, %b : tensor<2x1x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %mm, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %mm : tensor<2x2x!ttcore.tile<32x32, f32>>
}
