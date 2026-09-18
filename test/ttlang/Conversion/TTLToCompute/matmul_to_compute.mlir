// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' --split-input-file | FileCheck %s

// Matmul lowered to ttl.compute with tile_matmul_block.
// 3D iteration space [M, N, K] with matmul indexing maps.

// CHECK-DAG: #[[$LHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$RHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$OUT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @matmul_1x1_bf16
func.func @matmul_1x1_bf16(
    %arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // CHECK: ttl.compute
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK: ttl.tile_matmul_block
  // CHECK: ttl.tile_store
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// CHECK-LABEL: func.func @matmul_1x1_f32
func.func @matmul_1x1_f32(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.compute
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK: ttl.tile_matmul_block
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %mm, %reserve : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %mm : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// The Kimi 1x32 BF16 activation/output tiles and 32x32 BFP4_B weight tiles
// retain their distinct formats through lowering.
// CHECK-LABEL: func.func @matmul_bf16_bfp4
func.func @matmul_bf16_bfp4(
    %arg0: tensor<1x2x!ttcore.tile<1x32, bf16>>,
    %arg1: tensor<2x2x!ttcore.tile<32x32, bfp_bf4>>)
    -> tensor<1x2x!ttcore.tile<1x32, bf16>> {
  // CHECK: ttl.compute
  // CHECK-SAME: ins({{.*}} : tensor<1x2x!ttcore.tile<1x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bfp_bf4>>)
  // CHECK-SAME: outs({{.*}} : tensor<1x2x!ttcore.tile<1x32, bf16>>)
  // CHECK: ^bb0(%[[LHS:.*]]: !ttcore.tile<1x32, bf16>, %[[RHS:.*]]: !ttcore.tile<32x32, bfp_bf4>
  // CHECK: %[[MM:.*]] = ttl.tile_matmul_block %[[LHS]], %[[RHS]]
  // CHECK-SAME: !ttcore.tile<1x32, bf16>, !ttcore.tile<32x32, bfp_bf4> -> !ttcore.tile<1x32, bf16>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bfp_bf4>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0
      : (tensor<1x2x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<1x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1
      : (tensor<2x2x!ttcore.tile<32x32, bfp_bf4>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bfp_bf4>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, bfp_bf4>>
  %reserve = ttl.cb_reserve %cb2
      : <[1, 2], !ttcore.tile<1x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<1x32, bf16>>
  %mm = ttl.matmul %a, %b
      : tensor<1x2x!ttcore.tile<1x32, bf16>>,
        tensor<2x2x!ttcore.tile<32x32, bfp_bf4>>
        -> tensor<1x2x!ttcore.tile<1x32, bf16>>
  ttl.store %mm, %reserve
      : tensor<1x2x!ttcore.tile<1x32, bf16>>,
        tensor<1x2x!ttcore.tile<1x32, bf16>>
  func.return %mm : tensor<1x2x!ttcore.tile<1x32, bf16>>
}

// -----

// The Kimi 1x32 BF16 activation/output tiles and 32x32 BFP8_B weight tiles
// retain their distinct formats through lowering.
// CHECK-LABEL: func.func @matmul_bf16_bfp8
func.func @matmul_bf16_bfp8(
    %arg0: tensor<1x2x!ttcore.tile<1x32, bf16>>,
    %arg1: tensor<2x2x!ttcore.tile<32x32, bfp_bf8>>)
    -> tensor<1x2x!ttcore.tile<1x32, bf16>> {
  // CHECK: ttl.compute
  // CHECK-SAME: ins({{.*}} : tensor<1x2x!ttcore.tile<1x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bfp_bf8>>)
  // CHECK-SAME: outs({{.*}} : tensor<1x2x!ttcore.tile<1x32, bf16>>)
  // CHECK: ^bb0(%[[LHS:.*]]: !ttcore.tile<1x32, bf16>, %[[RHS:.*]]: !ttcore.tile<32x32, bfp_bf8>
  // CHECK: %[[MM:.*]] = ttl.tile_matmul_block %[[LHS]], %[[RHS]]
  // CHECK-SAME: !ttcore.tile<1x32, bf16>, !ttcore.tile<32x32, bfp_bf8> -> !ttcore.tile<1x32, bf16>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bfp_bf8>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0
      : (tensor<1x2x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<1x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<1x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1
      : (tensor<2x2x!ttcore.tile<32x32, bfp_bf8>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bfp_bf8>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, bfp_bf8>>
  %reserve = ttl.cb_reserve %cb2
      : <[1, 2], !ttcore.tile<1x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<1x32, bf16>>
  %mm = ttl.matmul %a, %b
      : tensor<1x2x!ttcore.tile<1x32, bf16>>,
        tensor<2x2x!ttcore.tile<32x32, bfp_bf8>>
        -> tensor<1x2x!ttcore.tile<1x32, bf16>>
  ttl.store %mm, %reserve
      : tensor<1x2x!ttcore.tile<1x32, bf16>>,
        tensor<1x2x!ttcore.tile<1x32, bf16>>
  func.return %mm : tensor<1x2x!ttcore.tile<1x32, bf16>>
}

// -----

// Non-square [2,4] @ [4,3] -> [2,3].
// CHECK-LABEL: func.func @matmul_2x4_4x3
func.func @matmul_2x4_4x3(
    %arg0: tensor<2x4x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<4x3x!ttcore.tile<32x32, bf16>>) -> tensor<2x3x!ttcore.tile<32x32, bf16>> {
  // CHECK: ttl.compute
  // CHECK-SAME: ins({{.*}} : tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<4x3x!ttcore.tile<32x32, bf16>>)
  // CHECK-SAME: outs({{.*}} : tensor<2x3x!ttcore.tile<32x32, bf16>>)
  // CHECK-SAME: indexing_maps = [#[[$LHS_MAP]], #[[$RHS_MAP]], #[[$OUT_MAP]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK: ttl.tile_matmul_block
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<4x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x3x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<4x3x!ttcore.tile<32x32, bf16>> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<2x3x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<2x3x!ttcore.tile<32x32, bf16>>
}

// -----

// [1,8] @ [8,1] -> [1,1]. Large K.
// CHECK-LABEL: func.func @matmul_1x8_8x1
func.func @matmul_1x8_8x1(
    %arg0: tensor<1x8x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<8x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // CHECK: ttl.compute
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK: ttl.tile_matmul_block
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[8, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x8x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x8x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<8x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[8, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<8x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x8x!ttcore.tile<32x32, bf16>>, tensor<8x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Matmul preserves distinct physical tile dimensions through a multi-tile
// block: [1x2] of 4x32 tiles times [2x2] of 32x32 tiles produces [1x2] of
// 4x32 tiles.
// CHECK-LABEL: func.func @matmul_subtile_multi_tile
func.func @matmul_subtile_multi_tile(
    %arg0: tensor<1x2x!ttcore.tile<4x32, bf16>>,
    %arg1: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x2x!ttcore.tile<4x32, bf16>> {
  // CHECK: ttl.compute
  // CHECK-SAME: ins({{.*}} : tensor<1x2x!ttcore.tile<4x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>)
  // CHECK-SAME: outs({{.*}} : tensor<1x2x!ttcore.tile<4x32, bf16>>)
  // CHECK: ^bb0(%[[LHS:.*]]: !ttcore.tile<4x32, bf16>, %[[RHS:.*]]: !ttcore.tile<32x32, bf16>, %{{.*}}: !ttcore.tile<4x32, bf16>):
  // CHECK: %[[MM:.*]] = ttl.tile_matmul_block %[[LHS]], %[[RHS]]
  // CHECK-SAME: !ttcore.tile<4x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<4x32, bf16>
  // CHECK: ttl.tile_store %[[MM]]
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<4x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<4x32, bf16>, 2>
  %lhs = ttl.attach_cb %arg0, %lhs_dfb
      : (tensor<1x2x!ttcore.tile<4x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<4x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<4x32, bf16>>
  %rhs = ttl.attach_cb %arg1, %rhs_dfb
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2], !ttcore.tile<4x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<4x32, bf16>>
  %result = ttl.matmul %lhs, %rhs
      : tensor<1x2x!ttcore.tile<4x32, bf16>>,
        tensor<2x2x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x!ttcore.tile<4x32, bf16>>
  ttl.store %result, %output
      : tensor<1x2x!ttcore.tile<4x32, bf16>>,
        tensor<1x2x!ttcore.tile<4x32, bf16>>
  func.return %result : tensor<1x2x!ttcore.tile<4x32, bf16>>
}

// -----

// Transposed RHS: [2,4] @ [3,4]^T -> [2,3]. RHS is stored as [N, K] so its
// indexing map swaps to (d1, d2) and the transpose_rhs attr is propagated to
// tile_matmul_block.
// CHECK: #[[$RHS_T_MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-LABEL: func.func @matmul_transpose_rhs
func.func @matmul_transpose_rhs(
    %arg0: tensor<2x4x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<3x4x!ttcore.tile<32x32, bf16>>) -> tensor<2x3x!ttcore.tile<32x32, bf16>> {
  // CHECK: ttl.compute
  // CHECK-SAME: ins({{.*}} : tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<3x4x!ttcore.tile<32x32, bf16>>)
  // CHECK-SAME: outs({{.*}} : tensor<2x3x!ttcore.tile<32x32, bf16>>)
  // CHECK-SAME: indexing_maps = [#[[$LHS_MAP]], #[[$RHS_T_MAP]], #[[$OUT_MAP]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK: ttl.tile_matmul_block
  // CHECK-SAME: transpose_rhs
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[3, 4], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<3x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x4x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b {transpose_rhs} : tensor<2x4x!ttcore.tile<32x32, bf16>>, tensor<3x4x!ttcore.tile<32x32, bf16>> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<2x3x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<2x3x!ttcore.tile<32x32, bf16>>
}
