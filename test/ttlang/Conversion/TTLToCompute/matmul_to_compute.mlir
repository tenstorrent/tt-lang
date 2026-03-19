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

// CHECK-LABEL: func.func @matmul_1x1_f32
func.func @matmul_1x1_f32(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.compute
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK: ttl.tile_matmul_block
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %mm = ttl.matmul %a, %b : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %mm, %reserve : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %mm : tensor<1x1x!ttcore.tile<32x32, f32>>
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

// [1,8] @ [8,1] -> [1,1]. Large K.
// CHECK-LABEL: func.func @matmul_1x8_8x1
func.func @matmul_1x8_8x1(
    %arg0: tensor<1x8x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<8x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // CHECK: ttl.compute
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK: ttl.tile_matmul_block
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[8, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x8x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x8x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<8x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[8, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<8x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<1x8x!ttcore.tile<32x32, bf16>>, tensor<8x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %mm : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
