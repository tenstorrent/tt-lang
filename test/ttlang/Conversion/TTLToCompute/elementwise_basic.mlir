// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst-registers),canonicalize)' | FileCheck %s

// Basic elementwise operations lowered to ttl.compute with tile ops and DST assignment.
// Input provides explicit bind_cb and attach_cb ops; pass creates compute.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @binary_add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @binary_add(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>, %arg1: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // Input provides CBs and attachments
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CB2:.*]]
  // CHECK: %[[RESULT:.*]] = ttl.compute ins(%{{.*}}, %{{.*}} : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: !ttcore.tile<32x32, f32>, %[[RHS:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[SUM:.*]] = ttl.tile_add %[[LHS]], %[[RHS]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   ttl.yield %[[SUM]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[RESULT]]
  %0 = ttl.add %a, %b : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @unary_exp
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @unary_exp(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // Input provides CBs and attachments
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CB1:.*]]
  // CHECK: %[[RESULT:.*]] = ttl.compute ins(%{{.*}} : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[EXP:.*]] = ttl.tile_exp %[[IN]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   ttl.yield %[[EXP]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[RESULT]]
  %0 = ttl.exp %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @chain_binary_unary
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @chain_binary_unary(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>, %arg1: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[ADD_RESULT:.*]] = ttl.compute
  // CHECK: ttl.tile_add{{.*}}dst_idx = 0
  %0 = ttl.add %a, %b : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %add_cb = ttl.attach_cb %0, %cb2 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[RELU_RESULT:.*]] = ttl.compute
  // CHECK: ttl.tile_relu{{.*}}dst_idx = 0
  // CHECK: return %[[RELU_RESULT]]
  %1 = ttl.relu %add_cb : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @multiple_binary
// CHECK-SAME: (%[[A:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[B:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[C:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @multiple_binary(%a: tensor<4x4x!ttcore.tile<32x32, f32>>, %b: tensor<4x4x!ttcore.tile<32x32, f32>>, %c: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[ADD:.*]] = ttl.compute
  // CHECK: ttl.tile_add{{.*}}dst_idx = 0
  %0 = ttl.add %a_cb, %b_cb : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %add_cb = ttl.attach_cb %0, %cb2 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb3 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_mul{{.*}}dst_idx = 0
  %1 = ttl.mul %add_cb, %c_cb : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @unary_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @unary_chain(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[ABS:.*]] = ttl.compute
  // CHECK: ttl.tile_abs{{.*}}dst_idx = 0
  %0 = ttl.abs %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %abs_cb = ttl.attach_cb %0, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[SQRT:.*]] = ttl.compute
  // CHECK: ttl.tile_sqrt{{.*}}dst_idx = 0
  %1 = ttl.sqrt %abs_cb : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %sqrt_cb = ttl.attach_cb %1, %cb2 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_log{{.*}}dst_idx = 0
  %2 = ttl.log %sqrt_cb : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %2 : tensor<4x4x!ttcore.tile<32x32, f32>>
}
