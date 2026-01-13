// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-tile-and-assign-dst{dst-capacity=64}),cse,canonicalize)' | FileCheck %s

// Basic elementwise operations lowered to ttl.compute with tile ops and DST assignment.
// Input provides explicit bind_cb and attach_cb ops; pass creates compute.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @binary_add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @binary_add(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>, %arg1: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[CB0:.*]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.*]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[CB2:.*]] = ttl.bind_cb{cb_index = 2
  // CHECK-NEXT: %[[ARG0_CB:.*]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.*]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK-NEXT: %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CB2]]
  // CHECK-NEXT: %[[RESULT:.*]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]{{.*}}} {
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: !ttcore.tile<32x32, f32>, %[[RHS:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[LIN_IDX:.*]] = ttl.linearized_index
  // CHECK-NEXT:   %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[LHS]], %[[LIN_IDX]], %[[C0]]
  // CHECK-NEXT:   %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[RHS]], %[[LIN_IDX]], %[[C1]]
  // CHECK-NEXT:   %[[SUM:.*]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] {dst_idx = 2 : i32}
  // CHECK-NEXT:   ttl.yield %[[SUM]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %a, %b : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @unary_exp
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_exp(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[CB0:.*]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.*]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.*]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK-NEXT: %[[RESULT:.*]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]{{.*}}} {
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[LIN_IDX:.*]] = ttl.linearized_index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]], %[[LIN_IDX]], %[[C0]]
  // CHECK-NEXT:   %[[EXP:.*]] = ttl.tile_exp %[[DTILE]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   ttl.yield %[[EXP]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.exp %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @chain_binary_unary
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @chain_binary_unary(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>, %arg1: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[CB0:.*]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.*]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[CB2:.*]] = ttl.bind_cb{cb_index = 2
  // CHECK-NEXT: %[[CB3:.*]] = ttl.bind_cb{cb_index = 3
  // CHECK-NEXT: %[[ARG0_CB:.*]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.*]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // First compute: binary add
  // CHECK-NEXT: %[[INIT0_CB:.*]] = ttl.attach_cb %[[EMPTY]], %[[CB2]]
  // CHECK-NEXT: %[[ADD_RESULT:.*]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : {{.*}}) outs(%[[INIT0_CB]] : {{.*}}) {
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: !ttcore.tile<32x32, f32>, %[[RHS:.*]]: !ttcore.tile<32x32, f32>, %[[OUT0:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[LIN_IDX:.*]] = ttl.linearized_index
  // CHECK-NEXT:   %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[LHS]], %[[LIN_IDX]], %[[C0]]
  // CHECK-NEXT:   %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[RHS]], %[[LIN_IDX]], %[[C1]]
  // CHECK-NEXT:   %[[SUM:.*]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] {dst_idx = 2 : i32}
  // CHECK-NEXT:   ttl.yield %[[SUM]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %a, %b : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK-NEXT: %[[ADD_RESULT_CB:.*]] = ttl.attach_cb %[[ADD_RESULT]], %[[CB2]]
  %add_cb = ttl.attach_cb %0, %cb2 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // Second compute: unary relu
  // CHECK-NEXT: %[[INIT1_CB:.*]] = ttl.attach_cb %[[EMPTY]], %[[CB3]]
  // CHECK-NEXT: %[[RELU_RESULT:.*]] = ttl.compute ins(%[[ADD_RESULT_CB]] : {{.*}}) outs(%[[INIT1_CB]] : {{.*}}) {
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT1:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[LIN_IDX:.*]] = ttl.linearized_index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]], %[[LIN_IDX]], %[[C0]]
  // CHECK-NEXT:   %[[ACT:.*]] = ttl.tile_relu %[[DTILE]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   ttl.yield %[[ACT]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RELU_RESULT]]
  %1 = ttl.relu %add_cb : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @multiple_binary
// CHECK-SAME: (%[[A:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[B:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[C:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @multiple_binary(%a: tensor<4x4x!ttcore.tile<32x32, f32>>, %b: tensor<4x4x!ttcore.tile<32x32, f32>>, %c: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  // CHECK: %[[A_CB:.*]] = ttl.attach_cb %[[A]]
  // CHECK: %[[B_CB:.*]] = ttl.attach_cb %[[B]]
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[ADD:.*]] = ttl.compute
  // CHECK: ttl.tile_add{{.*}}dst_idx = 2
  %0 = ttl.add %a_cb, %b_cb : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[ADD_CB:.*]] = ttl.attach_cb %[[ADD]]
  // CHECK: %[[C_CB:.*]] = ttl.attach_cb %[[C]]
  %add_cb = ttl.attach_cb %0, %cb2 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb3 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_mul{{.*}}dst_idx = 2
  %1 = ttl.mul %add_cb, %c_cb : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @unary_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_chain(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  // CHECK: %[[ARG0_CB:.*]] = ttl.attach_cb %[[ARG0]]
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[ABS:.*]] = ttl.compute
  // CHECK: ttl.tile_abs{{.*}}dst_idx = 0
  %0 = ttl.abs %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[ABS_CB:.*]] = ttl.attach_cb %[[ABS]]
  %abs_cb = ttl.attach_cb %0, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[SQRT:.*]] = ttl.compute
  // CHECK: ttl.tile_sqrt{{.*}}dst_idx = 0
  %1 = ttl.sqrt %abs_cb : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[SQRT_CB:.*]] = ttl.attach_cb %[[SQRT]]
  %sqrt_cb = ttl.attach_cb %1, %cb2 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_log{{.*}}dst_idx = 0
  %2 = ttl.log %sqrt_cb : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %2 : tensor<4x4x!ttcore.tile<32x32, f32>>
}
