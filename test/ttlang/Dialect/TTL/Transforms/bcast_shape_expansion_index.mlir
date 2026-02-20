// Tests that broadcast with shape expansion computes correct input CB tile
// indices after tile loop unrolling.
//
// Regression: computeBcastShapeExpansionIndex must NOT generate arith.divui
// or arith.remui (ConvertTTKernelToEmitC cannot lower them). Instead, it
// computes row/col components directly from constant strides.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttcore-register-device,func.func(convert-ttl-to-compute,ttl-set-compute-kernel-config,ttl-assign-dst,ttl-subblock-compute-for-dst,ttl-insert-tile-regs-sync,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel)' --split-input-file | FileCheck %s

// Col broadcast with shape expansion: input CB (2,1), output CB (2,2).
// 4 output tiles, input indices should be:
//   tile 0 (row=0,col=0): linearized=0, 0/2=0
//   tile 1 (row=0,col=1): linearized=1, 1/2=0
//   tile 2 (row=1,col=0): linearized=2, 2/2=1
//   tile 3 (row=1,col=1): linearized=3, 3/2=1
// CHECK-LABEL: func.func @col_bcast_shape_expansion
// No runtime division/remainder (must use compile-time stride decomposition):
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// Input CB indices: 0, 0, 1, 1 (row index = linearized / numCols)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C0]], %[[C0]], <col>)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C0]], %[[C1]], <col>)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C1]], %[[C2]], <col>)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C1]], %[[C3]], <col>)
module {
  func.func @col_bcast_shape_expansion() attributes {ttl.base_cta_index = 6 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[2, 1], !ttcore.tile<32x32, f32>, 2>
    %cb1 = ttl.bind_cb{cb_index = 16, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, f32>, 2>
    %in = ttl.cb_wait %cb0 : <[2, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %in_cb = ttl.attach_cb %in, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %out = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %out_cb = ttl.attach_cb %out, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    // Col bcast: BcastType::Col = 1
    %result = ttl.bcast %in_cb, %out_cb 1 : i32 : (tensor<2x1x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %result, %out_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
    %result_cb = ttl.attach_cb %result, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2>
    ttl.cb_pop %cb0 : <[2, 1], !ttcore.tile<32x32, f32>, 2>
    return
  }
}

// -----

// Row broadcast with shape expansion: input CB (1,2), output CB (2,2).
// 4 output tiles, input indices should be:
//   tile 0 (row=0,col=0): linearized=0, 0%2=0
//   tile 1 (row=0,col=1): linearized=1, 1%2=1
//   tile 2 (row=1,col=0): linearized=2, 2%2=0
//   tile 3 (row=1,col=1): linearized=3, 3%2=1
// CHECK-LABEL: func.func @row_bcast_shape_expansion
// No runtime division/remainder:
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// Input CB indices: 0, 1, 0, 1 (col index = linearized % numCols)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C0]], %[[C0]], <row>)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C1]], %[[C1]], <row>)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C0]], %[[C2]], <row>)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C1]], %[[C3]], <row>)
module {
  func.func @row_bcast_shape_expansion() attributes {ttl.base_cta_index = 6 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 2], !ttcore.tile<32x32, f32>, 2>
    %cb1 = ttl.bind_cb{cb_index = 16, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, f32>, 2>
    %in = ttl.cb_wait %cb0 : <[1, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %in_cb = ttl.attach_cb %in, %cb0 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %out = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %out_cb = ttl.attach_cb %out, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    // Row bcast: BcastType::Row = 2
    %result = ttl.bcast %in_cb, %out_cb 2 : i32 : (tensor<1x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %result, %out_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
    %result_cb = ttl.attach_cb %result, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2>
    ttl.cb_pop %cb0 : <[1, 2], !ttcore.tile<32x32, f32>, 2>
    return
  }
}

// -----

// Scalar broadcast with shape expansion: input CB (1,1), output CB (2,2).
// All 4 output tiles should use input index 0.
// CHECK-LABEL: func.func @scalar_bcast_shape_expansion
// No runtime division/remainder:
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// All input CB indices are 0:
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C0]], %[[C0]], <scalar>)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C0]], %[[C1]], <scalar>)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C0]], %[[C2]], <scalar>)
// CHECK: ttkernel.unary_bcast(%{{.*}}, %[[C0]], %[[C3]], <scalar>)
module {
  func.func @scalar_bcast_shape_expansion() attributes {ttl.base_cta_index = 6 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %cb1 = ttl.bind_cb{cb_index = 16, buffer_factor = 2} : <[2, 2], !ttcore.tile<32x32, f32>, 2>
    %in = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %in_cb = ttl.attach_cb %in, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %out = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %out_cb = ttl.attach_cb %out, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    // Scalar bcast: BcastType::Scalar = 3
    %result = ttl.bcast %in_cb, %out_cb 3 : i32 : (tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %result, %out_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
    %result_cb = ttl.attach_cb %result, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    return
  }
}
