// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst{dst-capacity=4096}))' --split-input-file | FileCheck %s

// Capacity is 4096 (large to allow multi-tile grids in this test).
// We chain 5 adds (3 inputs). With capacity 4, reuse must succeed.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @chain_reuse
// CHECK: ttl.compute
// CHECK: ^bb0(%[[ARG0:.*]]: !ttcore.tile<32x32, f32>, %[[ARG1:.*]]: !ttcore.tile<32x32, f32>, %[[ARG2:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[LIN_IDX_0:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[DST0:.*]], %[[TILE0:.*]] = ttl.copy_tile %[[ARG0]], %[[LIN_IDX_0]], %[[C0]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[LIN_IDX_1:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[DST1:.*]], %[[TILE1:.*]] = ttl.copy_tile %[[ARG1]], %[[LIN_IDX_1]], %[[C1]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[X0:.*]] = ttl.tile_add %[[TILE0]], %[[TILE1]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[LIN_IDX_2:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[DST2:.*]], %[[TILE2:.*]] = ttl.copy_tile %[[ARG2]], %[[LIN_IDX_2]], %[[C0_2]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[X1:.*]] = ttl.tile_add %[[X0]], %[[TILE2]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[X2:.*]] = ttl.tile_add %[[X1]], %[[TILE2]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[X3:.*]] = ttl.tile_add %[[X2]], %[[TILE2]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[X4:.*]] = ttl.tile_add %[[X3]], %[[TILE2]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   ttl.yield %[[X4]]

func.func @chain_reuse(%i0: tensor<32x32xf32>, %i1: tensor<32x32xf32>,
                       %i2: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>

  // Bind CBs (omitted for brevity, just attach)
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

  %t0 = ttl.attach_cb %i0, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t1 = ttl.attach_cb %i1, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t2 = ttl.attach_cb %i2, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute
    ins(%t0, %t1, %t2 :
        tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map, #map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>,
       %arg2: !ttcore.tile<32x32, f32>,
       %out: !ttcore.tile<32x32, f32>):

    %x0 = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    %x1 = ttl.tile_add %x0, %arg2 : !ttcore.tile<32x32, f32>
    %x2 = ttl.tile_add %x1, %arg2 : !ttcore.tile<32x32, f32>
    %x3 = ttl.tile_add %x2, %arg2 : !ttcore.tile<32x32, f32>
    %x4 = ttl.tile_add %x3, %arg2 : !ttcore.tile<32x32, f32>

    ttl.yield %x4 : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>

  func.return %res : tensor<32x32xf32>
}

// -----

// Test that multiple uses of the same block argument share a single copy_tile operation.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @block_arg_multi_use
// CHECK: ttl.compute
// CHECK: ^bb0(%[[ARG0:.*]]: !ttcore.tile<32x32, f32>, %[[ARG1:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[LIN_IDX_0:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[COPY0TOK:.*]], %[[COPY0:.*]] = ttl.copy_tile %[[ARG0]], %[[LIN_IDX_0]], %[[C0]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[LIN_IDX_1:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[COPY1TOK:.*]], %[[COPY1:.*]] = ttl.copy_tile %[[ARG1]], %[[LIN_IDX_1]], %[[C1]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[ADD0:.*]] = ttl.tile_add %[[COPY0]], %[[COPY1]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[ADD1:.*]] = ttl.tile_add %[[COPY0]], %[[ADD0]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[ADD2:.*]] = ttl.tile_add %[[COPY0]], %[[ADD1]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   ttl.yield %[[ADD2]]

func.func @block_arg_multi_use(%i0: tensor<32x32xf32>, %i1: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>

  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

  %t0 = ttl.attach_cb %i0, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t1 = ttl.attach_cb %i1, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute
    ins(%t0, %t1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>,
       %out: !ttcore.tile<32x32, f32>):

    // arg0 is used 3 times - should share the same copy_tile
    %x0 = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    %x1 = ttl.tile_add %arg0, %x0 : !ttcore.tile<32x32, f32>
    %x2 = ttl.tile_add %arg0, %x1 : !ttcore.tile<32x32, f32>

    ttl.yield %x2 : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>

  func.return %res : tensor<32x32xf32>
}
