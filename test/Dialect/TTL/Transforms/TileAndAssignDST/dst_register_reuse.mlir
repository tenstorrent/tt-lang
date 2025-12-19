// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst{dst-capacity=4}))' --split-input-file | FileCheck %s

// Capacity is 4.
// We chain 5 adds (3 inputs). With capacity 4, reuse must succeed.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @chain_reuse
// CHECK: ttl.compute
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.yield

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
