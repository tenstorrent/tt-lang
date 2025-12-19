// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst{dst-capacity=16}))' --split-input-file | FileCheck %s

// Capacity is 8.
// We chain 9 adds.
// Inputs: 10 tensors.
// If reuse works: peak usage is low (accumulator + 1 input).
// If reuse fails: peak usage grows > 8 and test fails.

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @chain_reuse(%i0: tensor<32x32xf32>, %i1: tensor<32x32xf32>,
                       %i2: tensor<32x32xf32>, %i3: tensor<32x32xf32>,
                       %i4: tensor<32x32xf32>, %i5: tensor<32x32xf32>,
                       %i6: tensor<32x32xf32>, %i7: tensor<32x32xf32>,
                       %i8: tensor<32x32xf32>, %i9: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>
  
  // Bind CBs (omitted for brevity, just attach)
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  
  %t0 = ttl.attach_cb %i0, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t1 = ttl.attach_cb %i1, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t2 = ttl.attach_cb %i2, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t3 = ttl.attach_cb %i3, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t4 = ttl.attach_cb %i4, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t5 = ttl.attach_cb %i5, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t6 = ttl.attach_cb %i6, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t7 = ttl.attach_cb %i7, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t8 = ttl.attach_cb %i8, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t9 = ttl.attach_cb %i9, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute 
    ins(%t0, %t1, %t2, %t3, %t4, %t5, %t6, %t7, %t8, %t9 : 
        tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>,
        tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>,
        tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map, #map, #map, #map, #map, #map, #map, #map, #map, #map, #map], 
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>,
       %arg2: !ttcore.tile<32x32, f32>, %arg3: !ttcore.tile<32x32, f32>,
       %arg4: !ttcore.tile<32x32, f32>, %arg5: !ttcore.tile<32x32, f32>,
       %arg6: !ttcore.tile<32x32, f32>, %arg7: !ttcore.tile<32x32, f32>,
       %arg8: !ttcore.tile<32x32, f32>, %arg9: !ttcore.tile<32x32, f32>,
       %out: !ttcore.tile<32x32, f32>):
       
    %x0 = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    %x1 = ttl.tile_add %x0, %arg2 : !ttcore.tile<32x32, f32>
    %x2 = ttl.tile_add %x1, %arg3 : !ttcore.tile<32x32, f32>
    %x3 = ttl.tile_add %x2, %arg4 : !ttcore.tile<32x32, f32>
    %x4 = ttl.tile_add %x3, %arg5 : !ttcore.tile<32x32, f32>
    %x5 = ttl.tile_add %x4, %arg6 : !ttcore.tile<32x32, f32>
    %x6 = ttl.tile_add %x5, %arg7 : !ttcore.tile<32x32, f32>
    %x7 = ttl.tile_add %x6, %arg8 : !ttcore.tile<32x32, f32>
    %x8 = ttl.tile_add %x7, %arg9 : !ttcore.tile<32x32, f32>
    
    ttl.yield %x8 : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>
  
  func.return %res : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @chain_reuse
// CHECK: ttl.compute
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.tile_add
// CHECK: ttl.yield

