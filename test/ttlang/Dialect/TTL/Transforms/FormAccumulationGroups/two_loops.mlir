// Summary: Two separate user loops both accumulate to the same view.
// Both loops' computes get the same accumulation group ID.
//
// RUN: ttlang-opt %s -ttl-form-accumulation-groups | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @two_loops
// First loop compute.
// CHECK: scf.for
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 0
// CHECK: ttl.tile_store
// CHECK-SAME: acc = true
// Second loop compute.
// CHECK: scf.for
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 0
// CHECK: ttl.tile_store
// CHECK-SAME: acc = true

func.func @two_loops(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                      %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // First accumulation loop.
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %r0 = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %init) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
    %r1 = ttl.compute
        ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
        outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
        {indexing_maps = [#map, #map],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
         %out_tile: !ttcore.tile<32x32, f32>):
      %i0 = ttl.iter_index 0 : index
      %j0 = ttl.iter_index 1 : index
      ttl.tile_store %a_tile, %out_view[%i0, %j0] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, f32>>
    scf.yield %r1 : tensor<1x1x!ttcore.tile<32x32, f32>>
  }

  // Second accumulation loop.
  %r2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%acc = %r0) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
    %r3 = ttl.compute
        ins(%b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
        outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
        {indexing_maps = [#map, #map],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%b_tile: !ttcore.tile<32x32, f32>,
         %out_tile: !ttcore.tile<32x32, f32>):
      %i1 = ttl.iter_index 0 : index
      %j1 = ttl.iter_index 1 : index
      ttl.tile_store %b_tile, %out_view[%i1, %j1] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, f32>>
    scf.yield %r3 : tensor<1x1x!ttcore.tile<32x32, f32>>
  }

  func.return %r2 : tensor<1x1x!ttcore.tile<32x32, f32>>
}
