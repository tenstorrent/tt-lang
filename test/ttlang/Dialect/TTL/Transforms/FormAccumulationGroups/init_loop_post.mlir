// Summary: Init compute + loop + post compute all store to the same view.
// All three get the same accumulation group ID. The init store is
// converted to acc=true.
//
// RUN: ttlang-opt %s -ttl-form-accumulation-groups | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @init_loop_post
// Init compute: converted to acc=true, gets group.
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 0
// CHECK: ttl.tile_store
// CHECK-SAME: acc = true
// Loop compute.
// CHECK: scf.for
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 0
// Post compute.
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 0

func.func @init_loop_post(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                           %b: tensor<1x1x!ttcore.tile<32x32, f32>>,
                           %c: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb3 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Init store (acc=false, converted to acc=true).
  %r0 = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %o: !ttcore.tile<32x32, f32>):
    %i0 = ttl.iter_index 0 : index
    %j0 = ttl.iter_index 1 : index
    ttl.tile_store %a_tile, %out_view[%i0, %j0] {acc = false} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Accumulation loop.
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %r1 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %r0) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
    %r2 = ttl.compute
        ins(%b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
        outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
        {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%b_tile: !ttcore.tile<32x32, f32>, %o: !ttcore.tile<32x32, f32>):
      %i1 = ttl.iter_index 0 : index
      %j1 = ttl.iter_index 1 : index
      ttl.tile_store %b_tile, %out_view[%i1, %j1] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, f32>>
    scf.yield %r2 : tensor<1x1x!ttcore.tile<32x32, f32>>
  }

  // Post-loop store (acc=true).
  %r3 = ttl.compute
      ins(%c_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%c_tile: !ttcore.tile<32x32, f32>, %o: !ttcore.tile<32x32, f32>):
    %i2 = ttl.iter_index 0 : index
    %j2 = ttl.iter_index 1 : index
    ttl.tile_store %c_tile, %out_view[%i2, %j2] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %r3 : tensor<1x1x!ttcore.tile<32x32, f32>>
}
