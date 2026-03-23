// Summary: Acc stores to different views form separate groups with
// different group IDs.
//
// RUN: ttlang-opt %s -ttl-form-accumulation-groups | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @independent_views
// First group: two computes storing to view1.
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 0
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 0
// Second group: two computes storing to view2.
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 1
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 1

func.func @independent_views(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                              %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> (tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 17, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb1 = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb2 = ttl.attach_cb %init, %cb3 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %view1 = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %view2 = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Group 0: stores to view1.
  %r0 = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb1 : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%t0: !ttcore.tile<32x32, f32>, %o0: !ttcore.tile<32x32, f32>):
    %i0 = ttl.iter_index 0 : index
    %j0 = ttl.iter_index 1 : index
    ttl.tile_store %t0, %view1[%i0, %j0] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %r1 = ttl.compute
      ins(%b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb1 : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%t1: !ttcore.tile<32x32, f32>, %o1: !ttcore.tile<32x32, f32>):
    %i1 = ttl.iter_index 0 : index
    %j1 = ttl.iter_index 1 : index
    ttl.tile_store %t1, %view1[%i1, %j1] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // Group 1: stores to view2.
  %r2 = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb2 : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%t2: !ttcore.tile<32x32, f32>, %o2: !ttcore.tile<32x32, f32>):
    %i2 = ttl.iter_index 0 : index
    %j2 = ttl.iter_index 1 : index
    ttl.tile_store %t2, %view2[%i2, %j2] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %r3 = ttl.compute
      ins(%b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb2 : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%t3: !ttcore.tile<32x32, f32>, %o3: !ttcore.tile<32x32, f32>):
    %i3 = ttl.iter_index 0 : index
    %j3 = ttl.iter_index 1 : index
    ttl.tile_store %t3, %view2[%i3, %j3] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %r1, %r3 : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>
}
