// Summary: Two accumulation groups targeting different multi-tile views inside
// the same user loop. Loop peeling for view_a must not affect view_b's stores
// in the peeled iteration.
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-form-accumulation-groups{maximize-dst=0}))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @two_views_in_loop
//
// Peeled iteration: view_a store has acc stripped, view_b store retains acc.
// CHECK:       ttl.compute
// CHECK:         ttl.tile_store {{.*}} %[[VIEW_A:.*]][
// CHECK-NOT:     acc = true
// CHECK:       ttl.compute
// CHECK:         ttl.tile_store {{.*}} %[[VIEW_B:.*]][
// CHECK-SAME:    acc = true
//
// Loop body: both stores retain acc=true.
// CHECK:       scf.for
// CHECK:         ttl.compute
// CHECK:           ttl.tile_store {{.*}}[
// CHECK-SAME:      acc = true
// CHECK:         ttl.compute
// CHECK:           ttl.tile_store {{.*}}[
// CHECK-SAME:      acc = true

func.func @two_views_in_loop(
    %a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
    %b: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 17, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init_cb_a = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init_cb_b = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %view_a = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %view_b = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %init_cb_a) -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
    // Compute storing to view_a with acc=true.
    %r0 = ttl.compute
        ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
        outs(%init_cb_a : tensor<2x2x!ttcore.tile<32x32, bf16>>)
        {indexing_maps = [#map, #map],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
      %i0 = ttl.iter_index 0 : index
      %j0 = ttl.iter_index 1 : index
      ttl.tile_store %a_tile, %view_a[%i0, %j0] {acc = true} : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.yield
    } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

    // Compute storing to view_b with acc=true.
    %r1 = ttl.compute
        ins(%b_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
        outs(%init_cb_b : tensor<2x2x!ttcore.tile<32x32, bf16>>)
        {indexing_maps = [#map, #map],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%b_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
      %i1 = ttl.iter_index 0 : index
      %j1 = ttl.iter_index 1 : index
      ttl.tile_store %b_tile, %view_b[%i1, %j1] {acc = true} : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.yield
    } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

    scf.yield %r1 : tensor<2x2x!ttcore.tile<32x32, bf16>>
  }

  func.return %r : tensor<2x2x!ttcore.tile<32x32, bf16>>
}
