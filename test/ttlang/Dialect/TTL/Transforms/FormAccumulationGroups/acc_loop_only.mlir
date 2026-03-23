// Summary: A single compute with acc=true inside a user loop forms an
// accumulation group (loop iterations provide the multi-store semantics).
//
// RUN: ttlang-opt %s -ttl-form-accumulation-groups | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @acc_loop_only
// CHECK: scf.for
// CHECK: ttl.compute
// CHECK-SAME: ttl.acc_group = 0
// CHECK: ttl.tile_store
// CHECK-SAME: acc = true

func.func @acc_loop_only(%a: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
    %r0 = ttl.compute
        ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
        outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
        {indexing_maps = [#map, #map],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
         %out_tile: !ttcore.tile<32x32, f32>):
      %ii = ttl.iter_index 0 : index
      %jj = ttl.iter_index 1 : index
      ttl.tile_store %a_tile, %out_view[%ii, %jj] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, f32>>
    scf.yield %r0 : tensor<1x1x!ttcore.tile<32x32, f32>>
  }

  func.return %r : tensor<1x1x!ttcore.tile<32x32, f32>>
}
