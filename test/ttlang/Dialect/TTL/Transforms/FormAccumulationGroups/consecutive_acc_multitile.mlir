// Summary: Two consecutive computes with acc=true targeting a 2x2 output view.
// Multi-tile domains skip DST grouping; the first store is set to acc=false
// (overwrite) and the second retains acc=true (L1 accumulation).
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-form-accumulation-groups{maximize-dst=0}))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @consecutive_acc_multitile
// First compute: no acc_group, store has acc stripped (overwrite).
// CHECK:       ttl.compute
// CHECK-NOT:   ttl.acc_group
// CHECK:       ttl.tile_store
// CHECK-NOT:   acc = true
// CHECK-SAME:  tensor<2x2x!ttcore.tile<32x32, bf16>>
// Second compute: no acc_group, store retains acc=true.
// CHECK:       ttl.compute
// CHECK-NOT:   ttl.acc_group
// CHECK:       ttl.tile_store
// CHECK-SAME:  acc = true

func.func @consecutive_acc_multitile(%a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
                                      %b: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %r0 = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i0 = ttl.iter_index 0 : index
    %j0 = ttl.iter_index 1 : index
    ttl.tile_store %a_tile, %out_view[%i0, %j0] {acc = true} : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %r1 = ttl.compute
      ins(%b_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%b_tile: !ttcore.tile<32x32, bf16>,
       %out_tile: !ttcore.tile<32x32, bf16>):
    %i1 = ttl.iter_index 0 : index
    %j1 = ttl.iter_index 1 : index
    ttl.tile_store %b_tile, %out_view[%i1, %j1] {acc = true} : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  func.return %r1 : tensor<2x2x!ttcore.tile<32x32, bf16>>
}
