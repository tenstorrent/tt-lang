// Summary: Single acc=true compute inside a user scf.for with 2x2 output view.
// Multi-tile domains skip DST grouping; the first iteration is peeled so a
// compute with acc=false appears before the loop, and the loop body retains
// acc=true. No acc_group is formed.
//
// RUN: ttlang-opt %s -ttl-form-accumulation-groups | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @acc_loop_only_multitile
// Peeled first iteration before the loop with acc stripped (overwrite).
// CHECK:       ttl.compute
// CHECK-NOT:   ttl.acc_group
// CHECK:       ttl.tile_store
// CHECK-NOT:   acc = true
// CHECK-SAME:  tensor<2x2x!ttcore.tile<32x32, bf16>>
// Loop lower bound advanced by step.
// CHECK:       %[[NEW_LB:.*]] = arith.addi
// CHECK:       scf.for %{{.*}} = %[[NEW_LB]]
// Loop body retains acc=true, no acc_group.
// CHECK:       ttl.compute
// CHECK-NOT:   ttl.acc_group
// CHECK:       ttl.tile_store
// CHECK-SAME:  acc = true

func.func @acc_loop_only_multitile(%a: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
    %r0 = ttl.compute
        ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
        outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>)
        {indexing_maps = [#map, #map],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
         %out_tile: !ttcore.tile<32x32, bf16>):
      %ii = ttl.iter_index 0 : index
      %jj = ttl.iter_index 1 : index
      ttl.tile_store %a_tile, %out_view[%ii, %jj] {acc = true} : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.yield
    } -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    scf.yield %r0 : tensor<2x2x!ttcore.tile<32x32, bf16>>
  }

  func.return %r : tensor<2x2x!ttcore.tile<32x32, bf16>>
}
