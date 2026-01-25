// RUN: ttlang-opt %s --ttl-assign-dst --ttl-lower-to-loops --ttl-insert-inter-loop-cb-sync | FileCheck %s

// Test: Inter-loop CB synchronization pass inserts cb_wait when consecutive
// loops share the same CB (loop1.output_cb == loop2.input_cb).

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @two_computes_same_cb
// CHECK:         ttl.bind_cb{cb_index = 0
// CHECK:         ttl.bind_cb{cb_index = 1
// CHECK:         [[CB2:%.*]] = ttl.bind_cb{cb_index = 2
// First outer loop
// CHECK:         scf.for
// First inner loop (has tile_loop marker with output_cb = 2)
// CHECK:           scf.for
// CHECK:             ttl.copy_tile
// CHECK:             ttl.copy_tile
// CHECK:             ttl.tile_add
// After first loop nest, before second: cb_wait should be inserted for CB2
// Only ONE cb_wait should be inserted even though both inputs use CB2 (dedup test)
// CHECK:         ttl.cb_wait [[CB2]]
// CHECK-NOT:     ttl.cb_wait
// Second outer loop
// CHECK:         scf.for
// Second inner loop (has tile_loop marker with input_cb = 2)
// CHECK:           scf.for
// CHECK:             ttl.copy_tile
// CHECK:             ttl.tile_mul

func.func @two_computes_same_cb(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                 %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // First compute: a + b -> cb2
  %r0 = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %view0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %view0 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Attach r0 to SAME cb2 for second compute input (this creates the inter-loop dependency)
  %r0_cb = ttl.attach_cb %r0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2_cb = ttl.attach_cb %init2, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: r0 * r0 -> cb2 (reads from cb2, writes to cb2)
  %result = ttl.compute
      ins(%r0_cb, %r0_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                           tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init2_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%r0_tile1: !ttcore.tile<32x32, f32>,
       %r0_tile2: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %product = ttl.tile_mul %r0_tile1, %r0_tile2 : !ttcore.tile<32x32, f32>
    %view1 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %product, %view1 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %product : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: No cb_wait inserted when CBs don't match (different input/output CBs)

// CHECK-LABEL: func.func @two_computes_different_cbs
// First loop (output_cb = 2)
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             ttl.tile_add
// Between loops - NO cb_wait should be inserted (output_cb=2, input_cb=3)
// CHECK-NOT:     ttl.cb_wait
// Second loop (input_cb = 3)
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             ttl.tile_mul

func.func @two_computes_different_cbs(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                       %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // First compute: a + b -> cb2
  %r0 = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %view0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %view0 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Attach r0 to DIFFERENT cb3 (no inter-loop dependency)
  %r0_cb = ttl.attach_cb %r0, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2_cb = ttl.attach_cb %init2, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: r0 * r0 -> cb3 (reads from cb3, writes to cb3)
  %result = ttl.compute
      ins(%r0_cb, %r0_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                           tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init2_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%r0_tile1: !ttcore.tile<32x32, f32>,
       %r0_tile2: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %product = ttl.tile_mul %r0_tile1, %r0_tile2 : !ttcore.tile<32x32, f32>
    %view1 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %product, %view1 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %product : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
