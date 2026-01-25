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

// -----

// Test: User loop surrounding two computes - cb_wait inserted INSIDE user loop
// This tests that ttl.tile_loop.outer correctly identifies compute boundaries
// and doesn't treat the user loop as part of the compute nest.

// CHECK-LABEL: func.func @user_loop_around_computes
// CHECK:         [[CB2:%.*]] = ttl.bind_cb{cb_index = 2
// User loop (no tile_loop markers)
// CHECK:         scf.for
// First compute's outer loop (inside user loop)
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               ttl.tile_add
// cb_wait for CB2 inserted INSIDE user loop, between compute nests
// CHECK:           ttl.cb_wait [[CB2]]
// Second compute's outer loop (inside user loop)
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               ttl.tile_mul
// End of user loop
// CHECK:         return

func.func @user_loop_around_computes(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
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

  // USER LOOP surrounding both computes
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index

  %final = scf.for %user_iter = %c0 to %c3 step %c1 iter_args(%acc = %init_cb) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
    // Attach CB inside loop for outputs
    %acc_cb = ttl.attach_cb %acc, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

    // First compute: a + b -> cb2
    %r0 = ttl.compute
        ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                 tensor<2x2x!ttcore.tile<32x32, f32>>)
        outs(%acc_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
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

    // Attach r0 to SAME cb2 for second compute input
    %r0_cb = ttl.attach_cb %r0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %init2_cb = ttl.attach_cb %init2, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

    // Second compute: r0 * r0 -> cb2 (reads from cb2)
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

    scf.yield %result : tensor<2x2x!ttcore.tile<32x32, f32>>
  }

  func.return %final : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Non-consecutive CB dependency
// Loop 0: writes CB2
// Loop 1: writes CB3 (no CB2 dependency)
//          ↓
//     cb_wait CB2  ← inserted before consumer, not after producer
//          ↓
// Loop 2: reads CB2

// CHECK-LABEL: func.func @non_consecutive_cb_dependency
// CHECK:         [[CB2:%.*]] = ttl.bind_cb{cb_index = 2
// CHECK:         [[CB3:%.*]] = ttl.bind_cb{cb_index = 3
// First loop writes to CB2
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             ttl.tile_add
// CHECK-NOT:    ttl.cb_wait [[CB2]]
// Second loop writes to CB3 (no CB2 dependency)
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             ttl.tile_mul
// cb_wait for CB2 before third loop (non-consecutive dependency from loop 0)
// CHECK:         ttl.cb_wait [[CB2]]
// Third loop reads from CB2
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             ttl.tile_add

func.func @non_consecutive_cb_dependency(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                          %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init2 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init3 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  %a_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_ready = ttl.cb_wait %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb2 = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Loop 0: a + b -> CB2
  %r0 = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>)
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

  // Loop 1: a * b -> CB3 (no dependency on CB2)
  %init2_cb3 = ttl.attach_cb %init2, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %r1 = ttl.compute
      ins(%a_ready, %b_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                               tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init2_cb3 : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile2: !ttcore.tile<32x32, f32>,
       %b_tile2: !ttcore.tile<32x32, f32>,
       %out_tile2: !ttcore.tile<32x32, f32>):
    %prod = ttl.tile_mul %a_tile2, %b_tile2 : !ttcore.tile<32x32, f32>
    %view1 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %prod, %view1 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %prod : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Loop 2: r0 + a -> CB3 (reads from CB2 written by Loop 0, NOT Loop 1)
  %r0_cb = ttl.attach_cb %r0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init3_cb3 = ttl.attach_cb %init3, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%r0_cb, %a_ready : tensor<2x2x!ttcore.tile<32x32, f32>>,
                             tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init3_cb3 : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%r0_tile: !ttcore.tile<32x32, f32>,
       %a_tile3: !ttcore.tile<32x32, f32>,
       %out_tile3: !ttcore.tile<32x32, f32>):
    %final_sum = ttl.tile_add %r0_tile, %a_tile3 : !ttcore.tile<32x32, f32>
    %view2 = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %final_sum, %view2 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 1>
    ttl.yield %final_sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Producer in ancestor block dominates consumer in nested block
// compute1 at function level writes CB2
// compute2 inside user loop reads CB2
// cb_wait should be inserted before compute2's outermost loop

// CHECK-LABEL: func.func @producer_in_ancestor_block
// CHECK:         [[CB2:%.*]] = ttl.bind_cb{cb_index = 2
// First compute at function level (writes CB2)
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             ttl.tile_add
// User loop starts
// CHECK:         scf.for
// cb_wait for CB2 inside user loop, before second compute
// CHECK:           ttl.cb_wait [[CB2]]
// Second compute inside user loop (reads CB2)
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               ttl.tile_mul

func.func @producer_in_ancestor_block(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
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

  // First compute at FUNCTION LEVEL: a + b -> cb2
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

  // USER LOOP with second compute inside
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index

  %final = scf.for %user_iter = %c0 to %c3 step %c1 iter_args(%acc = %init2) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
    // Attach r0 (from ancestor block) to CB2
    %r0_cb = ttl.attach_cb %r0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %acc_cb = ttl.attach_cb %acc, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

    // Second compute INSIDE USER LOOP: r0 * r0 -> cb2 (reads from cb2)
    %result = ttl.compute
        ins(%r0_cb, %r0_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                             tensor<2x2x!ttcore.tile<32x32, f32>>)
        outs(%acc_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
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

    scf.yield %result : tensor<2x2x!ttcore.tile<32x32, f32>>
  }

  func.return %final : tensor<2x2x!ttcore.tile<32x32, f32>>
}
