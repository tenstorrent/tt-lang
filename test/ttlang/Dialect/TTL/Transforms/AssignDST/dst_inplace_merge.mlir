// Summary: In-place SFPU ops merge dst_idx with their copy_tile source.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize,cse)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=SEPARATE

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify copy_tile emits token+tile, unary ops consume copied tile,
// and dst_idx annotations are added to math ops. Unary operations are merged
// into a single equivalence class, so separate-output-region does not change
// dst_idx (operations remain at dst_idx = 0).
// CHECK-LABEL: func.func @inplace_unary_chain
func.func @inplace_unary_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

// CHECK: %[[RESULT:.*]] = ttl.compute
// CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
// CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[EXP:.*]] = ttl.tile_exp %[[DTILE]] into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[RELU:.*]] = ttl.tile_relu %[[EXP]] into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[SIG:.*]] = ttl.tile_sigmoid %[[RELU]] into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
// SEPARATE: ttl.tile_sigmoid {{.*}} into dst[%c0]
// CHECK:        ttl.tile_store %[[SIG]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:   ttl.yield
// CHECK: }
  %out_view = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %exp = ttl.tile_exp %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %relu = ttl.tile_relu %exp into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %sigmoid = ttl.tile_sigmoid %relu into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %sigmoid, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: return %[[RESULT]]
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: exp(a) + b. exp is in-place and its output feeds tile_add, which is
// NOT in-place and NOT yielded directly (only add's result is yielded).
// The in-place merge must be unconditional: copy_tile and exp must share the
// same dst_idx (both 0), so exp_tile reads from the correct register.
// dstPerIteration = 2 (copy+exp merged at 0, copy_b at 1). unroll_factor = 4.
// CHECK-LABEL: func.func @inplace_feeds_sfpu_binary
// CHECK:           ttl.compute
// CHECK-SAME:      ttl.unroll_factor = 4
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// copy A to DST[0]
// CHECK:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%c0]
// exp operates in-place on DST[0] — MUST be 0, not a different index
// CHECK-NEXT:      %[[EXP:.*]] = ttl.tile_exp %[[ATILE]] into dst[%c0]
// copy B to DST[1]
// CHECK:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]][%[[I0]], %[[I1]]] into dst[%c1]
// SFPU binary add: reads exp result (DST[0]) and B copy (DST[1])
// CHECK-NEXT:      %[[ADD:.*]] = ttl.tile_add %[[EXP]], %[[BTILE]] into dst[%c0]
// CHECK:           ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield
func.func @inplace_feeds_sfpu_binary(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                      %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %exp = ttl.tile_exp %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %exp, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %out_view_0[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: abs(a) feeds exp (in-place chain), then exp's output feeds tile_mul
// (non-in-place). The entire in-place chain (copy, abs, exp) must share dst_idx
// 0. Without unconditional merge, abs might get a different index from copy.
// dstPerIteration = 2 (chain at 0, copy_b at 1). unroll_factor = 4.
// CHECK-LABEL: func.func @inplace_chain_feeds_non_inplace
// CHECK:           ttl.compute
// CHECK-SAME:      ttl.unroll_factor = 4
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// copy A -> DST[0], abs in-place at DST[0], exp in-place at DST[0]
// CHECK:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%c0]
// CHECK-NEXT:      %[[ABS:.*]] = ttl.tile_abs %[[ATILE]] into dst[%c0]
// CHECK-NEXT:      %[[EXP:.*]] = ttl.tile_exp %[[ABS]] into dst[%c0]
// copy B -> DST[1]
// CHECK:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]][%[[I0]], %[[I1]]] into dst[%c1]
// mul reads exp result (DST[0]) and B (DST[1])
// CHECK-NEXT:      %[[MUL:.*]] = ttl.tile_mul %[[EXP]], %[[BTILE]] into dst[%c0]
// CHECK:           ttl.tile_store %[[MUL]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield
func.func @inplace_chain_feeds_non_inplace(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                            %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_1 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %abs = ttl.tile_abs %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %abs into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %exp, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %mul, %out_view_1[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: exp(a) and abs(b) both feed tile_add (non-in-place). Each chain
// must merge independently: exp shares DST with copy_a (DST[0]), abs shares
// DST with copy_b (DST[1]). Neither exp nor abs result is yielded directly.
// dstPerIteration = 2. unroll_factor = 4.
// CHECK-LABEL: func.func @two_inplace_chains_feed_binary
// CHECK:           ttl.compute
// CHECK-SAME:      ttl.unroll_factor = 4
// CHECK-NEXT:      ^bb0(%[[A:[^:]*]]: !ttcore.tile<32x32, f32>, %[[B:[^:]*]]: !ttcore.tile<32x32, f32>, %[[OUT:[^:]*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// chain 1: copy A -> DST[0], exp in-place at DST[0]
// CHECK:           %{{.*}}, %[[ATILE:.*]] = ttl.copy_tile %[[A]][%[[I0]], %[[I1]]] into dst[%c0]
// CHECK-NEXT:      %[[EXP:.*]] = ttl.tile_exp %[[ATILE]] into dst[%c0]
// chain 2: copy B -> DST[1], abs in-place at DST[1]
// CHECK:           %{{.*}}, %[[BTILE:.*]] = ttl.copy_tile %[[B]][%[[I0]], %[[I1]]] into dst[%c1]
// CHECK-NEXT:      %[[ABS:.*]] = ttl.tile_abs %[[BTILE]] into dst[%c1]
// add reads from DST[0] and DST[1]
// CHECK-NEXT:      %[[ADD:.*]] = ttl.tile_add %[[EXP]], %[[ABS]] into dst[%c0]
// CHECK:           ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:      ttl.yield
func.func @two_inplace_chains_feed_binary(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                           %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view_2 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %exp = ttl.tile_exp %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %abs = ttl.tile_abs %b_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %exp, %abs into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %out_view_2[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
