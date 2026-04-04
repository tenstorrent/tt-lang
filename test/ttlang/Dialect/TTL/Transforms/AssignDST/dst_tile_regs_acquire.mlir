// Summary: ensure DST assignment and tile_regs sync are correctly inserted in ttl.compute.
// FPU binary ops (both operands from CB block args) get ttl.fpu_binary and need no copy_tile.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst), canonicalize, cse)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{enable-fpu-binary-ops=0}), canonicalize, cse)' --split-input-file | FileCheck %s --check-prefix=SFPU

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder
// SFPU-NOT: placeholder

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify tile_regs_acquire wraps compute body, commit/wait/store/release
// before yield. tile_add with both operands from block args is FPU binary (no
// copy_tile needed).
// CHECK-LABEL:   func.func @acquire_insert
// CHECK-DAG:       %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK-DAG:       %[[CB1:.*]] = ttl.bind_cb{cb_index = 1, buffer_factor = 2}
// CHECK-DAG:       %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// CHECK:           %[[RES:.*]] = ttl.compute
// CHECK:           ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[O:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// FPU binary: no copy_tile needed, tile_add operates directly on block args
// CHECK-NEXT:        %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] {dst_idx = 0 : i32, ttl.fpu_binary}
// CHECK-NEXT:        ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:        ttl.yield
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      return %[[RES]]
//
// SFPU path: init_sfpu instead of init_binary, copy_tile for both operands
// SFPU-LABEL:   func.func @acquire_insert
// SFPU-DAG:       %[[CB0S:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// SFPU-DAG:       %[[CB2S:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// SFPU:           ttl.compute
// SFPU:           ^bb0
// SFPU-NOT:         fpu_binary
// SFPU:             ttl.copy_tile {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// SFPU:             %[[ADDS:.*]] = ttl.tile_add {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.tile_store %[[ADDS]], %{{.*}}[%{{.*}}, %{{.*}}]
// SFPU-NEXT:        ttl.yield
func.func @acquire_insert(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                          %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %result_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Re-declare map for split input.
#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: ensure per-compute acquire, commit/wait before yield, store before yield, and release after.
// Both computes have FPU binary tile_add (no copy_tile).
// CHECK-LABEL:   func.func @acquire_two_computes
// CHECK-DAG:       %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK-DAG:       %[[CB1:.*]] = ttl.bind_cb{cb_index = 1, buffer_factor = 2}
// CHECK-DAG:       %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// CHECK:           %[[R0:.*]] = ttl.compute
// CHECK:           ^bb0(%[[A0:.*]]: !ttcore.tile<32x32, f32>, %[[B0:.*]]: !ttcore.tile<32x32, f32>, %{{.*}}: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0_0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1_0:.*]] = ttl.iter_index 1 : index
// CHECK-NEXT:        %[[SUM0:.*]] = ttl.tile_add %[[A0]], %[[B0]] {dst_idx = 0 : i32, ttl.fpu_binary}
// CHECK-NEXT:        ttl.tile_store %[[SUM0]], %{{.*}}[%[[I0_0]], %[[I1_0]]]
// CHECK-NEXT:        ttl.yield
// Inter-compute: bind_cb for output of first compute, attach_cb
// CHECK:           %[[CB3:.*]] = ttl.bind_cb{cb_index = 3, buffer_factor = 2}
// CHECK-NEXT:      %{{.*}} = ttl.attach_cb %[[R0]], %[[CB3]]
// CHECK:           %[[R1:.*]] = ttl.compute
// CHECK:           ^bb0(%[[A1:.*]]: !ttcore.tile<32x32, f32>, %[[B1:.*]]: !ttcore.tile<32x32, f32>, %{{.*}}: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0_1:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1_1:.*]] = ttl.iter_index 1 : index
// CHECK-NEXT:        %[[SUM1:.*]] = ttl.tile_add %[[A1]], %[[B1]] {dst_idx = 0 : i32, ttl.fpu_binary}
// CHECK-NEXT:        ttl.tile_store %[[SUM1]], %{{.*}}[%[[I0_1]], %[[I1_1]]]
// CHECK-NEXT:        ttl.yield
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      return %[[R1]]
//
// SFPU path: init_sfpu for both computes, copy_tile for operands
// SFPU-LABEL:   func.func @acquire_two_computes
// SFPU-DAG:       %[[CB0S:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// SFPU-DAG:       %[[CB2S:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// First compute: SFPU binary add with copy_tiles
// SFPU:           ttl.compute
// SFPU:           ^bb0
// SFPU-NOT:         fpu_binary
// SFPU:             ttl.copy_tile
// SFPU:             ttl.copy_tile
// SFPU:             ttl.tile_add {{.*}} {dst_idx = 0 : i32}
// Second compute: also SFPU
// SFPU:           ttl.compute
// SFPU:           ^bb0
// SFPU-NOT:         fpu_binary
// SFPU:             ttl.copy_tile
// SFPU:             ttl.copy_tile
// SFPU:             ttl.tile_add {{.*}} {dst_idx = 0 : i32}
func.func @acquire_two_computes(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result_view0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %r0 = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %result_view0[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %r0_cb = ttl.attach_cb %r0, %cb3
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>)
      -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result_view1 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %r1 = ttl.compute
      ins(%r0_cb, %r0_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                           tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %result_view1[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %r1 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: op chain add->mul->exp with reg sync. tile_add is FPU binary (both
// operands from block args), tile_mul has one operand from DST so needs copy_tile
// for the other operand (%c_tile), tile_exp is SFPU.
// CHECK-LABEL:   func.func @acquire_chain_three_ops
// CHECK-SAME:      (%[[AARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[BARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[CARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK-DAG:       %[[CB1:.*]] = ttl.bind_cb{cb_index = 1, buffer_factor = 2}
// CHECK-DAG:       %[[CB3:.*]] = ttl.bind_cb{cb_index = 3, buffer_factor = 2}
// CHECK:           %[[RES:.*]] = ttl.compute
// CHECK:           ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[C:.*]]: !ttcore.tile<32x32, f32>, %[[O:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// FPU binary tile_add: no copy_tile needed
// CHECK-NEXT:        %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] {dst_idx = 0 : i32, ttl.fpu_binary}
// tile_mul needs copy_tile for %c (not in DST)
// CHECK-NEXT:        %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[C]][%[[I0]], %[[I1]]], %[[C1]] {dst_idx = 1 : i32}
// CHECK-NEXT:        %[[MUL:.*]] = ttl.tile_mul %[[ADD]], %[[DTILE]] {dst_idx = 0 : i32}
// CHECK-NEXT:        %[[EXP:.*]] = ttl.tile_exp %[[MUL]] {dst_idx = 0 : i32}
// CHECK-NEXT:        ttl.tile_store %[[EXP]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:        ttl.yield
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      return %[[RES]]
//
// SFPU path: init_sfpu, add uses copy_tile for both operands
// SFPU-LABEL:   func.func @acquire_chain_three_ops
// SFPU-DAG:       %[[CB0S:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// SFPU-DAG:       %[[CB3S:.*]] = ttl.bind_cb{cb_index = 3, buffer_factor = 2}
// SFPU:           ttl.compute
// SFPU:           ^bb0
// SFPU-NOT:         fpu_binary
// copy A and B for SFPU add
// SFPU:             ttl.copy_tile {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// SFPU:             ttl.tile_add {{.*}} {dst_idx = 0 : i32}
// copy C for mul
// SFPU:             ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// SFPU:             ttl.tile_mul {{.*}} {dst_idx = 0 : i32}
// SFPU:             %[[EXPS:.*]] = ttl.tile_exp {{.*}} {dst_idx = 0 : i32}
// SFPU:             ttl.tile_store %[[EXPS]]
// SFPU-NEXT:        ttl.yield
func.func @acquire_chain_three_ops(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result_view = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %c_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %result_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify pre-existing tile_regs_acquire in parent stays there (not moved
// inside body). tile_add is FPU binary (no copy_tile).
// CHECK-LABEL:   func.func @init_sfpu_with_preexisting_acquire
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK:           %[[CB1:.*]] = ttl.bind_cb{cb_index = 1, buffer_factor = 2}
// CHECK:           %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// CHECK:           ttl.tile_regs_acquire
// CHECK:           %[[RES:.*]] = ttl.compute
// CHECK:           ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %{{.*}}: !ttcore.tile<32x32, f32>):
// No tile_regs_acquire inside body (pre-existing one is in parent)
// CHECK-NEXT:        %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:        %[[I1:.*]] = ttl.iter_index 1 : index
// CHECK-NEXT:        %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] {dst_idx = 0 : i32, ttl.fpu_binary}
// CHECK-NEXT:        ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]]]
// CHECK-NEXT:        ttl.yield
// CHECK-NEXT:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK-NEXT:      return %[[RES]]
//
// SFPU path: pre-existing acquire stays, copy_tile for both operands
// SFPU-LABEL:   func.func @init_sfpu_with_preexisting_acquire
// SFPU:           %[[CB0S:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// SFPU:           %[[CB2S:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// SFPU:           ttl.tile_regs_acquire
// SFPU:           ttl.compute
// SFPU:           ^bb0
// SFPU-NOT:       fpu_binary
// SFPU:           ttl.copy_tile {{.*}} {dst_idx = 0 : i32}
// SFPU:           ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// SFPU:           ttl.tile_add {{.*}} {dst_idx = 0 : i32}
func.func @init_sfpu_with_preexisting_acquire(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                              %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Pre-inserted tile_regs_acquire - pass preserves it and doesn't insert a duplicate.
  ttl.tile_regs_acquire

  %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %result_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify pre-existing init_sfpu and tile_regs_acquire are preserved even
// with ops in between. Sync pass should not insert duplicates.
// tile_add is FPU binary (no copy_tile).
// CHECK-LABEL:   func.func @ops_between_acquire_and_compute
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK:           %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// CHECK:           ttl.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK-NEXT:      ttl.tile_regs_acquire
// CHECK:           tensor.empty
// CHECK:           ttl.compute
// Verify no duplicate sync ops were inserted
// CHECK-NOT:       ttl.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK-NOT:       ttl.tile_regs_acquire
//
// SFPU path: same behavior (source already has init_sfpu, add becomes SFPU)
// SFPU-LABEL:   func.func @ops_between_acquire_and_compute
// SFPU:           %[[CB0S:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// SFPU:           %[[CB2S:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// SFPU:           ttl.init_sfpu(%[[CB0S]], %[[CB2S]])
// SFPU-NEXT:      ttl.tile_regs_acquire
// SFPU:           ttl.compute
// SFPU-NOT:       fpu_binary
// SFPU:           ttl.copy_tile
// SFPU:           ttl.copy_tile
// SFPU:           ttl.tile_add {{.*}} {dst_idx = 0 : i32}
// No duplicate sync ops
// SFPU-NOT:       ttl.init_sfpu
// SFPU-NOT:       ttl.tile_regs_acquire
func.func @ops_between_acquire_and_compute(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                            %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Pre-inserted sync ops
  ttl.init_sfpu(%cb0, %cb2) : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  ttl.tile_regs_acquire

  // Operations between sync ops and compute
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %result_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
