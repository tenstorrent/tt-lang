// Summary: verify DST sync ops are inserted after loop lowering.
// The sync pass now operates on scf.for loops marked with ttl.tile_loop attribute.
// RUN: ttlang-opt %s --ttl-assign-dst --ttl-lower-to-loops --ttl-insert-tile-regs-sync --canonicalize --cse --split-input-file | FileCheck %s

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify DST sync ops are correctly inserted inside scf.for bodies.
// CHECK-LABEL:   func.func @acquire_insert
// CHECK-DAG:       %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK-DAG:       %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// CHECK:           ttl.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK-NEXT:      %[[RES:.*]] = scf.for
// CHECK:             scf.for
// CHECK:               ttl.tile_regs_acquire
// CHECK:               ttl.copy_tile
// CHECK-NEXT:          %{{.*}}, %[[DTILE1:.*]] = ttl.copy_tile
// CHECK-NEXT:          %[[ADD:.*]] = ttl.tile_add {{.*}} {dst_idx = 0 : i32}
// CHECK-NEXT:          %[[INS:.*]] = tensor.insert %[[ADD]]
// CHECK-NEXT:          ttl.tile_regs_commit
// CHECK-NEXT:          ttl.tile_regs_wait
// CHECK-NEXT:          %[[V:.*]] = ttl.cb_reserve %[[CB2]]
// CHECK-NEXT:          ttl.store %[[ADD]], %[[V]]
// CHECK-NEXT:          ttl.tile_regs_release
// CHECK-NEXT:          scf.yield %[[INS]]
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

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify each compute gets its own init_sfpu and sync ops.
// CHECK-LABEL:   func.func @acquire_two_computes
// CHECK-DAG:       %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK-DAG:       %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// First compute loop
// CHECK:           ttl.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK-NEXT:      %[[R0:.*]] = scf.for
// CHECK:             scf.for
// CHECK:               ttl.tile_regs_acquire
// CHECK:               %[[SUM0:.*]] = ttl.tile_add
// CHECK-NEXT:          %[[INS0:.*]] = tensor.insert %[[SUM0]]
// CHECK-NEXT:          ttl.tile_regs_commit
// CHECK-NEXT:          ttl.tile_regs_wait
// CHECK-NEXT:          %[[V0:.*]] = ttl.cb_reserve %[[CB2]]
// CHECK-NEXT:          ttl.store %[[SUM0]], %[[V0]]
// CHECK-NEXT:          ttl.tile_regs_release
// CHECK-NEXT:          scf.yield %[[INS0]]
// Second compute loop
// CHECK:           %[[CB3:.*]] = ttl.bind_cb{cb_index = 3, buffer_factor = 2}
// CHECK-NEXT:      %[[R0CB:.*]] = ttl.attach_cb %[[R0]], %[[CB3]]
// CHECK-NEXT:      ttl.init_sfpu(%[[CB3]], %[[CB2]])
// CHECK-NEXT:      %[[R1:.*]] = scf.for
// CHECK:             scf.for
// CHECK:               ttl.tile_regs_acquire
// CHECK:               %[[SUM1:.*]] = ttl.tile_add
// CHECK-NEXT:          %[[INS1:.*]] = tensor.insert %[[SUM1]]
// CHECK-NEXT:          ttl.tile_regs_commit
// CHECK-NEXT:          ttl.tile_regs_wait
// CHECK-NEXT:          %[[V1:.*]] = ttl.cb_reserve %[[CB2]]
// CHECK-NEXT:          ttl.store %[[SUM1]], %[[V1]]
// CHECK-NEXT:          ttl.tile_regs_release
// CHECK-NEXT:          scf.yield %[[INS1]]
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

  %r0 = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %result_view0 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %result_view0 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %r0_cb = ttl.attach_cb %r0, %cb3
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>)
      -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %r1 = ttl.compute
      ins(%r0_cb, %r0_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                           tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %result_view1 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %result_view1 : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %r1 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: op chain add->mul->exp with DST sync ops.
// CHECK-LABEL:   func.func @acquire_chain_three_ops
// CHECK-DAG:       %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK-DAG:       %[[CB3:.*]] = ttl.bind_cb{cb_index = 3, buffer_factor = 2}
// CHECK:           ttl.init_sfpu(%[[CB0]], %[[CB3]])
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               ttl.tile_regs_acquire
// CHECK:               %[[ADD:.*]] = ttl.tile_add
// CHECK-NEXT:          %{{.*}}, %[[CTILE:.*]] = ttl.copy_tile
// CHECK-NEXT:          %[[MUL:.*]] = ttl.tile_mul %[[ADD]], %[[CTILE]]
// CHECK-NEXT:          %[[EXP:.*]] = ttl.tile_exp %[[MUL]]
// CHECK-NEXT:          %[[INS:.*]] = tensor.insert %[[EXP]]
// CHECK-NEXT:          ttl.tile_regs_commit
// CHECK-NEXT:          ttl.tile_regs_wait
// CHECK-NEXT:          %[[V:.*]] = ttl.cb_reserve %[[CB3]]
// CHECK-NEXT:          ttl.store %[[EXP]], %[[V]]
// CHECK-NEXT:          ttl.tile_regs_release
// CHECK-NEXT:          scf.yield %[[INS]]
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
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %c_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb3 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %exp, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify init_sfpu is inserted even when tile_regs_acquire already exists
// outside the compute. The pre-existing acquire stays outside loops, and new sync
// ops are inserted inside the loop body.
// CHECK-LABEL:   func.func @init_sfpu_with_preexisting_acquire
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK:           %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// Pre-existing acquire remains outside loops
// CHECK:           ttl.tile_regs_acquire
// CHECK-NEXT:      ttl.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK-NEXT:      %[[OUTER:.*]] = scf.for {{.*}} iter_args
// CHECK-NEXT:        %[[INNER:.*]] = scf.for {{.*}} iter_args
// Sync ops inserted inside loop body
// CHECK-NEXT:          ttl.tile_regs_acquire
// CHECK-NEXT:          %[[T0:.*]] = tensor.extract
// CHECK-NEXT:          %[[T1:.*]] = tensor.extract
// CHECK-NEXT:          %[[IDX:.*]] = affine.apply
// CHECK-NEXT:          %[[TOK0:.*]], %[[TILE0:.*]] = ttl.copy_tile %[[T0]]
// CHECK-NEXT:          %[[TOK1:.*]], %[[TILE1:.*]] = ttl.copy_tile %[[T1]]
// CHECK-NEXT:          %[[SUM:.*]] = ttl.tile_add %[[TILE0]], %[[TILE1]] {dst_idx = 0 : i32}
// CHECK-NEXT:          %[[INS:.*]] = tensor.insert %[[SUM]]
// CHECK-NEXT:          ttl.tile_regs_commit
// CHECK-NEXT:          ttl.tile_regs_wait
// CHECK-NEXT:          %[[VIEW:.*]] = ttl.cb_reserve %[[CB2]]
// CHECK-NEXT:          ttl.store %[[SUM]], %[[VIEW]]
// CHECK-NEXT:          ttl.tile_regs_release
// CHECK-NEXT:          scf.yield %[[INS]]
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

  // Pre-inserted tile_regs_acquire without init_sfpu - pass should insert init_sfpu.
  ttl.tile_regs_acquire

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify pre-existing init_sfpu before loops is found and not duplicated.
// The pre-existing sync ops remain outside loops; new sync ops are inserted inside.
// CHECK-LABEL:   func.func @ops_between_acquire_and_compute
// CHECK:           %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2}
// CHECK:           %[[CB2:.*]] = ttl.bind_cb{cb_index = 2, buffer_factor = 2}
// CHECK:           ttl.attach_cb {{.*}}, %[[CB0]]
// Pre-existing sync ops remain outside loops
// CHECK:           ttl.init_sfpu(%[[CB0]], %[[CB2]])
// CHECK-NEXT:      ttl.tile_regs_acquire
// Operations between sync ops and loops (arith.constant hoisted by canonicalization)
// CHECK-NEXT:      %[[EMPTY:.*]] = tensor.empty
// CHECK-NEXT:      %[[OUT_CB:.*]] = ttl.attach_cb %[[EMPTY]], %[[CB2]]
// No duplicate init_sfpu before loops
// CHECK-NOT:       ttl.init_sfpu
// CHECK:           %[[OUTER:.*]] = scf.for {{.*}} iter_args(%[[ACC0:.*]] = %[[OUT_CB]])
// CHECK-NEXT:        %[[INNER:.*]] = scf.for {{.*}} iter_args(%[[ACC1:.*]] = %[[ACC0]])
// Sync ops inserted inside loop body
// CHECK-NEXT:          ttl.tile_regs_acquire
// CHECK-NEXT:          %[[T0:.*]] = tensor.extract
// CHECK-NEXT:          %[[T1:.*]] = tensor.extract
// CHECK-NEXT:          %[[IDX:.*]] = affine.apply
// CHECK-NEXT:          %[[TOK0:.*]], %[[TILE0:.*]] = ttl.copy_tile %[[T0]]
// CHECK-NEXT:          %[[TOK1:.*]], %[[TILE1:.*]] = ttl.copy_tile %[[T1]]
// CHECK-NEXT:          %[[SUM:.*]] = ttl.tile_add %[[TILE0]], %[[TILE1]] {dst_idx = 0 : i32}
// CHECK-NEXT:          %[[INS:.*]] = tensor.insert %[[SUM]]
// CHECK-NEXT:          ttl.tile_regs_commit
// CHECK-NEXT:          ttl.tile_regs_wait
// CHECK-NEXT:          %[[VIEW:.*]] = ttl.cb_reserve %[[CB2]]
// CHECK-NEXT:          ttl.store %[[SUM]], %[[VIEW]]
// CHECK-NEXT:          ttl.tile_regs_release
// CHECK-NEXT:          scf.yield %[[INS]]
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

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %result_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.store %sum, %result_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
