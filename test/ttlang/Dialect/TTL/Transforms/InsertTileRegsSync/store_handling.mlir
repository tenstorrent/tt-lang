// Tests for ttl.store handling in TTLInsertTileRegsSync with loop-based approach.
// The pass operates on scf.for loops marked with ttl.tile_loop attribute.
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-lower-to-loops,ttl-insert-tile-regs-sync))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Explicit store with cb_reserve outside compute is reordered after tile_regs_wait.
// CHECK-LABEL: func.func @store_reorder_after_wait
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[CB:.*]] = ttl.bind_cb{cb_index = 0
// CHECK:         %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB]]
// CHECK:         %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB]]
// CHECK-NEXT:    %[[OUT_VIEW:.*]] = ttl.cb_reserve %[[CB]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB]], %[[CB]])
// CHECK-NEXT:    %{{.*}} = scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ACC1:.*]] = %[[OUTPUT_CB]])
// CHECK-NEXT:      %{{.*}} = scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ACC2:.*]] = %[[ACC1]])
// CHECK-NEXT:        ttl.tile_regs_acquire
// CHECK-NEXT:        %[[TILE:.*]] = tensor.extract %[[ARG_CB]][%[[I]], %[[J]]]
// CHECK-NEXT:        %[[TOK:.*]], %[[DST_TILE:.*]] = ttl.copy_tile %[[TILE]], %[[C0]], %[[C0]]
// CHECK-NEXT:        %[[INS:.*]] = tensor.insert %[[DST_TILE]] into %[[ACC2]][%[[I]], %[[J]]]
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        ttl.store %[[DST_TILE]], %[[OUT_VIEW]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK-NEXT:        scf.yield %[[INS]]
func.func @store_reorder_after_wait(%arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      ttl.store %tile, %out_view : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Explicit reserve and push outside compute - pass reuses them, doesn't create new ones.
// CHECK-LABEL: func.func @explicit_reserve_push_outside
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[CB:.*]] = ttl.bind_cb{cb_index = 0
// CHECK:         %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB]]
// CHECK:         %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB]]
// CHECK-NEXT:    %[[VIEW:.*]] = ttl.cb_reserve %[[CB]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB]], %[[CB]])
// CHECK-NEXT:    %{{.*}} = scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT:      %{{.*}} = scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT:        ttl.tile_regs_acquire
// CHECK:             ttl.copy_tile
// CHECK:             ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        ttl.store %{{.*}}, %[[VIEW]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK:         ttl.cb_push %[[CB]]
// CHECK-NOT:     ttl.cb_reserve
// CHECK-NOT:     ttl.cb_push
// CHECK:         return
func.func @explicit_reserve_push_outside(%arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %view = ttl.cb_reserve %cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      ttl.store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Auto-inserted reserve goes BEFORE outermost loop, push goes AFTER.
// CHECK-LABEL: func.func @auto_reserve_push_placement
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[CB0:.*]] = ttl.bind_cb{cb_index = 0
// CHECK:         %[[CB1:.*]] = ttl.bind_cb{cb_index = 1
// CHECK:         %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB0]]
// CHECK:         %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB1]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB0]], %[[CB1]])
// CHECK-NEXT:    %[[VIEW:.*]] = ttl.cb_reserve %[[CB1]]
// CHECK-NEXT:    %{{.*}} = scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT:      %{{.*}} = scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT:        ttl.tile_regs_acquire
// CHECK:             ttl.copy_tile
// CHECK:             ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        ttl.store %{{.*}}, %[[VIEW]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK:       ttl.cb_push %[[CB1]]
// CHECK-NEXT:    return
func.func @auto_reserve_push_placement(%arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  %arg_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  // No explicit reserve/store/push - pass should auto-insert them
  %result = ttl.compute ins(%arg_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Explicit store inside compute body with explicit reserve/push - all preserved.
// CHECK-LABEL: func.func @explicit_store_with_reserve_push
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[CB0:.*]] = ttl.bind_cb{cb_index = 0
// CHECK:         %[[CB1:.*]] = ttl.bind_cb{cb_index = 1
// CHECK:         %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB0]]
// CHECK:         %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB1]]
// CHECK-NEXT:    %[[VIEW:.*]] = ttl.cb_reserve %[[CB1]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB0]], %[[CB1]])
// CHECK-NEXT:    %{{.*}} = scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT:      %{{.*}} = scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT:        ttl.tile_regs_acquire
// CHECK:             ttl.copy_tile
// CHECK:             ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        ttl.store %{{.*}}, %[[VIEW]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK:       ttl.cb_push %[[CB1]]
// CHECK-NEXT:    return
func.func @explicit_store_with_reserve_push(%arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  %arg_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %view = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<2x2x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      ttl.store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
      ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, bf16>>
}
