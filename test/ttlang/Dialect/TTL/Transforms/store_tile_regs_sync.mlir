// This file tests ttl.store handling and DST sync insertion after loop lowering.
// The pass now operates on scf.for loops marked with ttl.tile_loop attribute.
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-lower-to-loops,ttl-insert-tile-regs-sync))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Explicit store with cb_reserve outside compute is reordered after tile_regs_wait.
// After loop lowering, sync ops appear inside scf.for body.
// CHECK-LABEL: func.func @store_reorder_after_wait
// CHECK:         %[[CB:.*]] = ttl.bind_cb
// CHECK-NEXT:    %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB]]
// CHECK-NEXT:    %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB]]
// CHECK-NEXT:    %[[OUT_VIEW_PRE:.*]] = ttl.cb_reserve %[[CB]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB]], %[[CB]])
// CHECK-NEXT:    %[[OUTER:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[OUTPUT_CB]])
// CHECK-NEXT:      %[[INNER:.*]] = scf.for {{.*}} iter_args(%[[ACC2:.*]] = %[[ACC]])
// CHECK-NEXT:        ttl.tile_regs_acquire
// CHECK-NEXT:        %[[IN_TILE:.*]] = tensor.extract %[[ARG_CB]]{{.*}}
// CHECK-NEXT:        %[[TOK:.*]], %[[TILE:.*]] = ttl.copy_tile %[[IN_TILE]]
// CHECK-NEXT:        %[[INSERT:.*]] = tensor.insert %[[TILE]] into %[[ACC2]]{{.*}}
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        ttl.store %[[TILE]], %[[OUT_VIEW_PRE]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK-NEXT:        scf.yield %[[INSERT]]
func.func @store_reorder_after_wait(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view_pre = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      // Store appears before yield - pass should reorder it after tile_regs_wait.
      ttl.store %tile, %out_view_pre : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: multiple cb_reserves with explicit store to last one.
// CHECK-LABEL: func.func @store_with_multiple_reserves
// CHECK:         %[[CB:.*]] = ttl.bind_cb
// CHECK-NEXT:    %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB]]
// CHECK-NEXT:    %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB]], %[[CB]])
// CHECK-NEXT:    %[[OUTER:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[OUTPUT_CB]])
// CHECK-NEXT:      %[[INNER:.*]] = scf.for {{.*}} iter_args(%[[ACC2:.*]] = %[[ACC]])
// CHECK-NEXT:        ttl.tile_regs_acquire
// CHECK-NEXT:        %[[IN_TILE:.*]] = tensor.extract %[[ARG_CB]]{{.*}}
// CHECK-NEXT:        %[[TOK:.*]], %[[TILE:.*]] = ttl.copy_tile %[[IN_TILE]]
// CHECK-NEXT:        %[[INSERT:.*]] = tensor.insert %[[TILE]] into %[[ACC2]]{{.*}}
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        %[[VIEW0:.*]] = ttl.cb_reserve %[[CB]]
// CHECK-NEXT:        %[[VIEW1:.*]] = ttl.cb_reserve %[[CB]]
// CHECK-NEXT:        ttl.store %[[TILE]], %[[VIEW1]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK-NEXT:        scf.yield %[[INSERT]]
func.func @store_with_multiple_reserves(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result = ttl.compute ins(%arg_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%in: !ttcore.tile<32x32, bf16>,
       %out: !ttcore.tile<32x32, bf16>):
    %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    %view0 = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %view1 = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %tile, %view1 : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// cb_reserve inside compute body with explicit store.
// CHECK-LABEL: func.func @store_with_reserve_inside_compute
// CHECK:         %[[CB:.*]] = ttl.bind_cb
// CHECK-NEXT:    %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB]]
// CHECK-NEXT:    %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB]], %[[CB]])
// CHECK-NEXT:    %[[OUTER:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[OUTPUT_CB]])
// CHECK-NEXT:      %[[INNER:.*]] = scf.for {{.*}} iter_args(%[[ACC2:.*]] = %[[ACC]])
// CHECK-NEXT:        ttl.tile_regs_acquire
// CHECK-NEXT:        %[[IN_TILE:.*]] = tensor.extract %[[ARG_CB]]{{.*}}
// CHECK-NEXT:        %[[TOK:.*]], %[[TILE:.*]] = ttl.copy_tile %[[IN_TILE]]
// CHECK-NEXT:        %[[INSERT:.*]] = tensor.insert %[[TILE]] into %[[ACC2]]{{.*}}
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        %[[OUT_VIEW:.*]] = ttl.cb_reserve %[[CB]]
// CHECK-NEXT:        ttl.store %[[TILE]], %[[OUT_VIEW]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK-NEXT:        scf.yield %[[INSERT]]
func.func @store_with_reserve_inside_compute(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
    %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    // cb_reserve and store both inside compute body.
    %out_view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %tile, %out_view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: explicit store using parent cb_reserve.
// CHECK-LABEL: func.func @store_explicit_using_parent_reserve
// CHECK:         %[[CB:.*]] = ttl.bind_cb
// CHECK-NEXT:    %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB]]
// CHECK-NEXT:    %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB]]
// CHECK-NEXT:    %[[OUT_VIEW_PARENT:.*]] = ttl.cb_reserve %[[CB]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB]], %[[CB]])
// CHECK-NEXT:    %[[OUTER:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[OUTPUT_CB]])
// CHECK-NEXT:      %[[INNER:.*]] = scf.for {{.*}} iter_args(%[[ACC2:.*]] = %[[ACC]])
// CHECK-NEXT:        ttl.tile_regs_acquire
// CHECK-NEXT:        %[[IN_TILE:.*]] = tensor.extract %[[ARG_CB]]{{.*}}
// CHECK-NEXT:        %[[TOK:.*]], %[[TILE:.*]] = ttl.copy_tile %[[IN_TILE]]
// CHECK-NEXT:        %[[INSERT:.*]] = tensor.insert %[[TILE]] into %[[ACC2]]{{.*}}
// CHECK-NEXT:        ttl.tile_regs_commit
// CHECK-NEXT:        ttl.tile_regs_wait
// CHECK-NEXT:        ttl.store %[[TILE]], %[[OUT_VIEW_PARENT]]
// CHECK-NEXT:        ttl.tile_regs_release
// CHECK-NEXT:        scf.yield %[[INSERT]]
func.func @store_explicit_using_parent_reserve(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // cb_reserve in parent block - store explicitly uses this view.
  %out_view_parent = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      // Explicit store using parent's cb_reserve.
      ttl.store %tile, %out_view_parent : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
