// This file tests ttl.tile_store handling in TTLInsertTileRegsSync.
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-insert-tile-regs-sync))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Explicit store with cb_reserve outside compute is reordered after tile_regs_wait.
// CHECK-LABEL: func.func @store_reorder_after_wait
// CHECK:         %[[CB:.*]] = ttl.bind_cb
// CHECK-NEXT:    %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB]]
// CHECK-NEXT:    %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB]]
// CHECK-NEXT:    %[[OUT_VIEW_PRE:.*]] = ttl.cb_reserve %[[CB]]
// CHECK:         ttl.init_sfpu(%[[CB]], %[[CB]])
// CHECK:         %[[RES:.*]] = ttl.compute
// CHECK:         ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, bf16>):
// CHECK-NEXT:      ttl.tile_regs_acquire
// CHECK-NEXT:      %[[TOK:.*]], %[[TILE:.*]] = ttl.copy_tile %[[IN]]
// CHECK-NEXT:      ttl.tile_regs_commit
// CHECK-NEXT:      ttl.tile_regs_wait
// CHECK-NEXT:      ttl.tile_store %[[TILE]], %[[OUT_VIEW_PRE]]
// CHECK-NEXT:      ttl.tile_regs_release
// CHECK-NEXT:      ttl.yield
// CHECK-NEXT:    } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK-NEXT:    return %[[RES]]
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
      ttl.tile_store %tile, %out_view_pre : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// cb_reserve inside compute body with explicit store.
// The cb_reserve stays before tile_regs_commit; store moves after tile_regs_wait.
// CHECK-LABEL: func.func @store_with_reserve_inside_compute
// CHECK:         %[[CB:.*]] = ttl.bind_cb
// CHECK-NEXT:    %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB]]
// CHECK-NEXT:    %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB]]
// CHECK-NEXT:    %[[OUT_VIEW:.*]] = ttl.cb_reserve %[[CB]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB]], %[[CB]])
// CHECK-NEXT:    %[[RES:.*]] = ttl.compute
// CHECK:         ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, bf16>):
// CHECK-NEXT:      ttl.tile_regs_acquire
// CHECK-NEXT:      %[[TOK:.*]], %[[TILE:.*]] = ttl.copy_tile %[[IN]]
// CHECK-NEXT:      ttl.tile_regs_commit
// CHECK-NEXT:      ttl.tile_regs_wait
// CHECK-NEXT:      ttl.tile_store %[[TILE]], %[[OUT_VIEW]]
// CHECK-NEXT:      ttl.tile_regs_release
// CHECK-NEXT:      ttl.yield
// CHECK-NEXT:    } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK-NEXT:    return %[[RES]]
func.func @store_with_reserve_inside_compute(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
    %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
    ttl.tile_store %tile, %out_view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Explicit store with parent cb_reserve. No auto-insert needed.
// CHECK-LABEL: func.func @store_explicit_with_parent_reserve
// CHECK:         %[[CB:.*]] = ttl.bind_cb
// CHECK-NEXT:    %[[ARG_CB:.*]] = ttl.attach_cb %arg0, %[[CB]]
// CHECK-NEXT:    %[[OUTPUT:.*]] = tensor.empty
// CHECK-NEXT:    %[[OUTPUT_CB:.*]] = ttl.attach_cb %[[OUTPUT]], %[[CB]]
// CHECK-NEXT:    %[[OUT_VIEW:.*]] = ttl.cb_reserve %[[CB]]
// CHECK-NEXT:    ttl.init_sfpu(%[[CB]], %[[CB]])
// CHECK-NEXT:    %[[RES:.*]] = ttl.compute
// CHECK:         ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, bf16>):
// CHECK-NEXT:      ttl.tile_regs_acquire
// CHECK-NEXT:      %[[TOK:.*]], %[[TILE:.*]] = ttl.copy_tile %[[IN]]
// CHECK-NEXT:      ttl.tile_regs_commit
// CHECK-NEXT:      ttl.tile_regs_wait
// CHECK-NEXT:      ttl.tile_store %[[TILE]], %[[OUT_VIEW]]
// CHECK-NEXT:      ttl.tile_regs_release
// CHECK-NEXT:      ttl.yield
// CHECK-NEXT:    } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK-NEXT:    return %[[RES]]
func.func @store_explicit_with_parent_reserve(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      ttl.tile_store %tile, %out_view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
