// Idempotency tests: running ttl-insert-tile-regs-sync twice produces the
// same output as running it once (no duplicate sync ops).
// RUN: ttlang-opt %s --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(ttl-insert-tile-regs-sync,ttl-insert-tile-regs-sync))' \
// RUN:   | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Non-subblocked: sync ops inside compute body. Running the pass twice
// should not duplicate acquire/commit/wait/release.
// CHECK-LABEL: func.func @idempotent_non_subblocked
// CHECK:         ttl.compute
// CHECK:         ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, bf16>):
// CHECK-NEXT:      ttl.tile_regs_acquire
// CHECK:           ttl.iter_index
// CHECK:           ttl.iter_index
// CHECK:           %{{.*}}, %[[TILE:.*]] = ttl.copy_tile %[[IN]]
// CHECK-NEXT:      ttl.tile_regs_commit
// CHECK-NEXT:      ttl.tile_regs_wait
// CHECK-NEXT:      ttl.tile_store
// CHECK-NEXT:      ttl.tile_regs_release
// CHECK-NEXT:      ttl.yield
func.func @idempotent_non_subblocked(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %tok, %tile = ttl.copy_tile %in[%c0], %c0 : !ttcore.tile<32x32, bf16>, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      ttl.tile_store %tile, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Subblocked: sync ops outside compute body. Running the pass twice
// should not duplicate acquire/commit/wait/release.
// CHECK-LABEL: func.func @idempotent_subblocked
// CHECK:         ttl.tile_regs_acquire
// CHECK-NEXT:    %{{.*}} = ttl.compute
// CHECK:         ttl.tile_regs_commit
// CHECK-NEXT:    ttl.tile_regs_wait
// CHECK-NEXT:    ttl.tile_regs_release
// CHECK-NOT:     ttl.tile_regs_commit
// CHECK-NOT:     ttl.tile_regs_wait
// CHECK-NOT:     ttl.tile_regs_release
// CHECK:         return
func.func @idempotent_subblocked(%arg0: tensor<1x8x!ttcore.tile<32x32, f32>>) -> tensor<1x8x!ttcore.tile<32x32, f32>> {
  %0 = tensor.empty() : tensor<1x8x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x8x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x8x!ttcore.tile<32x32, f32>>
  %out_cb = ttl.attach_cb %0, %cb1 : (tensor<1x8x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x8x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb1 : <[1, 8], !ttcore.tile<32x32, f32>, 2> -> tensor<1x8x!ttcore.tile<32x32, f32>>
  %result = ttl.compute ins(%arg_cb : tensor<1x8x!ttcore.tile<32x32, f32>>) outs(%out_cb : tensor<1x8x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], ttl.full_linearization_strides = array<i64: 8, 1>} {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %c0 = arith.constant 0 : index
      %tok, %tile = ttl.copy_tile %in[%c0], %c0 {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %exp = ttl.tile_exp %tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
      ttl.tile_store %exp, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<1x8x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x8x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<1x8x!ttcore.tile<32x32, f32>>
}
