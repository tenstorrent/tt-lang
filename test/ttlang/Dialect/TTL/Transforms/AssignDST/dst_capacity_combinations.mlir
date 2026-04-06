// Verify all 4 capacity combinations: (isFloat32, fullSyncEn).
// Uses a 4x4 unary-only compute (dstPerIteration=1, totalTiles=16).
// unroll_factor = min(capacity, 16).
//
// The pass derives capacity from the IR, not from pass options:
//   - isFloat32: true when tile element type is f32 or fp32_dest_acc_en is set on the function.
//   - fullSyncEn: true when dst_full_sync_en attribute is set on the function.
// Each test case below uses different tile types and compute attributes to
// exercise a different combination.
//
// | isFloat32 | fullSyncEn | capacity | unroll_factor |
// |-----------|------------|----------|---------------|
// | false     | false      | 8        | 8             |
// | true      | false      | 4        | 4             |
// | false     | true       | 16       | 16            |
// | true      | true       | 8        | 8             |
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst))' --split-input-file \
// RUN:   | FileCheck %s

// bf16, double-buffered (default): capacity=8, unroll_factor=8.
// CHECK-LABEL: func.func @bf16_double_buffer
// CHECK: ttl.compute
// CHECK-SAME: ttl.unroll_factor = 8

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @bf16_double_buffer()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>
  %in_ready = ttl.cb_wait %cb0 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %in = ttl.attach_cb %in_ready, %cb0 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%in : tensor<4x4x!ttcore.tile<32x32, bf16>>)
      outs(%out_cb : tensor<4x4x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%in_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %exp = ttl.tile_exp %in_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %exp, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// f32, double-buffered: capacity=4, unroll_factor=4.
// CHECK-LABEL: func.func @f32_double_buffer
// CHECK: ttl.compute
// CHECK-SAME: ttl.unroll_factor = 4

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @f32_double_buffer()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %in_ready = ttl.cb_wait %cb0 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %in = ttl.attach_cb %in_ready, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %empty = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%in : tensor<4x4x!ttcore.tile<32x32, f32>>)
      outs(%out_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%in_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %exp = ttl.tile_exp %in_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<4x4x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb0 : <[4, 4], !ttcore.tile<32x32, f32>, 2>
  return
}

// -----

// bf16, full sync (single-buffered): capacity=16=totalTiles, unroll_factor=16.
// CHECK-LABEL: func.func @bf16_full_sync
// CHECK: ttl.compute
// CHECK-SAME: ttl.unroll_factor = 16

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @bf16_full_sync()
    attributes {dst_full_sync_en = true,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>
  %in_ready = ttl.cb_wait %cb0 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %in = ttl.attach_cb %in_ready, %cb0 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %empty = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, bf16>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%in : tensor<4x4x!ttcore.tile<32x32, bf16>>)
      outs(%out_cb : tensor<4x4x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%in_tile: !ttcore.tile<32x32, bf16>, %out_tile: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %exp = ttl.tile_exp %in_tile : !ttcore.tile<32x32, bf16>
    ttl.tile_store %exp, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %cb0 : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// f32, full sync (single-buffered): capacity=8, unroll_factor=8.
// CHECK-LABEL: func.func @f32_full_sync
// CHECK: ttl.compute
// CHECK-SAME: ttl.unroll_factor = 8

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @f32_full_sync()
    attributes {dst_full_sync_en = true,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %in_ready = ttl.cb_wait %cb0 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %in = ttl.attach_cb %in_ready, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %out = ttl.attach_cb %out_view, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %empty = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  %out_cb = ttl.attach_cb %empty, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%in : tensor<4x4x!ttcore.tile<32x32, f32>>)
      outs(%out_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%in_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %exp = ttl.tile_exp %in_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<4x4x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2>
  ttl.cb_pop %cb0 : <[4, 4], !ttcore.tile<32x32, f32>, 2>
  return
}
