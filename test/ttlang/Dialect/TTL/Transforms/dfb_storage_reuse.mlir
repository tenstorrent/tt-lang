// Tests byte-addressed storage reuse across distinct physical DFB formats.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// Different element types require distinct physical descriptors. Their
// strictly ordered lifetimes may still share one backing storage allocation.
// CHECK: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32},
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, f32>, num_tiles = 1 : i32, page_size = 4096 : i32, storage_index = 0 : i32}
// CHECK-SAME: ]
// CHECK-LABEL: func.func @ordered_mixed_formats
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0,
// CHECK: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 1,

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @ordered_mixed_formats()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>

    %first_produced = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %first_consumed = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>

    %second_produced = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    %second_consumed = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    return
  }
}

// -----

// Overlapping lifetimes retain separate backing allocations even when their
// descriptors have different element types.
// CHECK: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 1 : i32},
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, f32>, num_tiles = 1 : i32, page_size = 4096 : i32, storage_index = 0 : i32}
// CHECK-SAME: ]
// CHECK-LABEL: func.func @overlapping_mixed_formats
// CHECK-SAME: ttl.base_cta_index = 2 : i32

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @overlapping_mixed_formats()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>

    %first_produced = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second_produced = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    %first_consumed = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second_consumed = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    return
  }
}
