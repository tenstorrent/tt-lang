// Tests byte-addressed storage reuse across distinct physical DFB formats.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=NO-REUSE

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

// NO-REUSE: module attributes {ttl.dfb_allocations = [
// NO-REUSE-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32},
// NO-REUSE-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, f32>, num_tiles = 1 : i32, page_size = 4096 : i32, storage_index = 1 : i32}
// NO-REUSE-SAME: ]
// NO-REUSE-LABEL: func.func @ordered_mixed_formats

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

// Storage placement compares target-rounded allocation sizes. Sharing the
// 48-byte descriptor with the 72-byte descriptor needs 96 bytes, while sharing
// it with the 120-byte descriptor needs 144 bytes. Both changes add 24 payload
// bytes, but their 64-byte-rounded totals are 256 and 320 bytes respectively.
// CHECK: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 5 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bfp_bf4>, num_tiles = 1 : i32, page_size = 24 : i32, storage_index = 0 : i32},
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 3 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<1x16, bfp_bf4>, num_tiles = 1 : i32, page_size = 24 : i32, storage_index = 1 : i32},
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 2 : i32, element_type = !ttcore.tile<2x16, bfp_bf8>, num_tiles = 1 : i32, page_size = 48 : i32, storage_index = 1 : i32}
// CHECK-SAME: ]
// CHECK-LABEL: func.func @target_rounded_storage_placement

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @target_rounded_storage_placement()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %large = ttl.bind_cb {cb_index = 0, block_count = 5} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bfp_bf4>, 5>
    %medium = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bfp_bf4>, 3>
    %small = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<2x16, bfp_bf8>, 1>

    %large_produced = ttl.cb_reserve %large
        : <[1, 1], !ttcore.tile<1x16, bfp_bf4>, 5>
          -> tensor<1x1x!ttcore.tile<1x16, bfp_bf4>>
    ttl.cb_push %large : <[1, 1], !ttcore.tile<1x16, bfp_bf4>, 5>
    %medium_produced = ttl.cb_reserve %medium
        : <[1, 1], !ttcore.tile<1x16, bfp_bf4>, 3>
          -> tensor<1x1x!ttcore.tile<1x16, bfp_bf4>>
    ttl.cb_push %medium : <[1, 1], !ttcore.tile<1x16, bfp_bf4>, 3>
    %large_consumed = ttl.cb_wait %large
        : <[1, 1], !ttcore.tile<1x16, bfp_bf4>, 5>
          -> tensor<1x1x!ttcore.tile<1x16, bfp_bf4>>
    ttl.cb_pop %large : <[1, 1], !ttcore.tile<1x16, bfp_bf4>, 5>
    %medium_consumed = ttl.cb_wait %medium
        : <[1, 1], !ttcore.tile<1x16, bfp_bf4>, 3>
          -> tensor<1x1x!ttcore.tile<1x16, bfp_bf4>>
    ttl.cb_pop %medium : <[1, 1], !ttcore.tile<1x16, bfp_bf4>, 3>

    %small_produced = ttl.cb_reserve %small
        : <[1, 1], !ttcore.tile<2x16, bfp_bf8>, 1>
          -> tensor<1x1x!ttcore.tile<2x16, bfp_bf8>>
    ttl.cb_push %small : <[1, 1], !ttcore.tile<2x16, bfp_bf8>, 1>
    %small_consumed = ttl.cb_wait %small
        : <[1, 1], !ttcore.tile<2x16, bfp_bf8>, 1>
          -> tensor<1x1x!ttcore.tile<2x16, bfp_bf8>>
    ttl.cb_pop %small : <[1, 1], !ttcore.tile<2x16, bfp_bf8>, 1>
    return
  }
}

// -----

// Descriptors used on disjoint launch nodes may share storage without a
// program-order relation because no worker allocates both descriptors.
// CHECK: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32},
// CHECK-SAME: {allocation_nodes = {{\[\[1, 0\]\]}}, block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, f32>, num_tiles = 1 : i32, page_size = 4096 : i32, storage_index = 0 : i32}
// CHECK-SAME: ]
// CHECK-LABEL: func.func @disjoint_launch_nodes

module attributes {ttl.launch_grid = [2, 1]} {
  func.func @disjoint_launch_nodes()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %first_node = arith.cmpi eq, %core_x, %zero : index
    %second_node = arith.cmpi ne, %core_x, %zero : index
    scf.if %first_node {
      ttl.opaque_call "use_first" dfb_dependencies(
          %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          () {header = "effects.hpp"} : () -> ()
    }
    scf.if %second_node {
      ttl.opaque_call "use_second" dfb_dependencies(
          %second : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
          () {header = "effects.hpp"} : () -> ()
    }
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
