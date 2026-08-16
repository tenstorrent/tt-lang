// Tests finalized node-specific storage metadata for tensor-backed DFBs.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefixes=COMMON,REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefixes=COMMON,DISTINCT
// RUN: ttlang-opt %s --split-input-file --ttl-validate-cb-budget='l1-budget-override=1' -o /dev/null

// COMMON: module attributes {ttl.dfb_allocations = [{allocation_nodes = {{\[\[0, 0\], \[1, 0\]\]}}, block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_segments = [{nodes = {{\[\[0, 0\], \[1, 0\]\]}}, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]}], ttl.launch_grid = array<i64: 2, 1>}
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  // COMMON-LABEL: func.func @publish_input
  func.func @publish_input()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i32} {
    // COMMON: ttl.bind_cb{{.*}}tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %view = ttl.cb_reserve %dfb {num_tiles = 1 : i64} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb {num_tiles = 1 : i64} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// DFBs with different tensor backing may share one hardware index when their
// exact launch-node domains are disjoint.

// REUSE: module attributes {ttl.dfb_allocations = [{allocation_nodes = {{\[\[0, 0\], \[1, 0\]\]}}, block_count = 2 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_segments = [{nodes = {{\[\[0, 0\]\]}}, tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 64>}, {nodes = {{\[\[1, 0\]\]}}, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}]}], ttl.launch_grid = array<i64: 2, 1>}
// REUSE-LABEL: func.func @per_node_backing
// REUSE: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {{.*}}tensor_index = 0
// REUSE: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {{.*}}tensor_index = 1

// DISTINCT: module attributes {ttl.dfb_allocations = [{allocation_nodes = {{\[\[1, 0\]\]}}, block_count = 2 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_segments = [{nodes = {{\[\[1, 0\]\]}}, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}]}, {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 2 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_segments = [{nodes = {{\[\[0, 0\]\]}}, tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 64>}]}], ttl.launch_grid = array<i64: 2, 1>}
// DISTINCT-LABEL: func.func @per_node_backing
// DISTINCT: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {{.*}}tensor_index = 0
// DISTINCT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {{.*}}tensor_index = 1
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @per_node_backing()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_node_zero = arith.cmpi eq, %core_x, %zero : index
    %is_node_one = arith.cmpi eq, %core_x, %one : index
    scf.if %is_node_zero {
      %second_slot = ttl.cb_reserve %second
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %second : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    scf.if %is_node_one {
      %first_slot = ttl.cb_reserve %first
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %first : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }
}
