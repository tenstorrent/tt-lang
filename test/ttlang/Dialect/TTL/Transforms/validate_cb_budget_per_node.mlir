// Verifies that finalized DFB residency is accounted independently per launch
// node. Each node requires 8192 bytes even though the global sum is 16384.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=10000})' -o /dev/null

module attributes {
  ttl.dfb_allocations = [
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>,
     num_tiles = 4 : i32, page_size = 2048 : i32,
     storage_index = 0 : i32},
    {allocation_nodes = [[1, 0]], block_count = 1 : i32,
     dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, bf16>,
     num_tiles = 4 : i32, page_size = 2048 : i32,
     storage_index = 1 : i32}
  ],
  ttl.launch_grid = [2, 1]
} {
  func.func @disjoint_node_allocations() {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[4, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[4, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}

// -----

// Static storage contributes only on its segment nodes. The tensor-backed
// segment on node (0,0) does not add the mixed descriptor's 2048 bytes to that
// node, whose independent static descriptor consumes the complete budget.

module attributes {
  ttl.dfb_allocations = [
    {allocation_nodes = [[0, 0], [1, 0]], block_count = 1 : i32,
     dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>,
     num_tiles = 1 : i32, page_size = 2048 : i32,
     storage_index = 0 : i32,
     storage_segments = [
       {nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<
          tensor_index = 0, byte_offset = 0, byte_size = 2048>},
       {nodes = [[1, 0]]}]},
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, bf16>,
     num_tiles = 4 : i32, page_size = 2048 : i32,
     storage_index = 1 : i32}
  ],
  ttl.launch_grid = [2, 1]
} {
  func.func @mixed_tensor_and_static_segments() {
    %tensor_backed = ttl.bind_cb {cb_index = 0, block_count = 1}
        {tensor_backing = #ttl.tensor_backing<
          tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %static = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %independent = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[4, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
