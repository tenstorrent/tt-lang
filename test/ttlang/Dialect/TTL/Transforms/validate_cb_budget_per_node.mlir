// Verifies that finalized DFB residency is accounted independently per launch
// node. Each node requires 8192 bytes even though the global sum is 16384.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=10000})' -o /dev/null

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
