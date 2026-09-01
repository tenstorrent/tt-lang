// Verifies that allocations resident on the same launch node are summed.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=10000})'

module attributes {
  ttl.dfb_allocations = [
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>,
     num_tiles = 4 : i32, page_size = 2048 : i32,
     storage_index = 0 : i32},
    {allocation_nodes = [[0, 0]], block_count = 1 : i32,
     dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, bf16>,
     num_tiles = 4 : i32, page_size = 2048 : i32,
     storage_index = 1 : i32}
  ],
  ttl.launch_grid = [2, 1]
} {
  func.func @same_node_allocations() {
    // expected-error @below {{total DFB allocation (16384 bytes) exceeds L1 budget (10000 bytes)}}
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[4, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[4, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
