// Verifies that malformed compiler-managed lifecycle metadata fails before lowering.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})'

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "duplicate_metadata">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "duplicate_metadata">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "duplicate_metadata">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// Duplicate entries make the terminal-state reset plan ambiguous.
// expected-error @below {{'builtin.module' op contains duplicate compiler-l1 reconfiguration reset metadata}}
module attributes {ttl.compiler_l1_reconfiguration_resets = [{dfb_indices = array<i32: 0>, ordinal = 0 : i64}, {dfb_indices = array<i32: 0>, ordinal = 0 : i64}], ttl.dfb_allocations = [{block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}], ttl.l1_arena_bytes = 2112 : i64, ttl.launch_grid = [1, 1], ttl.memory_model = "compiler-l1", ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "unknown_index">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "unknown_index">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "unknown_index">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// Every reset index must identify a finalized allocation.
// expected-error @below {{'builtin.module' op compiler-l1 reconfiguration ordinal 0 references unknown DFB index 1}}
module attributes {ttl.compiler_l1_reconfiguration_resets = [{dfb_indices = array<i32: 1>, ordinal = 0 : i64}], ttl.dfb_allocations = [{block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}], ttl.l1_arena_bytes = 2112 : i64, ttl.launch_grid = [1, 1], ttl.memory_model = "compiler-l1", ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "noncanonical_indices">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "noncanonical_indices">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "noncanonical_indices">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// Sorted unique indices make reset emission deterministic.
// expected-error @below {{'builtin.module' op contains noncanonical compiler-l1 reconfiguration reset indices}}
module attributes {ttl.compiler_l1_reconfiguration_resets = [{dfb_indices = array<i32: 0, 0>, ordinal = 0 : i64}], ttl.dfb_allocations = [{block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}], ttl.l1_arena_bytes = 2112 : i64, ttl.launch_grid = [1, 1], ttl.memory_model = "compiler-l1", ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    return
  }
}
