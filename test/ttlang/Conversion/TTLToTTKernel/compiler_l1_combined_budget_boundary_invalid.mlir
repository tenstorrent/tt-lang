// Verifies that the combined arena and reset state reject a one-byte-short budget.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false l1-budget-override=4223})'

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">
#selected = #ttl.synchronized_dfb_reset<0, participants[#compute, #reader, #writer]>
#all = #ttl.synchronized_dfb_reset<1, participants[#compute, #reader, #writer]>

// expected-error @below {{combined DFB and runtime resources require 4224 L1 bytes but the budget is 4223}}
module attributes {ttl.dfb_allocations = [{block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}, {block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 8 : i64, l1_payload_offset = 2112 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 1 : i32}], ttl.l1_arena_bytes = 4160 : i64, ttl.launch_grid = [1, 1], ttl.memory_model = "compiler-sram", ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs #selected(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.reset_all_dfbs #all
    return
  }
  func.func @read() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #reader, ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs #selected(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.reset_all_dfbs #all
    return
  }
  func.func @write() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #writer, ttl.noc_index = 1 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs #selected(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.reset_all_dfbs #all
    return
  }
}
