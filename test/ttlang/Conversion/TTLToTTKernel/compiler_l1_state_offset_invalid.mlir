// Invalid compiler-managed SRAM control offsets fail before reset lowering.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --convert-ttl-to-ttkernel

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "invalid_offset">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "invalid_offset">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "invalid_offset">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// Negative control offsets cannot identify an SRAM address.
module attributes {ttl.memory_model = "compiler-sram", ttl.target_arch = #ttcore.arch<blackhole>, ttl.launch_grid = [1, 1], ttl.l1_arena_bytes = 2112 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i32, l1_offset = -1 : i64}]} {
  func.func @negative_state_offset() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    // expected-error @below {{requires a representable compiler-sram state offset}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "invalid_offset">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "invalid_offset">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "invalid_offset">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// Control offsets must fit the 32-bit device-address calculation.
module attributes {ttl.memory_model = "compiler-sram", ttl.target_arch = #ttcore.arch<blackhole>, ttl.launch_grid = [1, 1], ttl.l1_arena_bytes = 2112 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i32, l1_offset = 4294967296 : i64}]} {
  func.func @oversized_state_offset() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    // expected-error @below {{requires a representable compiler-sram state offset}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    return
  }
}
