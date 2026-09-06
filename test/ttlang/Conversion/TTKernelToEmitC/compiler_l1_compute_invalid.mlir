// Unsupported compute contracts fail before C++ conversion.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc --verify-diagnostics --split-input-file

// A small tile is outside the address-based compute contract.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @small_tile() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<16x32, bf16>>
    %zero = arith.constant 0 : index
    // expected-error @below {{compiler-l1 compute requires 32x32 BF16 or FP32 tiles}}
    ttkernel.copy_tile_init(%storage) : (!ttkernel.cb<1, !ttcore.tile<16x32, bf16>>) -> ()
    return
  }
}

// -----

// Packing without an explicit output index would use Metal descriptor state.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @implicit_pack_index() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    // expected-error @below {{compiler-l1 packing requires an explicit tile index}}
    ttkernel.pack_tile(%zero, %storage, %zero, false) : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
    return
  }
}

// -----

// Synchronization must cover one complete block.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 2 : i64, block_count = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @partial_block() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %one = arith.constant 1 : i32
    // expected-error @below {{compiler-l1 POC requires full-block synchronization to preserve contiguous acquisitions}}
    ttkernel.cb_wait_front(%storage, %one) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }
}

// -----

// Runtime page counts cannot establish the fixed-size transaction contract.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @dynamic_page_count(%pages : i32) attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{compiler-l1 requires a static storage identity and page count}}
    ttkernel.cb_wait_front(%storage, %pages) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }
}

// -----

// Pre-lowered C++ can contain storage effects that the validator cannot classify.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = []} {
  func.func @prelowered_effect() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %unused = ttkernel.get_compile_time_arg_val(0) : () -> i32
    // expected-error @below {{compiler-l1 cannot validate pre-lowered C++ effects}}
    emitc.verbatim "side_effect();"
    return
  }
}

// -----

// The backend requires the finalized allocation table before conversion.
// expected-error @below {{compiler-l1 requires finalized allocation metadata}}
module attributes {ttl.memory_model = "compiler-l1"} {
  func.func @missing_allocation_metadata() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %unused = ttkernel.get_compile_time_arg_val(0) : () -> i32
    return
  }
}

// -----

// Allocation metadata must use the finalized array representation.
// expected-error @below {{compiler-l1 requires finalized allocation metadata}}
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = 0 : i64} {
  func.func @malformed_metadata() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    ttkernel.tile_regs_acquire() : () -> ()
    return
  }
}

// -----

// An integer tile is outside the address-based compute contract.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @integer_tile() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, si32>>
    %zero = arith.constant 0 : index
    // expected-error @below {{compiler-l1 compute requires 32x32 BF16 or FP32 tiles}}
    ttkernel.copy_tile_init(%storage) : (!ttkernel.cb<1, !ttcore.tile<32x32, si32>>) -> ()
    return
  }
}

// -----

// Consumer replacement is outside the address-based compute contract.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @consumer_replacement() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    // expected-error @below {{has no compiler-l1 lowering for ttkernel.pack_waited_tile; Metal DFB fallback is disabled}}
    ttkernel.pack_waited_tile(%zero, %storage, %zero, true) {acquired_tiles = 1 : i64} : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
    return
  }
}
