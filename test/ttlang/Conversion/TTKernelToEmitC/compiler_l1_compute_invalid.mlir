// Unsupported compute contracts fail before C++ conversion.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc --verify-diagnostics --split-input-file

// small tile is outside the address-based compute contract.
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

// integer tile is outside the address-based compute contract.
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

// consumer replacement is outside the address-based compute contract.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @consumer_replacement() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    // expected-error @below {{has no compiler-l1 lowering for ttkernel.pack_waited_tile; Metal DFB fallback is disabled}}
    ttkernel.pack_waited_tile(%zero, %storage, %zero, true) {acquired_tiles = 1 : i64} : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
    return
  }
}
