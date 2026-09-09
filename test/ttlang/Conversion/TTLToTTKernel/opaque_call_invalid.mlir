// Verify that opaque-call lowering rejects unrepresentable DFB descriptors.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --convert-ttl-to-ttkernel

// A descriptor requires a byte-addressable DFB page size.
func.func @sub_byte_descriptor() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 2, block_count = 4} : !ttl.cb<[2, 3], i4, 4>
  // expected-error @below {{'ttl.opaque_call' op DFB descriptor element type must occupy a positive whole number of bytes, got 'i4'}}
  ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%dfb : !ttl.cb<[2, 3], i4, 4>) () {header = "describe.hpp"} : () -> ()
  return
}

// -----

// Descriptor page counts must fit the generated uint32_t template parameter.
func.func @descriptor_page_count_overflow() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[4294967296], i8, 1>
  // expected-error @below {{'ttl.opaque_call' op DFB descriptor dimensions or page size exceed uint32_t}}
  ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%dfb : !ttl.cb<[4294967296], i8, 1>) () {header = "describe.hpp"} : () -> ()
  return
}

// -----

// Compiler-managed external calls use address-bearing descriptors instead of Metal indices.
module attributes {ttl.memory_model = "compiler-l1", ttl.l1_arena_bytes = 2112 : i64, ttl.dfb_allocations = [{block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}]} {
  func.func @compiler_l1_metal_index() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{compiler-l1 external calls cannot use Metal DFB indices; use ttl.dfb_descriptor()}}
    ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_index, 0>] template_dfbs(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) () {header = "describe.hpp"} : () -> ()
    return
  }
}

// -----

// Compiler-managed allocation requires explicit effects for external DFB access.
module attributes {ttl.memory_model = "compiler-l1", ttl.l1_arena_bytes = 0 : i64, ttl.dfb_allocations = []} {
  func.func @compiler_l1_unknown_effects() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{compiler-l1 requires typed DFB effects for external calls}}
    ttl.opaque_call "unknown" () {header = "unknown.hpp", unknown_dfb_access} : () -> ()
    return
  }
}
