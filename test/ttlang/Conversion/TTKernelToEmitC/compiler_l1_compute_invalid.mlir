// Unsupported compute contracts fail before C++ conversion.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc --verify-diagnostics --split-input-file

// A small tile is outside the address-based compute contract.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @small_tile() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<16x32, bf16>>
    %zero = arith.constant 0 : index
    // expected-error @below {{compiler-l1 compute requires 32x32 BF16 or FP32 tiles}}
    ttkernel.copy_tile_init(%storage) : (!ttkernel.cb<1, !ttcore.tile<16x32, bf16>>) -> ()
    return
  }
}

// -----

// Every allocation entry must contain the geometry and ordered offsets used by generated address types.
// expected-error @below {{'builtin.module' op compiler-l1 allocation entry 0 must define positive uint32 page_size, num_tiles, and block_count values, storage_capacity_pages at least num_tiles times block_count and less than 2^31, and representable ordered L1 offsets}}
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 64 : i64, l1_payload_offset = 32 : i64, num_tiles = 1 : i64, page_size = 2048 : i64}]} {
  func.func @invalid_allocation_offsets() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Every allocation entry must define the physical ring capacity.
// expected-error @below {{'builtin.module' op compiler-l1 allocation entry 0 must define positive uint32 page_size, num_tiles, and block_count values, storage_capacity_pages at least num_tiles times block_count and less than 2^31, and representable ordered L1 offsets}}
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{block_count = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 2048 : i64}]} {
  func.func @missing_storage_capacity() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Shared storage capacity must cover the logical DFB capacity.
// expected-error @below {{'builtin.module' op compiler-l1 allocation entry 0 must define positive uint32 page_size, num_tiles, and block_count values, storage_capacity_pages at least num_tiles times block_count and less than 2^31, and representable ordered L1 offsets}}
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{block_count = 2 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 2 : i64, page_size = 2048 : i64, storage_capacity_pages = 3 : i64}]} {
  func.func @invalid_storage_capacity() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// An external descriptor must agree with the finalized allocation geometry.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{block_count = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @descriptor_geometry_mismatch() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.opaque_call' op compiler-l1 descriptor geometry differs from its allocation metadata}}
    ttkernel.opaque_call "describe" template_args [#ttkernel.dfb_descriptor<0, 2, 1, 2048>] () {dfb_resource_indices = array<i32: 0>, header = "describe.hpp"} : () -> ()
    return
  }
}

// -----

// Compute descriptors require a supported tile type in the allocation table.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{block_count = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @descriptor_missing_element_type() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    // expected-error @below {{'ttkernel.opaque_call' op compiler-l1 compute descriptors require 32x32 BF16 or FP32 tiles}}
    ttkernel.opaque_call "describe" template_args [#ttkernel.dfb_descriptor<0, 1, 1, 2048>] () {dfb_resource_indices = array<i32: 0>, header = "describe.hpp"} : () -> ()
    return
  }
}

// -----

// Descriptor indices must identify an allocation-table entry.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{block_count = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @descriptor_index_out_of_range() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.opaque_call' op compiler-l1 descriptor index is absent from allocation metadata}}
    ttkernel.opaque_call "describe" template_args [#ttkernel.dfb_descriptor<1, 1, 1, 2048>] () {dfb_resource_indices = array<i32: 1>, header = "describe.hpp"} : () -> ()
    return
  }
}

// -----

// Packing without an explicit output index would use Metal descriptor state.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
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
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 2 : i64, block_count = 1 : i64, storage_capacity_pages = 2 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @partial_block() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %one = arith.constant 1 : i32
    // expected-error @below {{compiler-l1 requires one complete block or the complete tensor-backed capacity per synchronization operation}}
    ttkernel.cb_wait_front(%storage, %one) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }
}

// -----

// Runtime page counts cannot establish the fixed-size transaction contract.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @dynamic_page_count(%pages : i32) attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{compiler-l1 requires a static storage identity and page count}}
    ttkernel.cb_wait_front(%storage, %pages) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }
}

// -----

// A DFB operation without an address-based lowering must not use the Metal implementation.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @unsupported_dfb_operation() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttkernel.compute_kernel_hw_startup' op has no compiler-l1 lowering for ttkernel.compute_kernel_hw_startup; Metal DFB fallback is disabled}}
    ttkernel.compute_kernel_hw_startup(%storage, %storage) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>) -> ()
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
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [{cb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @integer_tile() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, si32>>
    %zero = arith.constant 0 : index
    // expected-error @below {{compiler-l1 compute requires 32x32 BF16 or FP32 tiles}}
    ttkernel.copy_tile_init(%storage) : (!ttkernel.cb<1, !ttcore.tile<32x32, si32>>) -> ()
    return
  }
}

// -----

// Tensor-backed storage requires the finalized tensor argument mapping.
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = [
  {block_count = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, l1_offset = 0 : i64, num_tiles = 1 : i64, page_size = 2048 : i64, storage_capacity_pages = 1 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]}
]} {
  func.func @missing_tensor_runtime_argument() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.get_compile_time_arg_val' op compiler-l1 tensor backing is absent from the kernel's common tensor arguments}}
    %dfb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}
