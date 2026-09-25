// Unsupported compute contracts fail before C++ conversion.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc --verify-diagnostics --split-input-file

// A small tile is outside the address-based compute contract.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<16x32, bf16>, page_size = 1024 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 1024 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @small_tile() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<16x32, bf16>>
    %zero = arith.constant 0 : index
    // expected-error @below {{compiler-sram compute requires 32x32 BF16 or FP32 tiles}}
    ttkernel.copy_tile_init(%storage) : (!ttkernel.cb<1, !ttcore.tile<16x32, bf16>>) -> ()
    return
  }
}

// -----

// Storage metadata queries require a tile element type before conversion.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @scalar_tile_size() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, f32>
    // expected-error @below {{'ttkernel.get_tile_size' op compiler-sram requires tiled storage metadata}}
    %size = ttkernel.get_tile_size(%storage) : (!ttkernel.cb<1, f32>) -> i32
    return
  }
}

// -----

// Data-format queries use the same tile metadata contract.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @scalar_data_format() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, f32>
    // expected-error @below {{'ttkernel.get_dataformat' op compiler-sram requires tiled storage metadata}}
    %format = ttkernel.get_dataformat(%storage) : (!ttkernel.cb<1, f32>) -> !ttkernel.DataFormat
    return
  }
}

// -----

// Every allocation entry must contain the geometry and ordered offsets used by generated address types.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 must define element_type, positive uint32 page_size, num_tiles, block_count, and l1_allocation_bytes values with representable ordered SRAM offsets}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 64 : i64, l1_payload_offset = 32 : i64, num_tiles = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @invalid_allocation_offsets() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Two logical DFBs cannot share any word of their control records.
// expected-error @below {{'builtin.module' op compiler-sram control records overlap}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, storage_capacity_pages = 1 : i64},
  {dfb_index = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 4 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}
]} {
  func.func @overlapping_control_records() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// A payload cannot begin in its own 8-byte control record.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 payload must follow all control records at a 64-byte-aligned offset}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 64 : i64, l1_payload_offset = 68 : i64, num_tiles = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}
]} {
  func.func @payload_overlaps_own_control() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// A payload cannot begin in another DFB's control record.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 payload must follow all control records at a 64-byte-aligned offset}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, storage_capacity_pages = 1 : i64},
  {dfb_index = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 64 : i64, l1_payload_offset = 128 : i64, num_tiles = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}
]} {
  func.func @payload_overlaps_other_control() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Blackhole payloads require 64-byte alignment for DRAM-to-SRAM reads.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 payload must follow all control records at a 64-byte-aligned offset}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.target_arch = #ttcore.arch<blackhole>, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 32 : i64, num_tiles = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}
]} {
  func.func @blackhole_payload_alignment() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// An external descriptor must agree with the finalized allocation geometry.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, block_count = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @descriptor_geometry_mismatch() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.opaque_call' op compiler-sram descriptor geometry differs from its allocation metadata}}
    ttkernel.opaque_call "describe" template_args [#ttkernel.dfb_descriptor<0, 2, 1, 2048>] () {dfb_resource_indices = array<i32: 0>, header = "describe.hpp"} : () -> ()
    return
  }
}

// -----

// Compute descriptors require a supported tile type in the allocation table.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, block_count = 1 : i64, l1_allocation_bytes = 4096 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, element_type = !ttcore.tile<32x32, si32>, page_size = 4096 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @descriptor_unsupported_element_type() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    // expected-error @below {{'ttkernel.opaque_call' op compiler-sram compute descriptors require 32x32 BF16 or FP32 tiles}}
    ttkernel.opaque_call "describe" template_args [#ttkernel.dfb_descriptor<0, 1, 1, 4096>] () {dfb_resource_indices = array<i32: 0>, header = "describe.hpp"} : () -> ()
    return
  }
}

// -----

// Descriptor indices must identify an allocation-table entry.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, block_count = 1 : i64, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 2048 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @descriptor_index_out_of_range() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.opaque_call' op compiler-sram descriptor index is absent from allocation metadata}}
    ttkernel.opaque_call "describe" template_args [#ttkernel.dfb_descriptor<1, 1, 1, 2048>] () {dfb_resource_indices = array<i32: 1>, header = "describe.hpp"} : () -> ()
    return
  }
}

// -----

// Packing without an explicit output index would use Metal descriptor state.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @implicit_pack_index() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    // expected-error @below {{compiler-sram packing requires an explicit tile index}}
    ttkernel.pack_tile(%zero, %storage, %zero, false) : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
    return
  }
}

// -----

// Synchronization must cover one complete block.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 2 : i64, block_count = 1 : i64, l1_allocation_bytes = 4096 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 2 : i64}]} {
  func.func @partial_block() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    %one = arith.constant 1 : i32
    // expected-error @below {{compiler-sram requires full-block synchronization to preserve contiguous acquisitions}}
    ttkernel.cb_wait_front(%storage, %one) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }
}

// -----

// Runtime page counts cannot establish the fixed-size transaction contract.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @dynamic_page_count(%pages : i32) attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    // expected-error @below {{compiler-sram requires a static storage identity and page count}}
    ttkernel.cb_wait_front(%storage, %pages) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }
}

// -----

// Pre-lowered C++ can contain storage effects that the validator cannot classify.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = []} {
  func.func @prelowered_effect() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %unused = ttkernel.get_compile_time_arg_val(0) : () -> i32
    // expected-error @below {{compiler-sram cannot validate pre-lowered C++ effects}}
    emitc.verbatim "side_effect();"
    return
  }
}

// -----

// The backend requires the finalized allocation table before conversion.
// expected-error @below {{compiler-sram requires finalized allocation metadata}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64} {
  func.func @missing_allocation_metadata() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %unused = ttkernel.get_compile_time_arg_val(0) : () -> i32
    return
  }
}

// -----

// Allocation metadata must use the finalized array representation.
// expected-error @below {{compiler-sram requires finalized allocation metadata}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = 0 : i64} {
  func.func @malformed_metadata() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    ttkernel.tile_regs_acquire() : () -> ()
    return
  }
}

// -----

// An integer tile is outside the address-based compute contract.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, si32>, page_size = 4096 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 4096 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @integer_tile() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, si32>>
    %zero = arith.constant 0 : index
    // expected-error @below {{compiler-sram compute requires 32x32 BF16 or FP32 tiles}}
    ttkernel.copy_tile_init(%storage) : (!ttkernel.cb<1, !ttcore.tile<32x32, si32>>) -> ()
    return
  }
}

// -----

// Consumer replacement is outside the address-based compute contract.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 65536 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @consumer_replacement() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    // expected-error @below {{has no compiler-sram lowering for ttkernel.pack_waited_tile; Metal DFB fallback is disabled}}
    ttkernel.pack_waited_tile(%zero, %storage, %zero, true) {acquired_tiles = 1 : i64} : (index, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, index) -> ()
    return
  }
}

// -----

// The declared allocation must cover every page in the payload.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 payload extent exceeds its allocation or arena}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 4160 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 2 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 2 : i64}]} {
  func.func @payload_exceeds_allocation() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// The complete allocation must fit in the declared arena.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 payload extent exceeds its allocation or arena}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 4159 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 2 : i64, l1_allocation_bytes = 4096 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 2 : i64}]} {
  func.func @payload_exceeds_arena() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Control records must also fit in the declared arena.
// expected-error @below {{'builtin.module' op compiler-sram control records exceed the arena}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 4 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @control_record_exceeds_arena() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// An address-based kernel requires a declared arena size.
// expected-error @below {{'builtin.module' op compiler-sram requires a representable arena size}}
module attributes {ttl.memory_model = "compiler-sram", ttl.dfb_allocations = []} {
  func.func @missing_arena_size() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %unused = ttkernel.get_compile_time_arg_val(0) : () -> i32
    return
  }
}

// -----

// The arena size must fit the device address type.
// expected-error @below {{'builtin.module' op compiler-sram requires a representable arena size}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 4294967296 : i64, ttl.dfb_allocations = []} {
  func.func @unrepresentable_arena_size() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %unused = ttkernel.get_compile_time_arg_val(0) : () -> i32
    return
  }
}

// -----

// The payload product 2048 * 2^31 * 2^31 wraps to zero without checked arithmetic.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 payload size overflows 64-bit arithmetic}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 4160 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 2147483648 : i64, block_count = 2147483648 : i64, l1_allocation_bytes = 4096 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64}]} {
  func.func @payload_product_overflow() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4611686018427387904, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Metadata validation also applies when no TTKernel operations remain.
// expected-error @below {{'builtin.module' op compiler-sram requires a representable arena size}}
module attributes {ttl.memory_model = "compiler-sram", ttl.dfb_allocations = []} {}

// -----

// A storage allocation must identify its element type.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 must define element_type}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 2112 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @missing_element_type() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// The stored page stride must match the element type.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 page size differs from its element type}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 4160 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 4096 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 4096 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @page_size_type_mismatch() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Equal byte sizes do not make different DFB element types interchangeable.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 2112 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 1 : i64}]} {
  func.func @dfb_element_type_mismatch() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.get_compile_time_arg_val' op compiler-sram DFB element type differs from allocation metadata}}
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, u16>>
    return
  }
}

// -----

// The DFB type must contain the number of pages reserved by its allocation.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 4160 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 2 : i64, l1_allocation_bytes = 4096 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, storage_capacity_pages = 2 : i64}]} {
  func.func @dfb_total_pages_mismatch() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.get_compile_time_arg_val' op compiler-sram DFB geometry differs from allocation metadata}}
    %storage = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    return
  }
}
