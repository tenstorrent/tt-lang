// Reject inconsistent finalized metadata for shared SRAM storage owners.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc --verify-diagnostics --split-input-file

// Allocation-group members must use one control record.
// expected-error @below {{'builtin.module' op compiler-sram storage owner 3 has inconsistent allocation metadata}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 72 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 2 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 8 : i64},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 2 : i64, l1_offset = 8 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 8 : i64}
]} {}

// -----

// Distinct owners cannot share bytes in the control section.
// expected-error @below {{'builtin.module' op compiler-sram control records overlap}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 72 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 4 : i64},
  {dfb_index = 1 : i64, storage_index = 4 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 4 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 4 : i64}
]} {}

// -----

// The shared capacity, not one member's logical capacity, determines extent.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 payload extent exceeds its allocation or arena}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 2 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 4 : i64}
]} {}

// -----

// One node cannot bind a shared owner to two different tensor arguments.
// expected-error @below {{'builtin.module' op compiler-sram storage owner 3 has different tensor backing on a shared launch node without a reconfiguration reset}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4>}]}
]} {}

// -----

// A different offset within one tensor is also a different payload address.
// expected-error @below {{'builtin.module' op compiler-sram storage owner 3 has different tensor backing on a shared launch node without a reconfiguration reset}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 4, byte_size = 4>}]}
]} {}

// -----

// Compare every owner member; the first member has no node in common here.
// expected-error @below {{'builtin.module' op compiler-sram storage owner 3 has different tensor backing on a shared launch node without a reconfiguration reset}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[1, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 2 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 2, byte_offset = 0, byte_size = 4>}]}
]} {}

// -----

// A reset of a different DFB does not establish this owner's handoff.
// expected-error @below {{'builtin.module' op compiler-sram storage owner 3 has different tensor backing on a shared launch node without a reconfiguration reset}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 16 : i64, ttl.compiler_sram_reconfiguration_resets = [{dfb_indices = array<i32: 2>, ordinal = 0 : i64}], ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 2 : i64, storage_index = 4 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 8 : i64, storage_segments = [{nodes = [[1, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 2, byte_offset = 0, byte_size = 4>}]}
]} {}

// -----

// Reset metadata must identify an allocation entry in this module.
// expected-error @below {{'builtin.module' op contains invalid compiler-sram reconfiguration reset index}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.compiler_sram_reconfiguration_resets = [{dfb_indices = array<i32: 1>, ordinal = 0 : i64}], ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]}
]} {}

// -----

// Reset metadata needs a representable boundary ordinal.
// expected-error @below {{'builtin.module' op contains malformed compiler-sram reconfiguration reset metadata}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.compiler_sram_reconfiguration_resets = [{dfb_indices = array<i32: 0>, ordinal = -1 : i64}], ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]}
]} {}
