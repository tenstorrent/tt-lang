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
