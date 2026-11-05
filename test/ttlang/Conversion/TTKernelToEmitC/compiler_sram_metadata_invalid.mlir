// Finalized SRAM allocation entries must be ordered and numerically representable.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc --verify-diagnostics --split-input-file

// An entry without its finalized identity cannot be bound by array position.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 requires a matching dfb_index}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.dfb_allocations = [{block_count = 1 : i64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 4 : i64}]} {}

// -----

// The serialized identity must equal its array position.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 requires a matching dfb_index}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 132 : i64, ttl.dfb_allocations = [
  {block_count = 1 : i64, dfb_index = 1 : i64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 4 : i64},
  {block_count = 1 : i64, dfb_index = 0 : i64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 8 : i64, l1_payload_offset = 128 : i64, num_tiles = 1 : i64, page_size = 4 : i64}
]} {}

// -----

// An allocation identity must use a signless integer.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 requires a matching dfb_index}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.dfb_allocations = [{block_count = 1 : i64, dfb_index = 0 : si64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 4 : i64}]} {}

// -----

// An arena size outside the supported integer range is invalid.
// expected-error @below {{'builtin.module' op compiler-sram requires a representable arena size}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 18446744073709551616 : i128, ttl.dfb_allocations = []} {}

// -----

// A page size outside the supported integer range is invalid.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 must define element_type, positive uint32 page_size}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.dfb_allocations = [{block_count = 1 : i64, dfb_index = 0 : i64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 18446744073709551616 : i128}]} {}
