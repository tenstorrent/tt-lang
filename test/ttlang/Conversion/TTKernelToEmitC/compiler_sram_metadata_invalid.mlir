// Finalized SRAM allocation entries require valid identities, sizes, and storage segments.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc --verify-diagnostics --split-input-file

// An entry without its finalized identity cannot be bound by array position.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 requires a matching dfb_index}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.dfb_allocations = [{block_count = 1 : i64, storage_capacity_pages = 1 : i64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 4 : i64}]} {}

// -----

// The serialized identity must equal its array position.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 requires a matching dfb_index}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 132 : i64, ttl.dfb_allocations = [
  {block_count = 1 : i64, storage_capacity_pages = 1 : i64, dfb_index = 1 : i64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 4 : i64},
  {block_count = 1 : i64, storage_capacity_pages = 1 : i64, dfb_index = 0 : i64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 8 : i64, l1_payload_offset = 128 : i64, num_tiles = 1 : i64, page_size = 4 : i64}
]} {}

// -----

// An allocation identity must use a signless integer.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 requires a matching dfb_index}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.dfb_allocations = [{block_count = 1 : i64, storage_capacity_pages = 1 : i64, dfb_index = 0 : si64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 4 : i64}]} {}

// -----

// An arena size outside the supported integer range is invalid.
// expected-error @below {{'builtin.module' op compiler-sram requires a representable arena size}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 18446744073709551616 : i128, ttl.dfb_allocations = []} {}

// -----

// A page size outside the supported integer range is invalid.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 must define element_type, positive uint32 page_size}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.dfb_allocations = [{block_count = 1 : i64, storage_capacity_pages = 1 : i64, dfb_index = 0 : i64, element_type = f32, l1_allocation_bytes = 4 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i64, page_size = 18446744073709551616 : i128}]} {}

// -----

// A tensor-backed segment must identify its launch nodes.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 must define element_type, positive uint32 page_size, num_tiles, block_count, storage_capacity_pages, and either an arena payload or tensor backing with valid launch nodes and representable SRAM offsets}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]}]} {}

// -----

// An empty node list cannot locate the tensor-backed segment.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 must define element_type, positive uint32 page_size, num_tiles, block_count, storage_capacity_pages, and either an arena payload or tensor backing with valid launch nodes and representable SRAM offsets}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]}]} {}

// -----

// A segment node must have two nonnegative integer coordinates.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 must define element_type, positive uint32 page_size, num_tiles, block_count, storage_capacity_pages, and either an arena payload or tensor backing with valid launch nodes and representable SRAM offsets}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, -1]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]}]} {}

// -----

// A node with one coordinate has no device position.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 must define element_type, positive uint32 page_size, num_tiles, block_count, storage_capacity_pages, and either an arena payload or tensor backing with valid launch nodes and representable SRAM offsets}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]}]} {}

// -----

// Duplicate nodes cannot define a unique backing for each launch node.
// expected-error @below {{'builtin.module' op compiler-sram allocation entry 0 must define element_type, positive uint32 page_size, num_tiles, block_count, storage_capacity_pages, and either an arena payload or tensor backing with valid launch nodes and representable SRAM offsets}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0], [0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]}]} {}

// -----

// A mistyped storage-segment field cannot convert an allocation to arena backing.
// expected-error @below {{compiler-sram allocation entry 0 must define element_type}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 4 : i64, storage_segments = 7 : i32}]} {}

// -----

// A mistyped payload offset cannot convert an allocation to tensor backing.
// expected-error @below {{compiler-sram allocation entry 0 must define element_type}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, l1_payload_offset = "invalid", storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]}]} {}

// -----

// A mistyped payload size cannot convert an allocation to tensor backing.
// expected-error @below {{compiler-sram allocation entry 0 must define element_type}}
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [{dfb_index = 0 : i64, element_type = !ttcore.tile<32x32, bf16>, page_size = 2048 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, l1_allocation_bytes = "invalid", storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}]}]} {}
