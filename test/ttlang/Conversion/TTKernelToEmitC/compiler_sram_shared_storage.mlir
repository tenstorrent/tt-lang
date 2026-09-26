// Finalized metadata accepts two logical DFBs sharing one storage owner.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc -o /dev/null

// The shared extent covers the larger logical DFB; both entries use one record.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 72 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 2 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 8 : i64},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 2 : i64, block_count = 1 : i64, storage_capacity_pages = 2 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 8 : i64}
]} {}
