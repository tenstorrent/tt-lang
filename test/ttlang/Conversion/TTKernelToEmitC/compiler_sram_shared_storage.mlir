// Finalized metadata accepts two logical DFBs sharing one storage owner.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc --split-input-file -o /dev/null

// The shared extent covers the larger logical DFB; both entries use one record.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 72 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 2 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 8 : i64},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 2 : i64, block_count = 1 : i64, storage_capacity_pages = 2 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, l1_allocation_bytes = 8 : i64}
]} {}

// -----

// Distinct tensor addresses may share a node-local record on disjoint nodes.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[1, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4>}]}
]} {}

// -----

// A synchronized reconfiguration can reset an owner's state before its backing changes.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 8 : i64, ttl.compiler_sram_reconfiguration_resets = [{dfb_indices = array<i32: 0>, ordinal = 0 : i64, backing_handoffs = [{from_dfb_index = 0 : i32, to_dfb_index = 1 : i32, node = [0, 0]}]}], ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4>}]}
]} {}

// -----

// An arena payload on another node does not share the tensor's control state.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, allocation_nodes = [[1, 0]], l1_payload_offset = 64 : i64, l1_allocation_bytes = 4 : i64}
]} {}

// -----

// Reconfiguration permits the same node to switch from tensor to arena storage.
module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 68 : i64, ttl.compiler_sram_reconfiguration_resets = [{dfb_indices = array<i32: 0>, ordinal = 0 : i64, backing_handoffs = [{from_dfb_index = 0 : i32, to_dfb_index = 1 : i32, node = [0, 0]}]}], ttl.dfb_allocations = [
  {dfb_index = 0 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4>}]},
  {dfb_index = 1 : i64, storage_index = 3 : i64, element_type = f32, page_size = 4 : i64, num_tiles = 1 : i64, block_count = 1 : i64, storage_capacity_pages = 1 : i64, l1_offset = 0 : i64, allocation_nodes = [[0, 0]], l1_payload_offset = 64 : i64, l1_allocation_bytes = 4 : i64}
]} {}
