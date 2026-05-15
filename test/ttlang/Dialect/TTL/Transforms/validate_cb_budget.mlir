// Tests for ttl-validate-cb-budget: overflow, multi-function/index behavior, and all four
// layout/dtype combinations for CB element types:
//   - ttcore.tile<32x32, bf16>  -> 2048 bytes per slot (explicit tile)
//   - ttcore.tile<32x32, f32>   -> 4096 bytes per slot (explicit tile)
//   - bf16 (row-wise, builtin)  -> TileType::get(bf16)  -> 2048 bytes per slot
//   - f32  (row-wise, builtin)  -> TileType::get(f32)   -> 4096 bytes per slot
// WH/BH fallback budget B = 1432 * 1024 = 1466368 bytes when the module has no system_desc.
// Final tests use a ttcore.system_desc with a smaller L1 to verify the pass reads the descriptor.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget)'

// -----

// Shared multi-function / multi-index scenarios (tile bf16; logic is dtype-agnostic).

// Single tile bf16 CB exceeds B (717 * 2048 = 1468416 > B).

func.func @overflow_single_cb_tile_bf16() {
  // expected-error @below {{exceeds L1 budget}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[717, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

func.func @compiler_allocated_overflow_tile_bf16() {
  // expected-error @below {{exceeds L1 budget}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[717, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

func.func @under_budget_tile_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

func.func @no_cbs() {
  func.return
}

// -----

// 645 * 2048 = 1320960 <= B (high usage but under budget).

func.func @high_usage_under_budget_tile_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 645], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

func.func @two_indices_under_budget_tile_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[20, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[30, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

func.func @same_index_compute_kernel_tile_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[400, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

func.func @same_index_dm_kernel_tile_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[400, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

func.func @same_index_smaller_binding_tile_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[10, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

func.func @same_index_larger_binding_tile_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[100, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

func.func @two_indices_combined_overflow_tile_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[50, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{exceeds L1 budget}}
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[668, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

func.func @two_funcs_cb_index0_tile_bf16() {
  // expected-error @below {{exceeds L1 budget}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[400, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

func.func @two_funcs_cb_index1_tile_bf16() {
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[324, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Explicit ttcore.tile<32x32, f32> (4096 bytes / slot).

func.func @under_budget_tile_f32() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// 359 * 4096 = 1470464 > B.

func.func @overflow_tile_f32() {
  // expected-error @below {{exceeds L1 budget}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[359, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// 323 * 4096 = 1323008 <= B (high usage but under budget).

func.func @high_usage_under_budget_tile_f32() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 323], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Row-wise builtin bf16 (2048 bytes per slot; same footprint as tile bf16).

func.func @under_budget_row_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], bf16, 2>
  func.return
}

// -----

func.func @overflow_row_bf16() {
  // expected-error @below {{exceeds L1 budget}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[717, 1], bf16, 1>
  func.return
}

// -----

func.func @high_usage_under_budget_row_bf16() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 645], bf16, 1>
  func.return
}

// -----

// Row-wise builtin f32 (4096 bytes per slot; same footprint as tile f32).

func.func @under_budget_row_f32() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
  func.return
}

// -----

func.func @overflow_row_f32() {
  // expected-error @below {{exceeds L1 budget}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[359, 1], f32, 1>
  func.return
}

// -----

func.func @high_usage_under_budget_row_f32() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 323], f32, 1>
  func.return
}

// -----

// Mixed layout and dtype pairs (two cb_index, under B).

func.func @mixed_tile_bf16_row_f32_under_budget() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[10, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
  func.return
}

// -----

func.func @mixed_row_bf16_tile_f32_under_budget() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], bf16, 2>
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

func.func @mixed_tile_bf16_tile_f32_under_budget() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[5, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[3, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

func.func @mixed_row_bf16_row_f32_under_budget() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], bf16, 2>
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
  func.return
}

// -----

// Budget derived from ttcore.system_desc instead of the WH/BH fallback.
// Chip has l1_size = 204800, l1_unreserved_base = 0 -> usableL1Size = 204800.
// 100 bf16 tiles * 2048 = 204800 = exactly at budget (should pass).

module attributes {ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 204800, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 0, erisc_l1_unreserved_base = 0, dram_unreserved_base = 0, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>], supported_tile_sizes = [32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(0, 0)], dram_bank_to_logical_worker_noc1 = [(0, 0)]}], [0], [1 : i32], [ 0x0x0x0]>} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @system_desc_at_budget_tile_bf16() {
    %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[100, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}

// -----

// Same system_desc (usableL1Size = 204800) but 101 tiles overflows.
// 101 * 2048 = 206848 > 204800. Would pass under the fallback (1466368),
// proving the pass reads the descriptor.

module attributes {ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 204800, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 0, erisc_l1_unreserved_base = 0, dram_unreserved_base = 0, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>], supported_tile_sizes = [32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(0, 0)], dram_bank_to_logical_worker_noc1 = [(0, 0)]}], [0], [1 : i32], [ 0x0x0x0]>} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @system_desc_overflow_tile_bf16() {
    // expected-error @below {{exceeds L1 budget}}
    %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[101, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}
