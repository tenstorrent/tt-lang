// Tests for ttl-validate-cb-budget: overflow, warnings, multi-function/index behavior, and all four
// layout/dtype combinations for CB element types:
//   - ttcore.tile<32x32, bf16>  -> 2048 bytes per slot (explicit tile)
//   - ttcore.tile<32x32, f32>   -> 4096 bytes per slot (explicit tile)
//   - bf16 (row-wise, builtin)  -> TileType::get(bf16)  -> 2048 bytes per slot
//   - f32  (row-wise, builtin)  -> TileType::get(f32)   -> 4096 bytes per slot
// WH/BH fallback budget B = 1432 * 1024 = 1466368 bytes when the module has no system_desc.
// 90% warn threshold T = (B * 90) / 100 = 1319731.
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

// T < 645 * 2048 = 1320960 <= B.

func.func @warn_high_usage_tile_bf16() {
  // expected-warning @below {{is above 90 percent}}
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

// T < 323 * 4096 = 1323008 <= B.

func.func @warn_high_usage_tile_f32() {
  // expected-warning @below {{is above 90 percent}}
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

func.func @warn_high_usage_row_bf16() {
  // expected-warning @below {{is above 90 percent}}
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

func.func @warn_high_usage_row_f32() {
  // expected-warning @below {{is above 90 percent}}
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
