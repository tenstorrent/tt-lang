// Tests for ttl-validate-cb-budget: overflow errors, acceptance of small or empty CB usage, default 90% usage warning,
// sums over distinct cb_index values, max-per-cb_index across functions (no double-count), and combined overflow
// from multiple functions each binding a different cb_index.
// WH/BH fallback budget is 1368064 bytes when the module has no system_desc (bf16 tile = 2048 bytes).
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget)'

// -----

// Test 1: single CB exceeds fallback budget (670 * 2048 = 1372160 > 1368064).

func.func @overflow_single_cb() {
  // expected-error @below {{exceeds L1 budget}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[670, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test 2: compiler-allocated bind_cb counts toward total.

func.func @compiler_allocated_overflow() {
  // expected-error @below {{exceeds L1 budget}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[670, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test 3: under budget (no diagnostic).

func.func @under_budget() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test 4: no circular buffers (pass is a no-op).

func.func @no_cbs() {
  func.return
}

// -----

// Test 5: above default 90% warn threshold but still under budget.
// 90% of 1368064 = 1231257; 602 * 2048 = 1232896.

func.func @warn_high_usage_default_threshold() {
  // expected-warning @below {{is above 90 percent}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 602], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test 6: multiple cb_index values in one function; sizes add across distinct indices.

func.func @two_indices_under_budget() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[20, 1], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[30, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test 7: same cb_index=0 in two functions with identical shapes; count once (400 * 2048), not twice.
// If both were summed, 800 * 2048 would exceed the fallback budget.

func.func @same_index_compute_kernel() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[400, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

func.func @same_index_dm_kernel() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[400, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test 8: same cb_index=0 across functions with different shapes; only the larger allocation counts.

func.func @same_index_smaller_binding() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[10, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

func.func @same_index_larger_binding() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[100, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test 9: two indices; combined max-per-index total exceeds budget (100 + 600 tiles) * 2048 > 1368064.

func.func @two_indices_combined_overflow() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[100, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{exceeds L1 budget}}
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[600, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Test 10: two functions, two different cb_index values; per-index maxima sum over budget.
// 400 * 2048 + 300 * 2048 = 1433600 > 1368064. Diagnostic on the larger slot (cb_index 0).

func.func @two_funcs_cb_index0() {
  // expected-error @below {{exceeds L1 budget}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[400, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

func.func @two_funcs_cb_index1() {
  %cb1 = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[300, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}
