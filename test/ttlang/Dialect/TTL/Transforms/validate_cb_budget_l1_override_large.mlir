// Tests for ttl-validate-cb-budget with a large l1-budget-override.
//
// 717 * 2048 = 1468416 bytes exceeds the default WH/BH fallback budget
// (1466368 bytes) but fits within the override (2000000 bytes).
//
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=2000000})'

// -----

// 900 * 2048 = 1843200 bytes, under the 2000000 override. No error.

func.func @normally_overflow_passes_with_large_override() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[900, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// 10 * 2048 = 20480 bytes, well under 2000000. No error.

func.func @under_large_override() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[10, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}
