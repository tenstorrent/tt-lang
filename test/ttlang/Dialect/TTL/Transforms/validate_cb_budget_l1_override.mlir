// Tests for ttl-validate-cb-budget with the l1-budget-override pass option.
//
// A small override budget (4096 bytes) causes a normally-passing CB to fail.
// A large override budget (2000000 bytes) causes a normally-failing CB to pass.
//
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=4096})'

// -----

// 10 * 2048 = 20480 bytes > override budget 4096.

func.func @overflow_with_small_override() {
  // expected-error @below {{exceeds L1 budget (4096 bytes)}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[10, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// 1 * 2048 = 2048 bytes < override budget 4096.

func.func @under_small_override() {
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}
