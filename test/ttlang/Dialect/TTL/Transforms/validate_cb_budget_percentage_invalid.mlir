// Tests that the DFB budget usage percentage does not overflow uint64_t.
//
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=1})'

// The allocation fits in uint64_t, but multiplying it by 100 requires 71 bits.

func.func @large_valid_allocation() {
  // expected-error @below {{1844674407370954956800 percent}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[9007199254740991, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}
