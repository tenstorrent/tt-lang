// Tests that the DFB budget usage percentage does not overflow uint64_t.
//
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget{l1-budget-override=1})'

// The allocation fits in uint64_t, but multiplying it by 100 requires 71 bits.

func.func @large_valid_allocation() {
  // expected-error @below {{1844674407370954956800 percent}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[9007199254740991, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// The next BF16 tile would require exactly 2^64 bytes. Allocation validation
// must reject the unrepresentable size instead of accepting a wrapped zero.

func.func @allocation_size_overflow() {
  // expected-error @below {{DFB allocation size is not representable}}
  %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[9007199254740992, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Individually representable physical allocations must also reject an
// unrepresentable aggregate instead of wrapping their sum.

// expected-error @below {{total DFB allocation size is not representable as uint64_t}}
module {
  func.func @total_allocation_size_overflow() {
    %cb0 = ttl.bind_cb{cb_index = 0, block_count = 1} : !ttl.cb<[4503599627370496, 1], !ttcore.tile<32x32, bf16>, 1>
    %cb1 = ttl.bind_cb{cb_index = 1, block_count = 1} : !ttl.cb<[4503599627370496, 1], !ttcore.tile<32x32, bf16>, 1>
    func.return
  }
}
