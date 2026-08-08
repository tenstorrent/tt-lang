// Verifies that DFB budget validation diagnoses unsupported element types.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget)'

func.func @unsupported_element_type() {
  // expected-error @below {{'ttl.bind_cb' op cannot determine DFB page size for element type i3}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], i3, 1>
  func.return
}

// -----

// Verifies that allocation sizes outside uint64_t are rejected.
func.func @allocation_size_overflow() {
  // expected-error @below {{'ttl.bind_cb' op DFB allocation size is not representable}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4294967296, 2147483648], i8, 2>
  func.return
}
