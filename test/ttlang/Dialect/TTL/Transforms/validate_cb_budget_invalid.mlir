// Verify that ttl-validate-cb-budget rejects DFB sizes that cannot be represented.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-validate-cb-budget)'

// A sub-byte scalar cannot define the byte-addressed hardware page size.
func.func @sub_byte_page_size() {
  // expected-error @below {{'ttl.bind_cb' op element type must occupy a positive whole number of bytes, got 'i1'}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 64} : !ttl.cb<[65536, 64], i1, 64>
  return
}

// -----

// The allocation size calculation rejects products outside uint64_t.
func.func @allocation_size_overflow() {
  // expected-error @below {{'ttl.bind_cb' op allocation size is not representable}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4294967296, 4294967296], i8, 2>
  return
}
