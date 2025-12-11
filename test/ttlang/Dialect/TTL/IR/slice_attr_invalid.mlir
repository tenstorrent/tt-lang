// RUN: ttlang-opt --split-input-file --verify-diagnostics %s

// Test invalid SliceAttr syntax

// expected-error @below {{expected integer value}}
// expected-error @below {{failed to parse TTL_SliceAttr parameter 'start'}}
func.func @test_slice_invalid_start() attributes {s = #ttl.slice<start = "foo", stop = 8, step = 1>} {
  return
}

// -----

// expected-error @below {{expected integer value}}
// expected-error @below {{failed to parse TTL_SliceAttr parameter 'stop'}}
func.func @test_slice_invalid_stop() attributes {s = #ttl.slice<start = 0, stop = "bar", step = 1>} {
  return
}

// -----

// expected-error @below {{expected integer value}}
// expected-error @below {{failed to parse TTL_SliceAttr parameter 'step'}}
func.func @test_slice_invalid_step() attributes {s = #ttl.slice<start = 0, stop = 8, step = "baz">} {
  return
}

// -----

// expected-error @below {{expected ','}}
func.func @test_slice_missing_field() attributes {s = #ttl.slice<start = 0, stop = 8>} {
  return
}

// -----

// Missing step (same as missing_field, kept for clarity)
// expected-error @below {{expected ','}}
func.func @test_slice_missing_step() attributes {s = #ttl.slice<start = 0, stop = 8>} {
  return
}
