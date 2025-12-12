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

// -----

// Test step = 0 (invalid)
// expected-error @below {{slice step cannot be zero}}
func.func @test_slice_zero_step() attributes {s = #ttl.slice<start = 0, stop = 8, step = 0>} {
  return
}

// -----

// Test stop < start with positive step (invalid)
// expected-error @below {{slice stop (2) must be >= start (8) when step is positive}}
func.func @test_slice_stop_less_than_start() attributes {s = #ttl.slice<start = 8, stop = 2, step = 1>} {
  return
}

// -----

// Test stop < start with step = 0 (invalid).
// expected-error @below {{slice step cannot be zero}}
func.func @test_slice_stop_less_than_start_zero_step() attributes {s = #ttl.slice<start = 8, stop = 2, step = 0>} {
  return
}

// -----

// Test stop > start with negative step (invalid)
// expected-error @below {{slice stop (8) must be <= start (2) when step is negative}}
func.func @test_slice_stop_greater_than_start_negative_step() attributes {s = #ttl.slice<start = 2, stop = 8, step = -1>} {
  return
}
