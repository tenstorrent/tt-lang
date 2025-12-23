// Summary: verify linearized_index verifier catches invalid cases
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// -----
// Test: map with zero dimensions
func.func @zero_dims() {
  // expected-error @below {{index_map must have at least one dimension}}
  %idx = ttl.linearized_index affine_map<() -> (0)> : index
  return
}

// -----
// Test: map with multiple results
func.func @multi_result() {
  // expected-error @below {{index_map must have exactly one result, got 2}}
  %idx = ttl.linearized_index affine_map<(d0, d1) -> (d0, d1)> : index
  return
}

// -----
// Test: map with zero results
func.func @zero_results() {
  // expected-error @below {{index_map must have exactly one result, got 0}}
  %idx = ttl.linearized_index affine_map<(d0) -> ()> : index
  return
}
