// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Summary: Verifier-level rejection cases for malformed `ttl.accumulation_scope`
// accumulation policy and unsupported nesting.

// One combiner is required for each output tensor.
func.func @combiner_count_mismatch() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op requires one combiner per output}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                    tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } {combiners = [0 : i32], initial_modes = [0 : i32, 0 : i32]}
  return
}

// -----

// Explicit initial-value mode requires a corresponding init operand.
func.func @missing_explicit_init() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op requires one explicit init per explicit initial mode}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } {combiners = [0 : i32], initial_modes = [2 : i32]}
  return
}

// -----

// Explicit init operands must have the same tensor type as their outputs.
func.func @explicit_init_type_mismatch() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op explicit init 0 type}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<2x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } {combiners = [0 : i32], initial_modes = [2 : i32]}
  return
}

// -----

// Nested accumulation scopes are rejected until nested policy composition is
// specified.
func.func @nested_accumulation_scope() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op nested ttl.accumulation_scope is not supported (#648); split nested accumulations into separate scopes}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      ttl.yield
    } {combiners = [0 : i32], initial_modes = [0 : i32]}
    ttl.yield
  } {combiners = [0 : i32], initial_modes = [0 : i32]}
  return
}
