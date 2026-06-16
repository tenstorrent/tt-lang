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

// One initial mode is required for each output tensor.
func.func @initial_mode_count_mismatch() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op requires one initial mode per output}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                    tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } {combiners = [0 : i32, 0 : i32], initial_modes = [0 : i32]}
  return
}

// -----

// Combiner attributes must be generated enum attributes, not arbitrary attrs.
func.func @malformed_combiner_attr() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op combiners must contain accumulation combiner enum attributes}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } {combiners = [99 : i32], initial_modes = [0 : i32]}
  return
}

// -----

// Initial-mode attributes must be generated enum attributes, not arbitrary attrs.
func.func @malformed_initial_mode_attr() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op initial_modes must contain accumulation initial-mode enum attributes}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } {combiners = [0 : i32], initial_modes = [99 : i32]}
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

// Stateful bodies require one block argument per output.
func.func @stateful_block_arg_count_mismatch() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op stateful body requires one block argument per output}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0, %init1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc0, %acc0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } {combiners = [0 : i32, 0 : i32], initial_modes = [2 : i32, 2 : i32]}
  return
}

// -----

// Stateful bodies require one yielded value per output.
func.func @stateful_yield_count_mismatch() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op stateful body must yield one value per output}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0, %init1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
       %acc1: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc0 : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } {combiners = [0 : i32, 0 : i32], initial_modes = [2 : i32, 2 : i32]}
  return
}

// -----

// Stateful bodies require explicit initial values for all accumulators.
func.func @stateful_non_explicit_initial_mode() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op stateful body requires explicit initial mode for every output}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0 : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
       %acc1: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc0, %acc1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } {combiners = [0 : i32, 0 : i32], initial_modes = [2 : i32, 0 : i32]}
  return
}

// -----

// Stateful yielded values must match their corresponding output types.
func.func @stateful_yield_type_mismatch() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %bad = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op stateful yielded value 0 type}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0, %init1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
       %acc1: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %bad, %acc1 : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } {combiners = [0 : i32, 0 : i32], initial_modes = [2 : i32, 2 : i32]}
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
