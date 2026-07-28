// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Summary: Verifier-level rejection cases for malformed `ttl.accumulation_scope`
// initial-state policy, yielded values, and unsupported nesting.

// One initial mode is required for each output tensor.
func.func @initial_mode_count_mismatch() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op requires one initial mode per output}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
       %acc1: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc0, %acc1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([overwrite])
  return
}

// -----

// Initial-mode policy entries must be known symbolic names.
func.func @malformed_initial_mode_attr() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{expected accumulation initial mode `overwrite`, `accumulate_existing`, or `init`}}
  } initial_modes([not_a_mode])
  return
}

// -----

// Init mode requires a corresponding init operand.
func.func @missing_init() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op requires one init operand per init mode}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init])
  return
}

// -----

// Init operands must have the same tensor type as their outputs.
func.func @init_type_mismatch() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op init operand 0 type}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<2x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init])
  return
}

// -----

// Bodies require one block argument per output.
func.func @body_arg_count_mismatch() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op body requires one block argument per output}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0, %init1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc0, %acc0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init, init])
  return
}

// -----

// Bodies must yield one value per output.
func.func @yield_count_mismatch() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op body must yield one value per output}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0, %init1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
       %acc1: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc0 : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init, init])
  return
}

// -----

// Body block argument types must match their corresponding output types.
func.func @body_arg_type_mismatch() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op body argument 0 type}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc: tensor<2x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc : tensor<2x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([overwrite])
  return
}

// -----

// Yielded values must match their corresponding output types.
func.func @yield_type_mismatch() {
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %bad = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op yielded value 0 type}}
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0, %init1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
       %acc1: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %bad, %acc1 : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init, init])
  return
}

// -----

// Nested accumulation scopes are rejected until nested policy composition is
// specified.
func.func @nested_accumulation_scope() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op nested ttl.accumulation_scope is not supported (#648); split nested accumulations into separate scopes}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%outer_acc: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%inner_acc: tensor<1x1x!ttcore.tile<32x32, bf16>>):
      ttl.yield %inner_acc : tensor<1x1x!ttcore.tile<32x32, bf16>>
    } initial_modes([overwrite])
    ttl.yield %outer_acc : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([overwrite])
  return
}
