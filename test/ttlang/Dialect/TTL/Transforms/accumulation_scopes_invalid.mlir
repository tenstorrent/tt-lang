// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes))' --verify-diagnostics --split-input-file

// Summary: Verifies tensor accumulation lowering rejects invalid scopes before
// mutation.

func.func @scope_output_must_be_reserve() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.accumulation_scope' op tensor accumulation lowering requires output from ttl.cb_reserve}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init])
  return
}
