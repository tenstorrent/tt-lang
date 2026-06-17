// Verifies required DST strategy reports unsupported grouped stateful tensor
// lowering.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=dst}))' --verify-diagnostics

func.func @stateful_required_dst() {
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower stateful tensor accumulation scope to DST: grouped DST lowering is not implemented}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%state: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %state : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } combiners([yielded]) initial_modes([explicit])
  func.return
}
