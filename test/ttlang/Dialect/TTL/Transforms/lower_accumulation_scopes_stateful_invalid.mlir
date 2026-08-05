// Verifies additive-form tensor scopes do not fall back to stateful lowering.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-accumulation-scopes{kind=tensor strategy=auto}))' --verify-diagnostics --split-input-file

// Purpose: A scope that already contains a final output store is the additive
// tensor form. If its recurrence is not additive, lowering must reject it
// instead of inserting a second final store through the stateful fallback.
func.func @stored_non_additive_scope_is_not_stateful_fallback() {
  %cb_init = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init_wait = ttl.cb_wait %cb_init : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %cb_init : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  // expected-error @below {{tensor accumulation lowering requires a loop-carried additive recurrence of the form acc = acc + contribution}}
  ttl.accumulation_scope outs(%reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%state: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    %loop = scf.for %iter = %c0 to %c4 step %c1 iter_args(%acc = %state) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
      %next = ttl.relu %acc : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ttl.store %loop, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield %loop : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init])
  func.return
}
