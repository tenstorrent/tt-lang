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

// -----

func.func @contribution_wait_must_use_default_block_size() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // expected-error @below {{'ttl.accumulation_scope' op cannot lower tensor accumulation scope to DST: expected a DST-compatible same-type additive recurrence with an attached init tensor, a streamed or resident contribution ttl.cb_wait using the default block size, balanced contribution releases, and a static output tile count that fits in DST}}
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc_init: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    %result = scf.for %iv = %c0 to %c3 step %c1
        iter_args(%acc = %acc_init)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %contrib_wait = ttl.cb_wait %contrib_cb {num_tiles = 1 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %next = ttl.add %acc, %contrib : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %contrib_cb {num_tiles = 1 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
      scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ttl.store %result, %out : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init])
  return
}
