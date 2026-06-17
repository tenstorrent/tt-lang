// Verifies ttl-insert-cb-sync rejects unsupported same-DFB state updates before
// inserting releases.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' --split-input-file --verify-diagnostics

// A value derived from a user-declared DFB wait is stored back into the same DFB
// while the SSA value remains live after the store. This requires explicit
// materialization before the state update.

func.func @same_dfb_derived_value_live_after_state_store(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %state = ttl.bind_cb{cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %sink = ttl.bind_cb{cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %state_wait = ttl.cb_wait %state : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %state_block = ttl.attach_cb %state_wait, %state : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %next_state = ttl.add %state_block, %input : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %dependent = ttl.mul %state_block, %input : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %state_reserve = ttl.cb_reserve %state : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{unsupported same-DFB tensor SSA state update}}
  ttl.store %next_state, %state_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  %sink_reserve = ttl.cb_reserve %sink : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %dependent, %sink_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}
