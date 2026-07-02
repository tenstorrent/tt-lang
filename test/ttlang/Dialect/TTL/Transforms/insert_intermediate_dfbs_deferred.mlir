// Tests deferred compiler DFB materialization after initial compute formation.
// The pass must materialize ttl.compute results as extra compute outputs and
// leave consumers ready for the final convert-ttl-to-compute pass.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-form-producer-compute,ttl-insert-intermediate-dfbs))' | FileCheck %s --check-prefix=DEFERRED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-form-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s --check-prefix=FULL

// -----

// A producer compute result is both stored to a user DFB and consumed by a
// reduce. Intermediate DFB materialization must add a second compute output for
// the compiler DFB instead of emitting a tensor-level store after the compute.

// DEFERRED-LABEL: func.func @stored_add_then_reduce
// DEFERRED: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// DEFERRED: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// DEFERRED: %{{.*}}:2 = ttl.compute
// DEFERRED: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// DEFERRED: ttl.cb_push %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// DEFERRED: ttl.reduce %[[INTERMEDIATE_ATTACHED]]

// FULL-LABEL: func.func @stored_add_then_reduce
// FULL: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// FULL: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// FULL: %{{.*}}:2 = ttl.compute
// FULL: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// FULL: ttl.cb_push %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// FULL: ttl.compute ins(%[[INTERMEDIATE_ATTACHED]],
// FULL: ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @stored_add_then_reduce()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_scaler = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_sum = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_wait = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %b_wait, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %cb_scaler : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %sum = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum_reserve = ttl.cb_reserve %cb_sum : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %sum_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  %reduced = ttl.reduce %sum, %scaler 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_reserve = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %out_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A stored state update also feeds a broadcast. This is the #666 ordering
// pattern: the DFB push for the compiler intermediate must remain after the
// producing compute and before the broadcast consumer wait.

// DEFERRED-LABEL: func.func @state_update_feeds_broadcast
// DEFERRED: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// DEFERRED: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// DEFERRED: %{{.*}}:2 = ttl.compute
// DEFERRED: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// DEFERRED: ttl.cb_push %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// DEFERRED: ttl.block.broadcast %[[INTERMEDIATE_ATTACHED]]

// FULL-LABEL: func.func @state_update_feeds_broadcast
// FULL: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// FULL: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// FULL: %{{.*}}:2 = ttl.compute
// FULL: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// FULL: ttl.cb_push %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// FULL: ttl.compute ins(%[[INTERMEDIATE_ATTACHED]]
// FULL: ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @state_update_feeds_broadcast()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_state_old = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_max = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_state_next = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>

  %state_wait = ttl.cb_wait %cb_state_old : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %state_old = ttl.attach_cb %state_wait, %cb_state_old : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %max_wait = ttl.cb_wait %cb_max : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %max_block = ttl.attach_cb %max_wait, %cb_max : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %state_new = ttl.max %state_old, %max_block : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %state_reserve = ttl.cb_reserve %cb_state_next : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %state_new, %state_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  %broadcast = ttl.block.broadcast %state_new dims = [-1], shape = [1, 2] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %out_reserve = ttl.cb_reserve %cb_out : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %broadcast, %out_reserve : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A broadcast producer has a different input and output shape. The compiler
// DFB output must use the broadcast result shape, not the producer input shape.

// DEFERRED-LABEL: func.func @broadcast_result_then_reduce
// DEFERRED: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated} : <[1, 2],
// DEFERRED: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// DEFERRED: tensor.empty() : tensor<1x2x!ttcore.tile<32x32, bf16>>
// DEFERRED: %{{.*}}:2 = ttl.compute
// DEFERRED: ttl.tile_bcast
// DEFERRED: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// DEFERRED: ttl.cb_push %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// DEFERRED: ttl.reduce %[[INTERMEDIATE_ATTACHED]]

// FULL-LABEL: func.func @broadcast_result_then_reduce
// FULL: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated} : <[1, 2],
// FULL: ttl.cb_push %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// FULL: ttl.compute ins(%[[INTERMEDIATE_ATTACHED]],
// FULL: ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @broadcast_result_then_reduce()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_scaler = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_bcast = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %cb_scaler : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %broadcast = ttl.block.broadcast %a dims = [-1], shape = [1, 2] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %broadcast_reserve = ttl.cb_reserve %cb_bcast : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %broadcast, %broadcast_reserve : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>

  %reduced = ttl.reduce %broadcast, %scaler 0 : i32 [1] : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_reserve = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %out_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Branch-local consumers share one materialized value when the attach is
// defined before the branch. This keeps the compiler DFB push/wait/pop balanced.

// DEFERRED-LABEL: func.func @branch_local_reductions
// DEFERRED: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// DEFERRED: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// DEFERRED: %{{.*}}:2 = ttl.compute
// DEFERRED: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// DEFERRED: ttl.cb_push %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// DEFERRED: scf.if
// DEFERRED: ttl.reduce %[[INTERMEDIATE_ATTACHED]]
// DEFERRED: else
// DEFERRED: ttl.reduce %[[INTERMEDIATE_ATTACHED]]

// FULL-LABEL: func.func @branch_local_reductions
// FULL: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// FULL: ttl.cb_push %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// FULL: scf.if
// FULL: ttl.compute ins(%[[INTERMEDIATE_ATTACHED]],
// FULL: else
// FULL: ttl.compute ins(%[[INTERMEDIATE_ATTACHED]],
// FULL: ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @branch_local_reductions(%cond: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_scaler = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_sum = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_then = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_else = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_wait = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %b_wait, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %cb_scaler : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %sum = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum_reserve = ttl.cb_reserve %cb_sum : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %sum_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  scf.if %cond {
    %sum_reduce = ttl.reduce %sum, %scaler 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %then_reserve = ttl.cb_reserve %cb_then : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %sum_reduce, %then_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %max_reduce = ttl.reduce %sum, %scaler 1 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %else_reserve = ttl.cb_reserve %cb_else : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %max_reduce, %else_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}
