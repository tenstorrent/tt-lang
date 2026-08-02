// Tests deferred compiler DFB materialization after initial `ComputeOp` creation.
// The pass must materialize ttl.compute results as extra compute outputs and
// leave consumers ready for the final convert-ttl-to-compute pass.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs))' | FileCheck %s --check-prefix=DEFERRED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s --check-prefix=FULL
// The unsplit run covers concurrent processing of multiple kernels.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs))' | FileCheck %s --check-prefix=MULTIFUNC

// MULTIFUNC-LABEL: func.func @stored_add_then_reduce
// MULTIFUNC: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// MULTIFUNC-LABEL: func.func @state_update_feeds_broadcast
// MULTIFUNC: ttl.bind_cb{{.*}} {ttl.compiler_allocated}

// -----

// A producer compute result is both stored to a user DFB and consumed by a
// reduce. Intermediate DFB materialization must add a second compute output for
// the compiler DFB instead of emitting a tensor-level store after the compute.

// DEFERRED-LABEL: func.func @stored_add_then_reduce
// DEFERRED: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
// DEFERRED: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// DEFERRED: %{{.*}}:2 = ttl.compute
// DEFERRED: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// DEFERRED: ttl.cb_push %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// DEFERRED: ttl.reduce %[[INTERMEDIATE_ATTACHED]]

// FULL-LABEL: func.func @stored_add_then_reduce
// FULL: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
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
// DEFERRED: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
// DEFERRED: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// DEFERRED: %{{.*}}:2 = ttl.compute
// DEFERRED: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// DEFERRED: ttl.cb_push %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// DEFERRED: ttl.block.broadcast %[[INTERMEDIATE_ATTACHED]]

// FULL-LABEL: func.func @state_update_feeds_broadcast
// FULL: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
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

// A same-DFB state update also feeds a broadcast. The compiler DFB for the
// broadcast input must use the deferred compute-output materialization form.

// DEFERRED-LABEL: func.func @same_dfb_state_update_feeds_broadcast
// DEFERRED: %[[STATE:.*]] = ttl.bind_cb{{.*cb_index = 0}}
// DEFERRED: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
// DEFERRED: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// DEFERRED: %{{.*}}:2 = ttl.compute
// DEFERRED: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// DEFERRED: ttl.cb_push %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// DEFERRED: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// DEFERRED: ttl.block.broadcast %[[INTERMEDIATE_ATTACHED]]

// FULL-LABEL: func.func @same_dfb_state_update_feeds_broadcast
// FULL: %[[STATE:.*]] = ttl.bind_cb{{.*cb_index = 0}}
// FULL: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
// FULL: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// FULL: %{{.*}}:2 = ttl.compute
// FULL: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// FULL: ttl.cb_push %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// FULL: ttl.compute ins(%[[INTERMEDIATE_ATTACHED]]
// FULL: ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @same_dfb_state_update_feeds_broadcast()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb_state = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_max = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>

  %state_wait = ttl.cb_wait %cb_state : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %state_old = ttl.attach_cb %state_wait, %cb_state : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %max_wait = ttl.cb_wait %cb_max : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %max_block = ttl.attach_cb %max_wait, %cb_max : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %state_new = ttl.max %state_old, %max_block : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %state_reserve = ttl.cb_reserve %cb_state : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
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
// DEFERRED: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 2], {{.*}}, 1>
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
// FULL: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 2], {{.*}}, 1>
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
// DEFERRED: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
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
// FULL: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
// FULL: ttl.cb_push %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// FULL: %[[INTERMEDIATE_ATTACHED:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// FULL: scf.if
// FULL: ttl.compute ins(%[[INTERMEDIATE_ATTACHED]],
// FULL: else
// FULL: ttl.compute ins(%[[INTERMEDIATE_ATTACHED]],
// FULL-NOT: ttl.cb_pop
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

// -----

// Out-of-order consumers of different results from one multi-output compute
// reuse producer-result materializations planned before rewriting the compute.

// DEFERRED-LABEL: func.func @multi_output_reuse_after_rebuild
// DEFERRED: %[[A_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
// DEFERRED: %[[B_DFB:.*]] = ttl.bind_cb{{.*block_count = 1.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
// DEFERRED-NOT: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// DEFERRED: %{{.*}}:4 = ttl.compute
// DEFERRED: ttl.cb_push %[[A_DFB]]
// DEFERRED: %[[A_WAIT:.*]] = ttl.cb_wait %[[A_DFB]]
// DEFERRED: %[[A_ATTACHED:.*]] = ttl.attach_cb %[[A_WAIT]], %[[A_DFB]]
// DEFERRED: ttl.cb_push %[[B_DFB]]
// DEFERRED: %[[B_WAIT:.*]] = ttl.cb_wait %[[B_DFB]]
// DEFERRED: %[[B_ATTACHED:.*]] = ttl.attach_cb %[[B_WAIT]], %[[B_DFB]]
// DEFERRED: ttl.reduce %[[A_ATTACHED]]
// DEFERRED: ttl.reduce %[[B_ATTACHED]]
// DEFERRED: ttl.reduce %[[A_ATTACHED]]
func.func @multi_output_reuse_after_rebuild()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c-1 = arith.constant -1 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_scaler = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_sum = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_prod = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out0 = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out1 = ttl.bind_cb {cb_index = 6, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out2 = ttl.bind_cb {cb_index = 7, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_wait = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %b_wait, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %cb_scaler : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %cb_scaler : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %sum_reserve = ttl.cb_reserve %cb_sum : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %prod_reserve = ttl.cb_reserve %cb_prod : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum_empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum_out = ttl.attach_cb %sum_empty, %cb_sum : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %prod_empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %prod_out = ttl.attach_cb %prod_empty, %cb_prod : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result:2 = ttl.compute
      ins(%a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                   tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%sum_out, %prod_out : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                  tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [
         affine_map<(d0, d1) -> (d0, d1)>,
         affine_map<(d0, d1) -> (d0, d1)>,
         affine_map<(d0, d1) -> (d0, d1)>,
         affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, bf16>,
       %b_tile: !ttcore.tile<32x32, bf16>,
       %sum_tile_out: !ttcore.tile<32x32, bf16>,
       %prod_tile_out: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum_tile = ttl.tile_add %a_tile, %b_tile into dst[%c-1] {ttl.dst_placeholder} : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %prod_tile = ttl.tile_mul %a_tile, %b_tile into dst[%c-1] {ttl.dst_placeholder} : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %sum_tile, %sum_reserve[%i, %j] from dst[%c-1] {ttl.dst_placeholder} : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.tile_store %prod_tile, %prod_reserve[%i, %j] from dst[%c-1] {ttl.dst_placeholder} : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)

  %reduce_sum0 = ttl.reduce %result#0, %scaler 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out0_reserve = ttl.cb_reserve %cb_out0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduce_sum0, %out0_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  %reduce_prod = ttl.reduce %result#1, %scaler 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1_reserve = ttl.cb_reserve %cb_out1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduce_prod, %out1_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>

  %reduce_sum1 = ttl.reduce %result#0, %scaler 1 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out2_reserve = ttl.cb_reserve %cb_out2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduce_sum1, %out2_reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A foldable branch condition retains the same single source materialization
// as the runtime-conditioned form. The dead branch has no availability
// requirement and cannot cause an additional compiler DFB or a rejection.
// DEFERRED-LABEL: func.func @folded_branch_local_reductions
// DEFERRED: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// DEFERRED-NOT: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// DEFERRED: scf.if
// DEFERRED: ttl.reduce
// DEFERRED: else
// DEFERRED: ttl.reduce
// FULL-LABEL: func.func @folded_branch_local_reductions
// FULL: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// FULL-NOT: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// FULL: scf.if
// FULL: ttl.compute
// FULL: else
// FULL: ttl.compute
func.func @folded_branch_local_reductions()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %never = arith.constant false
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %sum_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %then_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %else_dfb = ttl.bind_cb {cb_index = 5, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %lhs_wait = ttl.cb_wait %lhs_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %lhs = ttl.attach_cb %lhs_wait, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_wait = ttl.cb_wait %rhs_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %lhs, %rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum_output = ttl.cb_reserve %sum_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %sum_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %never {
    %then_result = ttl.reduce %sum, %scaler 0 : i32 [1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %then_output = ttl.cb_reserve %then_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %then_result, %then_output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %else_result = ttl.reduce %sum, %scaler 1 : i32 [1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %else_output = ttl.cb_reserve %else_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %else_result, %else_output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}
