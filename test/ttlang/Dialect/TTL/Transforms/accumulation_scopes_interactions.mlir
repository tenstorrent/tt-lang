// Summary: Verify tensor accumulation scope formation across multi-tile
// recurrences and non-additive work inside the recurrence loop.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes, ttl-lower-accumulation-scopes))' --split-input-file | FileCheck %s --check-prefix=DST
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes, ttl-lower-accumulation-scopes, ttl-materialize-loop-state))' --split-input-file | FileCheck %s --check-prefix=FALLBACK

// Multi-tile additive recurrence lowers to one streaming DST section. Each
// iteration consumes one 2x2 contribution block while the accumulator stays in
// DST across the source loop.
func.func @multitile_tensor_recurrence_scope() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %initial_cb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  %contribution_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  %output_cb = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>

  %initial_wait = ttl.cb_wait %initial_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %initial = ttl.attach_cb %initial_wait, %initial_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %result = scf.for %iv = %c0 to %c3 step %c1
      iter_args(%accumulator = %initial)
      -> (tensor<2x2x!ttcore.tile<32x32, bf16>>) {
    %contribution_wait = ttl.cb_wait %contribution_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %contribution = ttl.attach_cb %contribution_wait, %contribution_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %accumulator, %contribution : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contribution_cb : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
    scf.yield %next : tensor<2x2x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %output : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  return
}

// DST-LABEL: func.func @multitile_tensor_recurrence_scope
// DST: %[[CONTRIB_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 1
// DST: ttl.dst_section
// DST-COUNT-4: ttl.copy_tile
// DST: scf.for
// DST: ttl.cb_wait %[[CONTRIB_CB]] : <[2, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
// DST-COUNT-4: ttl.tile_accumulate
// DST: ttl.cb_pop %[[CONTRIB_CB]] : <[2, 2], !ttcore.tile<32x32, bf16>, 1>
// DST-COUNT-4: ttl.tile_store
// DST-NOT: ttl.compute

// -----

// Broadcast work inside the loop is not part of the additive recurrence and
// requires normal loop-state materialization.
func.func @broadcast_in_loop_falls_back() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %initial_cb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %contribution_cb = ttl.bind_cb {cb_index = 1, block_count = 3} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
  %output_cb = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>

  %initial_wait = ttl.cb_wait %initial_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %initial = ttl.attach_cb %initial_wait, %initial_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result = scf.for %iv = %c0 to %c3 step %c1
      iter_args(%accumulator = %initial)
      -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution_wait = ttl.cb_wait %contribution_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %contribution = ttl.attach_cb %contribution_wait, %contribution_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %broadcast = ttl.block.broadcast %contribution dims = [-1], shape = [1, 2] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %accumulator, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contribution_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %output : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// DST-LABEL: func.func @broadcast_in_loop_falls_back
// DST-NOT: ttl.accumulation_scope
// DST-NOT: ttl.tile_accumulate
// DST: %{{.*}} = scf.for
// FALLBACK-LABEL: func.func @broadcast_in_loop_falls_back
// FALLBACK: ttl.compiler_allocated
// FALLBACK: ttl.block.broadcast
// FALLBACK-NOT: ttl.accumulation_scope
// FALLBACK-NOT: ttl.tile_accumulate

// -----

// A reduce op in the loop is not part of the additive recurrence and requires
// normal loop-state materialization.
func.func @reduce_in_loop_falls_back() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %initial_cb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %contribution_cb = ttl.bind_cb {cb_index = 1, block_count = 3} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
  %scaler_cb = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_cb = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>

  %initial_wait = ttl.cb_wait %initial_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %initial = ttl.attach_cb %initial_wait, %initial_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result = scf.for %iv = %c0 to %c3 step %c1
      iter_args(%accumulator = %initial)
      -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contribution_wait = ttl.cb_wait %contribution_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %contribution = ttl.attach_cb %contribution_wait, %contribution_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %contribution, %scaler 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %accumulator, %contribution : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contribution_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %output : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// DST-LABEL: func.func @reduce_in_loop_falls_back
// DST-NOT: ttl.accumulation_scope
// DST-NOT: ttl.tile_accumulate
// DST: %{{.*}} = scf.for
// FALLBACK-LABEL: func.func @reduce_in_loop_falls_back
// FALLBACK: ttl.compiler_allocated
// FALLBACK: ttl.reduce
// FALLBACK-NOT: ttl.accumulation_scope
// FALLBACK-NOT: ttl.tile_accumulate
