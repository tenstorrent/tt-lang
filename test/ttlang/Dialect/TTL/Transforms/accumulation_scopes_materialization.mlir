// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes, ttl-lower-accumulation-scopes, ttl-materialize-loop-state, ttl-insert-copy-wait, ttl-auto-sync, ttl-insert-accumulation-scopes{kind=dfb}, ttl-lower-accumulation-scopes{kind=dfb}, ttl-create-producer-compute, ttl-insert-intermediate-dfbs, convert-ttl-to-compute, ttl-auto-sync))' | FileCheck %s

// Verify that a recurrence rejected from DST-resident accumulation uses
// intermediate DFB materialization instead.

// Two updates in one iteration require preserving the first result across the
// second contribution wait. The fallback assigns separate DFBs to loop state
// and the intermediate result.
// CHECK-LABEL: func.func @two_streamed_updates
// CHECK: %[[CONTRIB:.*]] = ttl.bind_cb{{.*}}cb_index = 1
// CHECK: %[[STATE:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: %[[MID:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: scf.for
// CHECK: ttl.cb_wait %[[CONTRIB]]
// CHECK: ttl.compute
// CHECK: ttl.cb_push %[[MID]]
// CHECK: ttl.cb_wait %[[MID]]
// CHECK: ttl.cb_pop %[[CONTRIB]]
// CHECK: ttl.cb_wait %[[CONTRIB]]
// CHECK: ttl.compute
func.func @two_streamed_updates() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c40 = arith.constant 40 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = scf.for %iv = %c0 to %c40 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %pos_wait = ttl.cb_wait %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %pos = ttl.attach_cb %pos_wait, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mid = ttl.add %acc, %pos : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %neg_wait = ttl.cb_wait %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %neg = ttl.attach_cb %neg_wait, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %mid, %neg : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
