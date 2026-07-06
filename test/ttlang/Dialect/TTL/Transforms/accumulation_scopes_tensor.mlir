// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes))' --split-input-file | FileCheck %s --check-prefix=FORM
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes, ttl-lower-accumulation-scopes))' --split-input-file | FileCheck %s --check-prefix=LOWER

// Summary: Verifies tensor recurrence scopes form and lower to a DST reduction
// compute with a coalesced contribution wait.

// FORM-LABEL: func.func @tensor_recurrence_scope
// FORM: ttl.accumulation_scope outs(%[[OUT:.*]] : tensor<1x1x!ttcore.tile<32x32, bf16>>) inits(%[[INIT:.*]] : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
// FORM:   scf.for
// FORM:   ttl.store %{{.*}}, %[[OUT]]
// FORM:   ttl.yield

// LOWER-LABEL: func.func @tensor_recurrence_scope
// LOWER: %[[WAIT:.*]] = ttl.cb_wait %[[CONTRIB_CB:.*]] {num_tiles = 3 : i64}
// LOWER: %[[COALESCED:.*]] = ttl.attach_cb %[[WAIT]], %[[CONTRIB_CB]]
// LOWER: ttl.compute ins(%{{.*}}, %[[COALESCED]]
// LOWER-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// LOWER: ttl.tile_accumulate %{{.*}}, %{{.*}} add into dst[%{{.*}}]
// LOWER: ttl.tile_store
// LOWER: ttl.cb_pop %[[CONTRIB_CB]] {num_tiles = 3 : i64}
// LOWER-NOT: ttl.accumulation_scope
func.func @tensor_recurrence_scope() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result = scf.for %iv = %c0 to %c3 step %c1
      iter_args(%acc = %init)
      -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contrib_wait = ttl.cb_wait %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %acc, %contrib : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
