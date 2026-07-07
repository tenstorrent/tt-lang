// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes))' --split-input-file | FileCheck %s --check-prefix=FORM
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes, ttl-lower-accumulation-scopes))' --split-input-file | FileCheck %s --check-prefix=LOWER

// Summary: Verifies tensor recurrence scopes form and lower to streaming
// DST-resident accumulation.

// FORM-LABEL: func.func @tensor_recurrence_scope
// FORM: ttl.accumulation_scope outs(%[[OUT:.*]] : tensor<1x1x!ttcore.tile<32x32, bf16>>) inits(%[[INIT:.*]] : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
// FORM:   scf.for
// FORM:   ttl.store %{{.*}}, %[[OUT]]
// FORM:   ttl.yield

// LOWER-LABEL: func.func @tensor_recurrence_scope
// LOWER: %[[CONTRIB_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 1
// LOWER: ttl.dst_section
// LOWER: ttl.copy_tile
// LOWER: scf.for
// LOWER: %[[WAIT:.*]] = ttl.cb_wait %[[CONTRIB_CB]]
// LOWER-NOT: num_tiles
// LOWER: ttl.attach_cb %[[WAIT]], %[[CONTRIB_CB]]
// LOWER: ttl.tile_accumulate %{{.*}}, %{{.*}} add into dst[%{{.*}}]
// LOWER: ttl.cb_pop %[[CONTRIB_CB]]
// LOWER-NOT: num_tiles
// LOWER: ttl.tile_store
// LOWER-NOT: ttl.compute
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

// -----

// The commuted form `acc = contribution + acc` forms and lowers identically.
// FORM-LABEL: func.func @commuted_recurrence_scope
// FORM: ttl.accumulation_scope outs({{.*}}) inits({{.*}}) {
// FORM: scf.for
// LOWER-LABEL: func.func @commuted_recurrence_scope
// LOWER: ttl.dst_section
// LOWER: scf.for
// LOWER: ttl.tile_accumulate %{{.*}}, %{{.*}} add into dst[%{{.*}}]
// LOWER-NOT: ttl.compute
// LOWER-NOT: ttl.accumulation_scope
func.func @commuted_recurrence_scope() {
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
    %next = ttl.add %contrib, %acc : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A dynamic trip count forms because streaming DST preserves the source loop.
// FORM-LABEL: func.func @dynamic_trip_recurrence_scope
// FORM: ttl.accumulation_scope outs({{.*}}) inits({{.*}}) {
// FORM: scf.for
// LOWER-LABEL: func.func @dynamic_trip_recurrence_scope
// LOWER: ttl.dst_section
// LOWER: scf.for
// LOWER: ttl.cb_wait
// LOWER: ttl.tile_accumulate
// LOWER: ttl.cb_pop
// LOWER: ttl.tile_store
// LOWER-NOT: ttl.compute
// LOWER-NOT: ttl.accumulation_scope
func.func @dynamic_trip_recurrence_scope(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result = scf.for %iv = %c0 to %n step %c1
      iter_args(%acc = %init)
      -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contrib_wait = ttl.cb_wait %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %acc, %contrib : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A zero trip count forms because the lowered loop executes zero contribution
// updates and stores the seeded init tile.
// FORM-LABEL: func.func @zero_trip_recurrence_scope
// FORM: ttl.accumulation_scope outs({{.*}}) inits({{.*}}) {
// FORM: scf.for
// LOWER-LABEL: func.func @zero_trip_recurrence_scope
// LOWER: ttl.dst_section
// LOWER: ttl.copy_tile
// LOWER: scf.for
// LOWER: ttl.tile_accumulate
// LOWER: ttl.tile_store
// LOWER-NOT: ttl.compute
// LOWER-NOT: ttl.accumulation_scope
func.func @zero_trip_recurrence_scope() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %result = scf.for %iv = %c0 to %c0 step %c1
      iter_args(%acc = %init)
      -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contrib_wait = ttl.cb_wait %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %acc, %contrib : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
