// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-form-accumulation-scopes{strategy=dst}, ttl-lower-accumulation-scopes{strategy=dst}))' --verify-diagnostics --split-input-file

// Summary: Required DST lowering diagnoses structurally matched recurrences
// that cannot use DST-resident accumulation.

// A contribution wait with a num_tiles attribute is not the default-block
// protocol used by DST-resident recurrence lowering.
func.func @explicit_num_tiles() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor accumulation scope to DST}}
  %result = scf.for %iv = %c0 to %c3 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %contrib_wait = ttl.cb_wait %contrib_cb {num_tiles = 1 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %acc, %contrib : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contrib_cb {num_tiles = 1 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A non-dataflow-buffer-backed init cannot be loaded into the accumulator DST
// slot before the reduction loop.
func.func @non_dfb_init() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor accumulation scope to DST}}
  %result = scf.for %iv = %c0 to %c3 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
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

// A 3x3 bf16 accumulator block requires nine DST slots, exceeding the default
// double-buffered bf16 capacity of eight slots.
func.func @resident_dst_capacity_overflow() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 1>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 1>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 1} : !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 1>
  %init_wait = ttl.cb_wait %init_cb : <[3, 3], !ttcore.tile<32x32, bf16>, 1> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<3x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 1>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  %out = ttl.cb_reserve %out_cb : <[3, 3], !ttcore.tile<32x32, bf16>, 1> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor accumulation scope to DST}}
  %result = scf.for %iv = %c0 to %c3 step %c1 iter_args(%acc = %init) -> (tensor<3x3x!ttcore.tile<32x32, bf16>>) {
    %contrib_wait = ttl.cb_wait %contrib_cb : <[3, 3], !ttcore.tile<32x32, bf16>, 1> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<3x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 1>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %next = ttl.add %acc, %contrib : tensor<3x3x!ttcore.tile<32x32, bf16>>, tensor<3x3x!ttcore.tile<32x32, bf16>> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %contrib_cb : <[3, 3], !ttcore.tile<32x32, bf16>, 1>
    scf.yield %next : tensor<3x3x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<3x3x!ttcore.tile<32x32, bf16>>, tensor<3x3x!ttcore.tile<32x32, bf16>>
  return
}

// A resident contribution released before the recurrence cannot be held in DST
// across the loop because the dataflow buffer slot no longer owns the tensor
// value.
func.func @resident_contribution_early_pop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_cb = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %init_wait = ttl.cb_wait %init_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = ttl.attach_cb %init_wait, %init_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %contrib_wait = ttl.cb_wait %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %contrib = ttl.attach_cb %contrib_wait, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %contrib_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out = ttl.cb_reserve %out_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor accumulation scope to DST}}
  %result = scf.for %iv = %c0 to %c3 step %c1 iter_args(%acc = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %next = ttl.add %acc, %contrib : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.store %result, %out : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
