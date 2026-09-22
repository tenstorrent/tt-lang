// RUN: ttlang-opt %s --ttl-lower-topk --split-input-file --verify-diagnostics

// Summary: ttl-lower-topk rejects an operand that is not a dataflow buffer, a
// result that is not stored once into a dominating reserve, and result stores
// that do not share a block.

func.func @values_not_attached(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                               %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %indices_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  // expected-error @below {{values must be attached to a dataflow buffer}}
  %out_values, %out_indices = ttl.topk %values, %indices_attached
      k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

func.func @indices_not_attached(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                                %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  // expected-error @below {{index tensor is the identity indices}}
  %out_values, %out_indices = ttl.topk %values_attached, %indices
      k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

func.func @result_stored_twice(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                               %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %extra_cb = ttl.bind_cb {cb_index = 4, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %extra_view = ttl.cb_reserve %extra_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  // expected-error @below {{each result must be stored exactly once}}
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_values, %extra_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

func.func @store_view_is_a_wait(
    %values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_wait %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  // expected-error @below {{each result must be stored into a reserved dataflow buffer}}
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

func.func @stores_in_different_blocks(
    %values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x2x!ttcore.tile<32x32, u16>>, %cond: i1) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  // expected-error @below {{value and index results must be stored in the same block}}
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  scf.if %cond {
    ttl.store %out_values, %values_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield
  } else {
    ttl.store %out_indices, %indices_view
        : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
    scf.yield
  }
  return
}

// -----

func.func @reserve_does_not_dominate_first_store(
    %values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  // expected-error @below {{result buffer reserves must dominate both result stores}}
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
