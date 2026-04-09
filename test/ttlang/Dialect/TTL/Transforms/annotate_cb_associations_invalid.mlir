// Negative tests for ttl-annotate-cb-associations pass error diagnostics.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-annotate-cb-associations))'

// Purpose: tile_bcast output without an attached circular buffer.
// The output is a bare tile (block argument), not from a tensor with an
// attached CB. getAttachedCB returns null.
func.func @bcast_output_no_cb(%out_tile: !ttcore.tile<32x32, bf16>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %input = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input_cb = ttl.attach_cb %input, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %c0 = arith.constant 0 : index
  %in_tile = tensor.extract %input_cb[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>

  // expected-error @+1 {{output does not have an attached circular buffer}}
  %bcast = ttl.tile_bcast %in_tile, %out_tile 3 : i32 into dst[%c0] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>

  return
}

// -----

// Purpose: tile_bcast output traces to a CB that is not from ttl.bind_cb.
// The CB is a function argument, so the pass can't extract cb_index needed
// for broadcast lowering.
// expected-note @+1 {{circular buffer defined here}}
func.func @bcast_output_cb_not_from_bind(%cb_arg: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %input = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input_cb = ttl.attach_cb %input, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Output traces through cb_wait to %cb_arg (a function argument, not bind_cb).
  %out_tensor = ttl.cb_wait %cb_arg : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %c0 = arith.constant 0 : index
  %in_tile = tensor.extract %input_cb[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_tile = tensor.extract %out_tensor[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>

  // expected-error @+1 {{output circular buffer is not from ttl.bind_cb; cb_index required for broadcast lowering}}
  %bcast = ttl.tile_bcast %in_tile, %out_tile 3 : i32 into dst[%c0] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>

  return
}
