// The complete pipeline must reject non-singleton DFB reshapes before any
// canonicalization or materialization can reinterpret them as storage views.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --ttl-to-ttkernel-pipeline

// A four-tile DFB cannot be reinterpreted as a two-by-two block.
func.func @general_reshape_in_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[4], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<4x!ttcore.tile<32x32, bf16>>
  %view = tensor.expand_shape %input [[0, 1]] output_shape [2, 2]
      : tensor<4x!ttcore.tile<32x32, bf16>>
        into tensor<2x2x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %result = ttl.exp %view : tensor<2x2x!ttcore.tile<32x32, bf16>>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: store input is not dataflow-buffer-backed}}
  ttl.store %result, %output
      : tensor<2x2x!ttcore.tile<32x32, bf16>>,
        tensor<2x2x!ttcore.tile<32x32, bf16>>
  return
}
