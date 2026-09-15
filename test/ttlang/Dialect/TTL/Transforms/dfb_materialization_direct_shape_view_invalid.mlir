// A generic cast must not reinterpret an acquired DFB's tile coordinates.
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --verify-diagnostics

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
func.func @direct_non_singleton_view_store()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %view = builtin.unrealized_conversion_cast %input
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        to tensor<1x4x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{cannot lower tensor store to ttl.compute: store input is not dataflow-buffer-backed}}
  ttl.store %view, %output
      : tensor<1x4x!ttcore.tile<32x32, bf16>>,
        tensor<1x4x!ttcore.tile<32x32, bf16>>
  return
}
func.func @direct_non_singleton_view_store_io()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_reserve %input_dfb
      : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %input_dfb : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %output = ttl.cb_wait %output_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %output_dfb : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
  return
}
}
