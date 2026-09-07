// Singleton views keep producer acquisition ownership: publishing a DFB before
// a later store through its shape view is invalid.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))'

// A producer push cannot precede a store through a singleton shape view.
func.func @published_producer_view(%input: tensor<2x3x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
  %output = ttl.cb_reserve %output_dfb
      : <[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x3x1x!ttcore.tile<32x32, f32>>
  %squeezed = tensor.collapse_shape %output [[0], [1, 2]]
      : tensor<2x3x1x!ttcore.tile<32x32, f32>>
        into tensor<2x3x!ttcore.tile<32x32, f32>>
  // expected-error @below {{dataflow buffer push must follow all uses owned by its acquisition}}
  ttl.cb_push %output_dfb : <[2, 3, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.store %input, %squeezed
      : tensor<2x3x!ttcore.tile<32x32, f32>>,
        tensor<2x3x!ttcore.tile<32x32, f32>>
  return
}
