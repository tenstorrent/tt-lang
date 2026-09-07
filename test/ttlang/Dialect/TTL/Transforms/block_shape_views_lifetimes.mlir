// Singleton views and later associations retain the original acquisition
// identity, including its released state.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-dfb-value-lifetimes))' -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=S0

// Attaching a squeezed and expanded tensor does not create a new entry-state
// association, so the original consumer pop remains visible afterward.
// CHECK-LABEL: DFB value lifetimes @released_consumer_view
// CHECK: A0 consumer tiles=6
// CHECK-NEXT: R0 consumer tiles=6 owner=exact A0
// CHECK: tensor.collapse_shape A0=available
// CHECK-NEXT: {{.*}}tensor.expand_shape A0=available
// CHECK: ttl.attach_cb A0=may-be-released
// CHECK-NEXT: {{.*}}ttl.signpost A0=may-be-released
func.func @released_consumer_view()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x3x!ttcore.tile<32x32, bf16>>
  %squeezed = tensor.collapse_shape %input [[0, 1], [2]]
      : tensor<1x2x3x!ttcore.tile<32x32, bf16>>
        into tensor<2x3x!ttcore.tile<32x32, bf16>>
  %expanded = tensor.expand_shape %squeezed [[0], [1, 2]] output_shape [2, 1, 3]
      : tensor<2x3x!ttcore.tile<32x32, bf16>>
        into tensor<2x1x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb : <[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>
  %attached = ttl.attach_cb %expanded, %input_dfb
      : (tensor<2x1x3x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 2, 3], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<2x1x3x!ttcore.tile<32x32, bf16>>
  ttl.signpost "after shape-view association"
  return
}
