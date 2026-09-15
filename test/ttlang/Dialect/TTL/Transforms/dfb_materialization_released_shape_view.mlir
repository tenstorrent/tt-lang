// A computed producer is stored before its input release even when a later
// singleton view is the value published to the output DFB.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s --implicit-check-not=ttl.store

// CHECK-LABEL: func.func @released_before_shape_view
// CHECK: %[[INPUT:.*]] = ttl.bind_cb{cb_index = 0,
// CHECK: %[[MID:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK-NOT: ttl.compiler_allocated
// CHECK: ttl.compute
// CHECK: ttl.tile_neg
// CHECK: ttl.cb_push %[[MID]]
// CHECK-NEXT: %[[WAIT:.*]] = ttl.cb_wait %[[MID]]
// CHECK-NEXT: %[[VALUE:.*]] = ttl.attach_cb %[[WAIT]], %[[MID]]
// CHECK-NEXT: ttl.cb_pop %[[INPUT]]
// CHECK-NEXT: %[[VIEW:.*]] = tensor.expand_shape %[[VALUE]]
// CHECK: ttl.compute ins(%[[VIEW]]
// CHECK: ttl.cb_pop %[[MID]]
func.func @released_before_shape_view()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %negative = ttl.neg %input
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %view = tensor.expand_shape %negative [[0, 1], [2]] output_shape [1, 2, 2]
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        into tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %view, %output
      : tensor<1x2x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  return
}
