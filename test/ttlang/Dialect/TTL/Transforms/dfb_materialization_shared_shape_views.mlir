// A shared view chain preserves inner, outer, and non-view expression users.
// All publications share one stored producer; surviving views are not erased.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s --implicit-check-not=ttl.store

// CHECK-LABEL: func.func @shared_shape_views
// CHECK: %[[MID:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK-NOT: ttl.compiler_allocated
// CHECK: ttl.tile_neg
// CHECK: %[[WAIT:.*]] = ttl.cb_wait %[[MID]]
// CHECK-NEXT: %[[VALUE:.*]] = ttl.attach_cb %[[WAIT]], %[[MID]]
// CHECK-NEXT: %[[INNER:.*]] = tensor.expand_shape %[[VALUE]]
// CHECK-NEXT: %[[FIRST:.*]] = tensor.expand_shape %[[INNER]]
// CHECK-NEXT: %[[SECOND:.*]] = tensor.expand_shape %[[INNER]]
// CHECK: ttl.compute ins(%[[FIRST]]
// CHECK: ttl.compute ins(%[[SECOND]]
// CHECK: ttl.compute ins(%[[INNER]]
// CHECK: ttl.compute ins(%[[INNER]]
// CHECK: ttl.tile_exp
// CHECK: ttl.cb_pop %[[MID]]
func.func @shared_shape_views()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %first_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
  %second_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 2, 2, 1], !ttcore.tile<32x32, bf16>, 2>
  %inner_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
  %expression_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
      : !ttl.cb<[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %negative = ttl.neg %input
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %inner = tensor.expand_shape %negative [[0, 1], [2]] output_shape [1, 2, 2]
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        into tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  %first = tensor.expand_shape %inner [[0, 1], [2], [3]] output_shape [1, 1, 2, 2]
      : tensor<1x2x2x!ttcore.tile<32x32, bf16>>
        into tensor<1x1x2x2x!ttcore.tile<32x32, bf16>>
  %second = tensor.expand_shape %inner [[0], [1], [2, 3]] output_shape [1, 2, 2, 1]
      : tensor<1x2x2x!ttcore.tile<32x32, bf16>>
        into tensor<1x2x2x1x!ttcore.tile<32x32, bf16>>
  %first_output = ttl.cb_reserve %first_dfb
      : <[1, 1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %first, %first_output
      : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x2x2x!ttcore.tile<32x32, bf16>>
  %second_output = ttl.cb_reserve %second_dfb
      : <[1, 2, 2, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x2x1x!ttcore.tile<32x32, bf16>>
  ttl.store %second, %second_output
      : tensor<1x2x2x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x2x1x!ttcore.tile<32x32, bf16>>
  %inner_output = ttl.cb_reserve %inner_dfb
      : <[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %inner, %inner_output
      : tensor<1x2x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  %expression = ttl.exp %inner
      : tensor<1x2x2x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  %expression_output = ttl.cb_reserve %expression_dfb
      : <[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %expression, %expression_output
      : tensor<1x2x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  return
}
