// A published compute result supplies both a shape view and a DFB-only
// reduction through one atomic materialization, preserving its original store.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s --implicit-check-not=ttl.store

// CHECK-LABEL: func.func @published_shape_view
// CHECK: %[[PUBLISHED:.*]] = ttl.bind_cb{cb_index = 2,
// CHECK: %[[MID:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK-NOT: ttl.compiler_allocated
// CHECK: %[[ORIGINAL:.*]] = ttl.cb_reserve %[[PUBLISHED]]
// CHECK: %[[MATERIALIZED:.*]] = ttl.cb_reserve %[[MID]]
// CHECK: %{{.*}}:2 = ttl.compute
// CHECK: %[[NEGATIVE:.*]] = ttl.tile_neg
// CHECK: ttl.tile_store %[[NEGATIVE]], %[[ORIGINAL]]
// CHECK-NEXT: ttl.tile_store %[[NEGATIVE]], %[[MATERIALIZED]]
// CHECK: ttl.cb_push %[[MID]]
// CHECK: %[[WAIT:.*]] = ttl.cb_wait %[[MID]]
// CHECK-NEXT: %[[VALUE:.*]] = ttl.attach_cb %[[WAIT]], %[[MID]]
// CHECK-NEXT: %[[VIEW:.*]] = tensor.expand_shape %[[VALUE]]
// CHECK: ttl.compute ins(%[[VIEW]]
// CHECK: ttl.compute ins(%[[VALUE]],
// CHECK: ttl.tile_reduce
// CHECK: ttl.cb_pop %[[MID]]
func.func @published_shape_view()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
  %published_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reduced_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
      : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  %input = ttl.cb_wait %input_dfb
      : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %negative = ttl.neg %input
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %published = ttl.cb_reserve %published_dfb
      : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %negative, %published
      : tensor<2x2x!ttcore.tile<32x32, bf16>>,
        tensor<2x2x!ttcore.tile<32x32, bf16>>
  %view = tensor.expand_shape %negative [[0, 1], [2]] output_shape [1, 2, 2]
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        into tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %view, %output
      : tensor<1x2x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %negative, %scaler 0 : i32 [1]
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %reduced_output = ttl.cb_reserve %reduced_dfb
      : <[2, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %reduced_output
      : tensor<2x1x!ttcore.tile<32x32, bf16>>,
        tensor<2x1x!ttcore.tile<32x32, bf16>>
  return
}
