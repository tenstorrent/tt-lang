// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute))' | FileCheck %s --check-prefix=PRODUCER
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,convert-ttl-to-compute))' | FileCheck %s --check-prefix=FINAL

// Summary: producer creation and the final compute conversion leave ttl.topk
// and its result stores in place. Those stores are not elementwise compute
// creations and not passthrough copies.

// PRODUCER-LABEL: func.func @topk_store_is_not_a_compute_creation
// PRODUCER: ttl.topk
// PRODUCER: ttl.store
// PRODUCER: ttl.store
// PRODUCER-NOT: ttl.compute
// FINAL-LABEL: func.func @topk_store_is_not_a_compute_creation
// FINAL: ttl.topk
// FINAL: ttl.store
// FINAL: ttl.store
// FINAL-NOT: ttl.compute
func.func @topk_store_is_not_a_compute_creation()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 2>
  %values_wait = ttl.cb_wait %values_cb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %values = ttl.attach_cb %values_wait, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_wait = ttl.cb_wait %indices_cb
      : <[1, 2], !ttcore.tile<32x32, u16>, 2> -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %indices = ttl.attach_cb %indices_wait, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values, %indices k = 32
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 2> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  ttl.cb_push %out_values_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %out_indices_cb : <[1, 1], !ttcore.tile<32x32, u16>, 2>
  return
}
