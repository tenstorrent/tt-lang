// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs))' | FileCheck %s

// Summary: ttl-insert-intermediate-dfbs decisions for ttl.topk. Both operands
// are dataflow-buffer inputs, so a computed values or indices tensor is
// packed before the sort. A multiply of a topk result is packed after the
// sort. An already attached operand, a direct store, and an elementwise use
// other than a multiply are left in place.

// CHECK-LABEL: func.func @computed_values_are_materialized
// CHECK: ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.add
// CHECK: ttl.cb_reserve
// CHECK: ttl.store
// CHECK: ttl.cb_wait
// CHECK: ttl.attach_cb
// CHECK: ttl.topk
func.func @computed_values_are_materialized()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %bias_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %indices_cb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>
  %values_wait = ttl.cb_wait %values_cb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %values = ttl.attach_cb %values_wait, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %bias_wait = ttl.cb_wait %bias_cb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %bias = ttl.attach_cb %bias_wait, %bias_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_wait = ttl.cb_wait %indices_cb
      : <[1, 2], !ttcore.tile<32x32, u16>, 2> -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %indices = ttl.attach_cb %indices_wait, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %added = ttl.add %values, %bias
      : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %out_values, %out_indices = ttl.topk %added, %indices k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// CHECK-LABEL: func.func @computed_indices_are_materialized
// CHECK: ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.add
// CHECK: ttl.cb_reserve
// CHECK: ttl.store
// CHECK: ttl.cb_wait
// CHECK: ttl.attach_cb
// CHECK: ttl.topk
func.func @computed_indices_are_materialized()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>
  %bias_cb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>
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
  %bias_wait = ttl.cb_wait %bias_cb
      : <[1, 2], !ttcore.tile<32x32, u16>, 2> -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %bias = ttl.attach_cb %bias_wait, %bias_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %added = ttl.add %indices, %bias
      : tensor<1x2x!ttcore.tile<32x32, u16>>, tensor<1x2x!ttcore.tile<32x32, u16>>
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values, %added k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// CHECK-LABEL: func.func @attached_operands_stay_attached
// CHECK-NOT: ttl.compiler_allocated
// CHECK: ttl.topk
func.func @attached_operands_stay_attached()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>
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
  %out_values, %out_indices = ttl.topk %values, %indices k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  return
}

// -----

// CHECK-LABEL: func.func @mul_unary_const_of_result_is_materialized
// CHECK: ttl.compiler_allocated
// CHECK: ttl.topk
// CHECK: ttl.cb_reserve
// CHECK: ttl.store
// CHECK: ttl.cb_wait
// CHECK: ttl.attach_cb
// CHECK: ttl.mul_unary_const
// CHECK: ttl.store
func.func @mul_unary_const_of_result_is_materialized()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>
  %out_indices_cb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
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
  %out_values, %out_indices = ttl.topk %values, %indices k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  %scaled = ttl.mul_unary_const %out_values, 2.000000e+00
      : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 2> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

// CHECK-LABEL: func.func @mul_of_result_is_materialized
// CHECK: ttl.compiler_allocated
// CHECK: ttl.topk
// CHECK: ttl.attach_cb
// CHECK: ttl.mul
func.func @mul_of_result_is_materialized()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>
  %scale_cb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
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
  %scale_wait = ttl.cb_wait %scale_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scale = ttl.attach_cb %scale_wait, %scale_cb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_values, %out_indices = ttl.topk %values, %indices k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  %scaled = ttl.mul %out_values, %scale
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// CHECK-LABEL: func.func @abs_of_result_is_not_materialized
// CHECK-NOT: ttl.compiler_allocated
// CHECK: ttl.topk
// CHECK: ttl.abs
func.func @abs_of_result_is_not_materialized()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 2>
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
  %out_values, %out_indices = ttl.topk %values, %indices k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  %absolute = ttl.abs %out_values
      : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
