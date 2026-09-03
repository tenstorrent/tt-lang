// Verifies that DFB materialization recognizes only singleton-dimension
// insertion and removal as zero-copy shape views.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s

// A singleton-dimension view is a valid zero-copy shape change. Its computed
// source must be materialized through the source-rank reserve view.
// CHECK-LABEL: ComputeOp creation plan @singleton_dimension_shape_view
// CHECK:       unassigned-store
// CHECK:       operand=0
// CHECK-NEXT:  reason=store-input-shape-view
func.func @singleton_dimension_shape_view()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %negative = ttl.neg %input
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %view = builtin.unrealized_conversion_cast %negative
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        to tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %view, %output
      : tensor<1x2x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x2x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Equal tile counts do not make an arbitrary extent change a shape view.
// Such a cast must not enter the compiler's singleton-view materialization
// path, which would silently reinterpret the producer's tile coordinates.
// CHECK-LABEL: ComputeOp creation plan @non_singleton_reshape
// CHECK:       unassigned-store
// CHECK-NOT:   reason=store-input-shape-view
func.func @non_singleton_reshape()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %negative = ttl.neg %input
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %reshape = builtin.unrealized_conversion_cast %negative
      : tensor<2x2x!ttcore.tile<32x32, bf16>>
        to tensor<1x4x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  ttl.store %reshape, %output
      : tensor<1x4x!ttcore.tile<32x32, bf16>>,
        tensor<1x4x!ttcore.tile<32x32, bf16>>
  return
}
