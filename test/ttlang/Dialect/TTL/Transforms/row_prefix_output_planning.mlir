// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// Summary: Verifies row-prefix output planning rejects publication contracts
// that cannot preserve one valid compute output representation.

// A compact formal result cannot replace an observable full-tile result.
// CHECK-LABEL: ComputeOp creation plan @non_store_result_use
// CHECK:       rejected-source {{.*}} ttl.add
// CHECK-SAME:  reason=row-prefix output cannot preserve a non-store use of the full-tile result
// CHECK:       unassigned-store {{.*}} reason=row-prefix output cannot preserve a non-store use of the full-tile result
func.func @non_store_result_use(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 16, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %attached_lhs = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached_rhs = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
        -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  %sum = ttl.add %attached_lhs, %attached_rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x14x!ttcore.tile<1x32, bf16>>
  return %sum : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// One compute cannot publish formal outputs with different tile types.
// CHECK-LABEL: ComputeOp creation plan @different_output_tile_types
// CHECK:       rejected-source {{.*}} ttl.add
// CHECK-SAME:  reason=one compute cannot publish output dataflow buffers with different tile types
func.func @different_output_tile_types(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %full_output_dfb = ttl.bind_cb {cb_index = 16, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %compact_output_dfb = ttl.bind_cb {cb_index = 17, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %attached_lhs = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached_rhs = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %full_output = ttl.cb_reserve %full_output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %compact_output = ttl.cb_reserve %compact_output_dfb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
        -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  %sum = ttl.add %attached_lhs, %attached_rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %full_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %compact_output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x14x!ttcore.tile<1x32, bf16>>
  return
}

// -----

// Stores sharing one DFB must use one publication strategy.
// CHECK-LABEL: ComputeOp creation plan @mixed_store_kinds
// CHECK:       rejected-source {{.*}} ttl.add reason=one dataflow buffer cannot mix row-prefix and regular stores
func.func @mixed_store_kinds(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 16, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %attached_lhs = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached_rhs = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %attached_lhs, %attached_rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Row-prefix stores sharing one DFB must agree on its complete view type.
// CHECK-LABEL: ComputeOp creation plan @different_destination_types
// CHECK:       rejected-source {{.*}} ttl.add
// CHECK-SAME:  reason=row-prefix stores to one dataflow buffer require one destination tensor type
func.func @different_destination_types(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 16, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %attached_lhs = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached_rhs = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
        -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  %short_output = tensor.extract_slice %output[0, 0] [1, 13] [1, 1]
      : tensor<1x14x!ttcore.tile<1x32, bf16>>
        to tensor<1x13x!ttcore.tile<1x32, bf16>>
  %sum = ttl.add %attached_lhs, %attached_rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x14x!ttcore.tile<1x32, bf16>>
  ttl.store %sum, %short_output {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x13x!ttcore.tile<1x32, bf16>>
  return
}

// -----

// Row-normalization creation does not support compact output publication.
// CHECK-LABEL: ComputeOp creation plan @row_normalization_output
// CHECK:       rejected-source {{.*}} ttl.mul
// CHECK-SAME:  reason=row-prefix output is unsupported for row-normalization block creation
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @row_normalization_output()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mean_square = ttl.mul_unary_const %reduced, 9.765625e-04
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %epsilon = ttl.fill 1.000000e-05
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %biased = ttl.add %epsilon, %mean_square
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %inverse_rms = ttl.rsqrt %biased
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scalar = ttl.block.broadcast %inverse_rms dims = [0, 1], shape = [1, 1]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.mul %scalar, %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 14], !ttcore.tile<1x32, bf16>, 2>
          -> tensor<1x14x!ttcore.tile<1x32, bf16>>
    ttl.store %result, %output {row_prefix}
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x14x!ttcore.tile<1x32, bf16>>
    return
  }
}
