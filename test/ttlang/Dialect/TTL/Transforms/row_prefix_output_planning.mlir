// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

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
