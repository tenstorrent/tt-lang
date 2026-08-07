// Verifies direct compute creation preserves nonstandard physical tile types
// for elementwise, reduction, and broadcast recipes.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),cse,canonicalize)' | FileCheck %s

// A direct elementwise recipe uses the attached 16x32 tile type for every
// compute argument and result.
// CHECK-LABEL: func.func @elementwise_16x32
// CHECK:       ttl.compute
// CHECK-NEXT:  ^bb0(%[[LHS:.*]]: !ttcore.tile<16x32, bf16>, %[[RHS:.*]]: !ttcore.tile<16x32, bf16>, %[[OUT:.*]]: !ttcore.tile<16x32, bf16>):
// CHECK-NEXT:    %[[ROW:.*]] = ttl.iter_index 0
// CHECK-NEXT:    %[[COL:.*]] = ttl.iter_index 1
// CHECK-NEXT:    %[[PRODUCT:.*]] = ttl.tile_mul %[[LHS]], %[[RHS]] into dst
// CHECK-NEXT:    ttl.tile_store %[[PRODUCT]], %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NEXT:    ttl.yield
func.func @elementwise_16x32()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<16x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<16x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<16x32, bf16>, 2>
  %lhs_wait = ttl.cb_wait %lhs_dfb
      : <[1, 2], !ttcore.tile<16x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  %lhs = ttl.attach_cb %lhs_wait, %lhs_dfb
      : (tensor<1x2x!ttcore.tile<16x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<16x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  %rhs_wait = ttl.cb_wait %rhs_dfb
      : <[1, 2], !ttcore.tile<16x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
      : (tensor<1x2x!ttcore.tile<16x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<16x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2], !ttcore.tile<16x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  %product = ttl.mul %lhs, %rhs
      : tensor<1x2x!ttcore.tile<16x32, bf16>>,
        tensor<1x2x!ttcore.tile<16x32, bf16>>
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  ttl.store %product, %output
      : tensor<1x2x!ttcore.tile<16x32, bf16>>,
        tensor<1x2x!ttcore.tile<16x32, bf16>>
  return
}

// -----

// A direct reduction preserves 16x32 input, scaler, and output tile types.
// CHECK-LABEL: func.func @reduce_16x32
// CHECK:       ttl.compute
// CHECK-NEXT:  ^bb0(%[[INPUT:.*]]: !ttcore.tile<16x32, bf16>, %[[SCALER:.*]]: !ttcore.tile<16x32, bf16>, %[[OUT:.*]]: !ttcore.tile<16x32, bf16>):
// CHECK-NEXT:    %[[COL:.*]] = ttl.iter_index 1
// CHECK-NEXT:    %[[REDUCED:.*]] = ttl.tile_reduce %[[INPUT]], %[[SCALER]], %[[OUT]] 0 : i32 <reduce_dim_col>
// CHECK-NEXT:    ttl.tile_store %[[REDUCED]], %{{.*}}[%{{.*}}, %[[COL]]]
// CHECK-NEXT:    ttl.yield
func.func @reduce_16x32()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<16x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<16x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[2, 2], !ttcore.tile<16x32, bf16>, 2>
        -> tensor<2x2x!ttcore.tile<16x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<2x2x!ttcore.tile<16x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<16x32, bf16>, 2>)
        -> tensor<2x2x!ttcore.tile<16x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<16x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<16x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2], !ttcore.tile<16x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  %reduced = ttl.reduce %input, %scaler 0 : i32 [0]
      : (tensor<2x2x!ttcore.tile<16x32, bf16>>,
         tensor<1x1x!ttcore.tile<16x32, bf16>>)
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  ttl.store %reduced, %output
      : tensor<1x2x!ttcore.tile<16x32, bf16>>,
        tensor<1x2x!ttcore.tile<16x32, bf16>>
  return
}

// -----

// A direct scalar broadcast preserves the 16x32 physical tile type.
// CHECK-LABEL: func.func @broadcast_16x32
// CHECK:       ttl.compute
// CHECK-NEXT:  ^bb0(%[[INPUT:.*]]: !ttcore.tile<16x32, bf16>, %[[OUT:.*]]: !ttcore.tile<16x32, bf16>):
// CHECK-NEXT:    %[[ROW:.*]] = ttl.iter_index 0
// CHECK-NEXT:    %[[COL:.*]] = ttl.iter_index 1
// CHECK-NEXT:    %[[BROADCAST:.*]] = ttl.tile_bcast %[[INPUT]], %[[OUT]] 3 : i32
// CHECK-NEXT:    ttl.tile_store %[[BROADCAST]], %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NEXT:    ttl.yield
func.func @broadcast_16x32()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<16x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<16x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<16x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2], !ttcore.tile<16x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  %broadcast = ttl.block.broadcast %input dims = [0, 1], shape = [1, 2]
      : tensor<1x1x!ttcore.tile<16x32, bf16>>
        -> tensor<1x2x!ttcore.tile<16x32, bf16>>
  ttl.store %broadcast, %output
      : tensor<1x2x!ttcore.tile<16x32, bf16>>,
        tensor<1x2x!ttcore.tile<16x32, bf16>>
  return
}
