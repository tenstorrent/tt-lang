// Verifies intermediate DFB materialization in loops, for f32 multi-tile
// matmul results, across a two-requirement fixed point, and for multiple
// consumers of a materialized compute result.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN

// Materialization inside a loop keeps the complete compiler DFB lifecycle in
// the loop body, so every dynamic iteration publishes and releases one slot.
// CHECK-LABEL: func.func @loop_stored_add_then_reduce
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK: scf.for
// CHECK:   %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// CHECK:   ttl.compute
// CHECK:     ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// CHECK:   ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK:   %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK:   %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK:   ttl.compute ins(%[[INTERMEDIATE]],
// CHECK:   ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @loop_stored_add_then_reduce(%upper_bound: index)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %sum_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  scf.for %iteration = %zero to %upper_bound step %one {
    %lhs_wait = ttl.cb_wait %lhs_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %lhs = ttl.attach_cb %lhs_wait, %lhs_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs_wait = ttl.cb_wait %rhs_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler_wait = ttl.cb_wait %scaler_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %lhs, %rhs
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sum_output = ttl.cb_reserve %sum_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %sum_output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %sum, %scaler 0 : i32 [1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A four-tile f32 matmul result is replicated to the user and compiler DFBs
// with the same iteration indices before the reduction consumes it.
// CHECK-LABEL: func.func @f32_multitile_matmul_then_reduce
// CHECK: %[[MATMUL_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated} : <[2, 2], !ttcore.tile<32x32, f32>, 1>
// CHECK: %[[MATMUL_RESERVE:.*]] = ttl.cb_reserve %[[MATMUL_DFB]]
// CHECK: ttl.compute
// CHECK:   %[[ROW:.*]] = ttl.iter_index 0
// CHECK:   %[[COL:.*]] = ttl.iter_index 1
// CHECK:   %[[PRODUCT:.*]] = ttl.tile_matmul_block
// CHECK:   ttl.tile_store %[[PRODUCT]], {{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NEXT: ttl.tile_store %[[PRODUCT]], %[[MATMUL_RESERVE]][%[[ROW]], %[[COL]]]
// CHECK: ttl.cb_push %[[MATMUL_DFB]]
// CHECK: %[[MATMUL_WAIT:.*]] = ttl.cb_wait %[[MATMUL_DFB]]
// CHECK: %[[MATMUL:.*]] = ttl.attach_cb %[[MATMUL_WAIT]], %[[MATMUL_DFB]]
// CHECK: ttl.compute ins(%[[MATMUL]],
// CHECK: ttl.tile_reduce
// CHECK: ttl.cb_pop %[[MATMUL_DFB]]
func.func @f32_multitile_matmul_then_reduce()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %matmul_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
      : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %lhs_wait = ttl.cb_wait %lhs_dfb
      : <[2, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %lhs = ttl.attach_cb %lhs_wait, %lhs_dfb
      : (tensor<2x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %rhs_wait = ttl.cb_wait %rhs_dfb
      : <[1, 2], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
      : (tensor<1x2x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %matmul = ttl.matmul %lhs, %rhs
      : tensor<2x1x!ttcore.tile<32x32, f32>>,
        tensor<1x2x!ttcore.tile<32x32, f32>>
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %matmul_output = ttl.cb_reserve %matmul_dfb
      : <[2, 2], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %matmul, %matmul_output
      : tensor<2x2x!ttcore.tile<32x32, f32>>,
        tensor<2x2x!ttcore.tile<32x32, f32>>
  %reduced = ttl.reduce %matmul, %scaler 0 : i32 [1]
      : (tensor<2x2x!ttcore.tile<32x32, f32>>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %output = ttl.cb_reserve %output_dfb
      : <[2, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<2x1x!ttcore.tile<32x32, f32>>
  ttl.store %reduced, %output
      : tensor<2x1x!ttcore.tile<32x32, f32>>,
        tensor<2x1x!ttcore.tile<32x32, f32>>
  return
}

// -----

// The add result requires one compiler DFB for reduction, which makes the
// reduction result require a second compiler DFB for broadcast. The second
// producer compute and push must precede its wait.
// CHECK-LABEL: func.func @chained_add_reduce_broadcast
// CHECK: %[[REDUCED_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated} : <[1, 1], {{.*}}, 1>
// CHECK: %[[SUM_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated} : <[1, 2], {{.*}}, 1>
// CHECK-NOT: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK: ttl.compute
// CHECK:   ttl.tile_add
// CHECK: ttl.cb_push %[[SUM_DFB]]
// CHECK: %[[SUM_WAIT:.*]] = ttl.cb_wait %[[SUM_DFB]]
// CHECK: %[[SUM:.*]] = ttl.attach_cb %[[SUM_WAIT]], %[[SUM_DFB]]
// CHECK: ttl.compute ins(%[[SUM]],
// CHECK:   ttl.tile_reduce
// CHECK: ttl.cb_push %[[REDUCED_DFB]]
// CHECK: %[[REDUCED_WAIT:.*]] = ttl.cb_wait %[[REDUCED_DFB]]
// CHECK: %[[REDUCED:.*]] = ttl.attach_cb %[[REDUCED_WAIT]], %[[REDUCED_DFB]]
// CHECK: ttl.compute ins(%[[REDUCED]]
// CHECK:   ttl.tile_bcast
// CHECK: ttl.cb_pop %[[REDUCED_DFB]]
func.func @chained_add_reduce_broadcast()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %sum_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %reduced_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 5, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %lhs_wait = ttl.cb_wait %lhs_dfb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %lhs = ttl.attach_cb %lhs_wait, %lhs_dfb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %rhs_wait = ttl.cb_wait %rhs_dfb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %lhs, %rhs
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %sum_output = ttl.cb_reserve %sum_dfb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %sum_output
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %sum, %scaler 0 : i32 [1]
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reduced_output = ttl.cb_reserve %reduced_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %reduced_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %broadcast = ttl.block.broadcast %reduced dims = [-1], shape = [1, 2]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %broadcast, %output
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A published add becomes ttl.compute before intermediate DFB planning. Its
// reduction and elementwise consumers must read the same compiler DFB rather
// than leaving the elementwise operation attached to the compute result.
// CHECK-LABEL: func.func @published_result_with_mixed_consumers
// PLAN-LABEL: ComputeOp creation plan @published_result_with_mixed_consumers
// PLAN:       operand=0
// PLAN-NEXT:  reason=required-dfb-operand
// PLAN:       reason=compute-result-has-materialized-use
// CHECK-COUNT-2: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK-NOT: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK: ttl.compute
// CHECK:   ttl.tile_add
// CHECK: ttl.cb_push
// CHECK-NEXT: ttl.cb_push [[SUM_DFB:%[^ ]+]]
// CHECK: %[[SUM_WAIT:.*]] = ttl.cb_wait [[SUM_DFB]]
// CHECK: %[[SUM:.*]] = ttl.attach_cb %[[SUM_WAIT]], [[SUM_DFB]]
// CHECK: ttl.compute ins(%[[SUM]],
// CHECK:   ttl.tile_reduce
// CHECK: ttl.cb_push [[REDUCED_DFB:%[^ ]+]]
// CHECK: %[[REDUCED_WAIT:.*]] = ttl.cb_wait [[REDUCED_DFB]]
// CHECK: %[[REDUCED:.*]] = ttl.attach_cb %[[REDUCED_WAIT]], [[REDUCED_DFB]]
// CHECK: ttl.compute ins(%[[SUM]], %[[REDUCED]]
// CHECK:   ttl.tile_bcast
// CHECK:   ttl.tile_mul
func.func @published_result_with_mixed_consumers()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %scaler_dfb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %published_dfb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 4, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %lhs_wait = ttl.cb_wait %lhs_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %lhs = ttl.attach_cb %lhs_wait, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs_wait = ttl.cb_wait %rhs_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %lhs, %rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %published = ttl.cb_reserve %published_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %published
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %row_sum = ttl.reduce %sum, %scaler 0 : i32 [1]
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %row_sum_bcast = ttl.block.broadcast %row_sum dims = [1], shape = [1, 1]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaled = ttl.mul %sum, %row_sum_bcast
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %scaled, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Intermediate planning is independent of the element type. This f32 case
// checks that a published sum used by matmul and its elementwise accumulator
// is materialized in one shared DFB.
// CHECK-LABEL: func.func @published_result_with_matmul_consumer
// PLAN-LABEL: ComputeOp creation plan @published_result_with_matmul_consumer
// PLAN:       operand=0
// PLAN-NEXT:  reason=required-dfb-operand
// PLAN:       operand=1
// PLAN-NEXT:  reason=compute-result-has-materialized-use
// CHECK: %[[MATMUL_SUM_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK-NOT: ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// CHECK: ttl.compute
// CHECK:   ttl.tile_add
// CHECK: ttl.cb_push
// CHECK-NEXT: ttl.cb_push %[[MATMUL_SUM_DFB]]
// CHECK: %[[MATMUL_SUM_WAIT:.*]] = ttl.cb_wait %[[MATMUL_SUM_DFB]]
// CHECK: %[[MATMUL_SUM:.*]] = ttl.attach_cb %[[MATMUL_SUM_WAIT]], %[[MATMUL_SUM_DFB]]
// CHECK: ttl.compute ins(%[[MATMUL_SUM]], %[[MATMUL_SUM]], %{{.*}}
// CHECK:   ttl.tile_matmul_block
func.func @published_result_with_matmul_consumer()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %matrix_dfb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %published_dfb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 4, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %lhs_wait = ttl.cb_wait %lhs_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %lhs = ttl.attach_cb %lhs_wait, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs_wait = ttl.cb_wait %rhs_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %matrix_wait = ttl.cb_wait %matrix_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %matrix = ttl.attach_cb %matrix_wait, %matrix_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %sum = ttl.add %lhs, %rhs
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %published = ttl.cb_reserve %published_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %sum, %published
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
  %product = ttl.matmul %sum, %matrix
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %accumulated = ttl.add %product, %sum
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %accumulated, %output
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
  return
}
