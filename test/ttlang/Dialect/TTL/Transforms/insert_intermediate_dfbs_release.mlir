// Verifies that an intermediate is materialized before a pop that would
// otherwise release one of its fused source values.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=ELIDED-PLAN

// CHECK-LABEL: func.func @preserve_before_release
// CHECK: %[[DELTA_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 1
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.cb_wait %[[DELTA_DFB]]
// CHECK: ttl.compute
// CHECK: ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_pop %[[DELTA_DFB]]
// CHECK: ttl.cb_wait %[[DELTA_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]],
// CHECK: ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @preserve_before_release()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %initial_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %delta_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %initial_wait = ttl.cb_wait %initial_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %initial = ttl.attach_cb %initial_wait, %initial_dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %positive_wait = ttl.cb_wait %delta_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %positive = ttl.attach_cb %positive_wait, %delta_dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %intermediate = ttl.add %initial, %positive : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %delta_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %negative_wait = ttl.cb_wait %delta_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %negative = ttl.attach_cb %negative_wait, %delta_dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.add %intermediate, %negative : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A planned materialization terminates a nested fusion trace at the selected
// operand. Later elementwise operations must reuse that materialization.
// CHECK-LABEL: func.func @nested_materialization_is_reused
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK-NOT: ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.compute
// CHECK: ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_pop
// CHECK: ttl.compute ins(%[[INTERMEDIATE]]
func.func @nested_materialization_is_reused()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
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
  %sum = ttl.add %lhs, %rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %rhs_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %middle = ttl.exp %sum
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.exp %middle
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A source defined outside a nested region still must be preserved when that
// region pops the source DFB before the final consumer.
// CHECK-LABEL: func.func @preserve_inside_nested_region
// CHECK: %[[DELTA_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 1
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: scf.if
// CHECK: ttl.compute
// CHECK: ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_pop %[[DELTA_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]],
func.func @preserve_inside_nested_region(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %initial_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %delta_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %initial_wait = ttl.cb_wait %initial_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %initial = ttl.attach_cb %initial_wait, %initial_dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %positive_wait = ttl.cb_wait %delta_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %positive = ttl.attach_cb %positive_wait, %delta_dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.if %condition {
    %intermediate = ttl.add %initial, %positive : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %delta_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %negative_wait = ttl.cb_wait %delta_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %negative = ttl.attach_cb %negative_wait, %delta_dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.add %intermediate, %negative : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// A terminal elementwise result must be preserved when its output store follows
// a source pop. The attached intermediate feeds a passthrough output compute.
// CHECK-LABEL: func.func @preserve_before_output_store
// PLAN-LABEL: ComputeOp creation plan @preserve_before_output_store
// PLAN:       kind=direct recipe=elementwise legal=false
// PLAN:       rejected=moving tensor evaluation to the final output store would read a dataflow buffer value after its pop
// PLAN:       M0 {{.*}} operand=0
// PLAN-NEXT:  reason=compute-op-input-may-be-released
// CHECK: %[[INPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.cb_wait %[[INPUT_DFB]]
// CHECK: ttl.compute
// CHECK: ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_pop %[[INPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]] :
// CHECK: ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @preserve_before_output_store()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.exp %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Block broadcast uses the same input-lifetime proof as general fusion. Its
// result is materialized before the source DFB pop.
// CHECK-LABEL: func.func @preserve_block_broadcast_before_release
// CHECK: %[[INPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.compute
// CHECK: ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_pop %[[INPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]] :
func.func @preserve_block_broadcast_before_release()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.block.broadcast %input dims = [-1], shape = [1, 2]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x2x!ttcore.tile<32x32, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Fill uses the same output-transaction proof as general fusion. Repeated
// reserves of one DFB remain separate after one result materialization.
// CHECK-LABEL: func.func @preserve_fill_output_transactions
// PLAN-LABEL: ComputeOp creation plan @preserve_fill_output_transactions
// PLAN:       kind=direct recipe=fill legal=false
// PLAN-SAME:  transactions=2
// PLAN:       reason=multiple-output-transactions
// PLAN:       reason=multiple-output-transactions
// CHECK: %[[OUTPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.compute
// CHECK: ttl.tile_fill
// CHECK: ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_reserve %[[OUTPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]] :
// CHECK: ttl.cb_push %[[OUTPUT_DFB]]
// CHECK: ttl.cb_reserve %[[OUTPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]] :
// CHECK: ttl.cb_pop %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_push %[[OUTPUT_DFB]]
func.func @preserve_fill_output_transactions()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %output_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %result = ttl.fill 0.000000e+00
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %first
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %second
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// One tensor result cannot be absorbed into two reserve transactions of the
// same output DFB. Materializing it once lets each store retain its own
// compute/push transaction.
// CHECK-LABEL: func.func @preserve_repeated_output_transactions
// CHECK: %[[OUTPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 1
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: ttl.compute
// CHECK: ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_reserve %[[OUTPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]] :
// CHECK: ttl.cb_push %[[OUTPUT_DFB]]
// CHECK: ttl.cb_reserve %[[OUTPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]] :
// CHECK: ttl.cb_pop %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_push %[[OUTPUT_DFB]]
func.func @preserve_repeated_output_transactions()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.exp %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %first
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %second
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %input_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// An illegal outer fusion must defer an otherwise legal absorbed producer.
// The producer publishes both its original result and one compiler DFB before
// the input release; the outer compute then reads the compiler DFB.
// CHECK-LABEL: func.func @defer_absorbed_candidate
// CHECK: %[[INPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// CHECK: %[[SIDE_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 1
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: %[[SIDE_RESERVE:.*]] = ttl.cb_reserve %[[SIDE_DFB]]
// CHECK: %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// CHECK: ttl.compute
// CHECK:   ttl.tile_exp
// CHECK:   ttl.tile_store {{.*}}, %[[SIDE_RESERVE]]
// CHECK-NEXT: ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// CHECK: ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_pop %[[INPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]]
func.func @defer_absorbed_candidate()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %side_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %final_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %inner = ttl.exp %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %side = ttl.cb_reserve %side_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %inner, %side
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %outer = ttl.exp %inner
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %final = ttl.cb_reserve %final_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %outer, %final
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Identity typecast elision retains the input DFB's storage. If that storage
// may be released before a DFB-input consumer, one compiler DFB preserves the
// value and the replacement storage no longer constrains later creation.
// CHECK-LABEL: func.func @materialize_released_input_after_identity_elision
// CHECK: %[[INPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// CHECK: %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK-NOT: ttl.bind_cb{{.*}}ttl.compiler_allocated
// CHECK: %[[INPUT_WAIT:.*]] = ttl.cb_wait %[[INPUT_DFB]]
// CHECK: %[[INPUT:.*]] = ttl.attach_cb %[[INPUT_WAIT]], %[[INPUT_DFB]]
// CHECK-NOT: ttl.typecast
// CHECK: ttl.compute ins(%[[INPUT]] :
// CHECK: ttl.cb_push %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// CHECK: %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// CHECK: ttl.cb_pop %[[INPUT_DFB]]
// CHECK: ttl.compute ins(%[[INTERMEDIATE]],
// ELIDED-PLAN-LABEL: ComputeOp creation plan @materialize_released_input_after_identity_elision
// ELIDED-PLAN:       M0 {{.*}} operand=0
// ELIDED-PLAN-NEXT:  reason=dfb-input-may-be-released
// ELIDED-PLAN-NOT:   M1
func.func @materialize_released_input_after_identity_elision()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %identity = ttl.typecast %input
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %identity, %scaler 0 : i32 [1]
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
