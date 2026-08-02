// Verifies whole-kernel compute-op-creation ordering and exact consumer-operand
// materialization decisions discovered by adversarial producer/use sweeps.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s --check-prefix=FULL

// A pure producer that dominates a nested consumer can be recomputed in the
// consumer's region. No storage is needed across the region boundary.
// PLAN-LABEL: ComputeOp creation plan @cross_region_elementwise_consumer
// PLAN:       ttl.exp kind=fused recipe=fused legal=true
// PLAN:       order=[C0]
// PLAN-NOT:   materialize
// FULL-LABEL: func.func @cross_region_elementwise_consumer
// FULL-NOT:   ttl.compiler_allocated
// FULL:       scf.if
// FULL:         ttl.compute
// FULL:           ttl.tile_add
// FULL:           ttl.tile_exp
// FULL-NOT:   ttl.add
// FULL-NOT:   ttl.exp
// FULL-NOT:   ttl.store
func.func @cross_region_elementwise_consumer(%condition: i1)
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
  scf.if %condition {
    %exponential = ttl.exp %sum
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %exponential, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  return
}

// -----

// Both consumers must be created before the shared producer. The publication
// store keeps the producer alive until its independent creation executes.
// PLAN-LABEL: ComputeOp creation plan @two_consumers_absorb_one_producer
// PLAN:       ttl.add kind=direct recipe=elementwise legal=true
// PLAN:       preserved-by {{.*}} operand=0
// PLAN-COUNT-2: ttl.exp kind=fused recipe=fused legal=true
// PLAN:       order=[C2, C1, C0]
// FULL-LABEL: func.func @two_consumers_absorb_one_producer
// FULL-NOT:   ttl.compiler_allocated
// FULL:       ttl.tile_add
// FULL:       ttl.tile_add
// FULL-NEXT:  ttl.tile_exp
// FULL:       ttl.tile_add
// FULL-NEXT:  ttl.tile_exp
// FULL-NOT:   ttl.add
// FULL-NOT:   ttl.exp
// FULL-NOT:   ttl.store
func.func @two_consumers_absorb_one_producer()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %published_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_output_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_output_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
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
  %published = ttl.cb_reserve %published_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %published
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first = ttl.exp %sum
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first_output = ttl.cb_reserve %first_output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %first, %first_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second = ttl.exp %sum
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second_output = ttl.cb_reserve %second_output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %second, %second_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Reduction cannot be fused into an elementwise consumer. Analysis records
// the exact consumer operand and inserts one compiler DFB before conversion.
// PLAN-LABEL: ComputeOp creation plan @unstored_reduce_to_elementwise
// PLAN:       order=[]
// PLAN:       operand=0
// PLAN-NEXT:  reason=compute-op-requires-materialized-input
// FULL-LABEL: func.func @unstored_reduce_to_elementwise
// FULL:       %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// FULL-NOT:   ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// FULL:       ttl.compute
// FULL:         ttl.tile_reduce
// FULL:       ttl.cb_push %[[INTERMEDIATE_DFB]]
// FULL:       %[[WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// FULL:       %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[WAIT]], %[[INTERMEDIATE_DFB]]
// FULL:       ttl.compute ins(%[[INTERMEDIATE]]
// FULL:         ttl.tile_exp
// FULL:       ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @unstored_reduce_to_elementwise()
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
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %input, %scaler 0 : i32 [1]
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %exponential = ttl.exp %reduced
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %exponential, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// Multiple reserve transactions prevent direct producer creation. All uses
// must consume one shared materialization so no use retains the tensor value.
// PLAN-LABEL: ComputeOp creation plan @two_transactions_and_elementwise_use
// PLAN:       ttl.reduce kind=direct recipe=reduce legal=false
// PLAN:       rejected=one compute cannot publish multiple reserve transactions of the same dataflow buffer
// PLAN-COUNT-3: reason=multiple-output-transactions
// FULL-LABEL: func.func @two_transactions_and_elementwise_use
// FULL:       %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// FULL-NOT:   ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// FULL:       ttl.compute
// FULL:         ttl.tile_reduce
// FULL:       ttl.cb_push %[[INTERMEDIATE_DFB]]
// FULL:       %[[WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// FULL:       %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[WAIT]], %[[INTERMEDIATE_DFB]]
// FULL-COUNT-3: ttl.compute ins(%[[INTERMEDIATE]]
// FULL:       ttl.cb_pop %[[INTERMEDIATE_DFB]]
func.func @two_transactions_and_elementwise_use()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %published_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %input, %scaler 0 : i32 [1]
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first_published = ttl.cb_reserve %published_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %first_published
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second_published = ttl.cb_reserve %published_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %second_published
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %exponential = ttl.exp %reduced
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %exponential, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
