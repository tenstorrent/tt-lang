// Tests materialization when `ComputeOp` creation at an output store would not
// dominate another result use.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-auto-sync))' | FileCheck %s --check-prefix=FULL

// The reduce consumes the multiply result before its user-DFB store. Moving
// the multiply to that store would violate SSA dominance. Materializing every
// result use leaves the compiler-DFB store as the multiply's only publication,
// while the original store becomes a passthrough compute.

// PLAN-LABEL: ComputeOp creation plan @nonstore_use_before_store
// PLAN:       ttl.mul kind=direct recipe=elementwise legal=false
// PLAN-NEXT:  iterators=
// PLAN-NEXT:  removed-before {{.*}} operand=0
// PLAN-NEXT:  rejected=ttl.compute inserted at the final output store would not dominate every surviving result use
// PLAN:       rejected-source {{.*}}ttl.reduce reason=reduce input is an unstored compute result
// PLAN:       unassigned-store {{.*}} reason=ttl.compute inserted at the final output store would not dominate every surviving result use
// PLAN-NEXT:  unassigned-store {{.*}} reason=reduce input is an unstored compute result
// PLAN:       operand=0
// PLAN:       reason=compute-op-would-not-dominate-use
// PLAN:       operand=0
// PLAN-NEXT:  reason=compute-op-would-not-dominate-use

// FULL-LABEL: func.func @nonstore_use_before_store
// FULL:       %[[USER_OUTPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 3
// FULL:       %[[REDUCE_OUTPUT_DFB:.*]] = ttl.bind_cb{{.*}}cb_index = 4
// FULL:       %[[INTERMEDIATE_DFB:.*]] = ttl.bind_cb{{.*}}ttl.compiler_allocated
// FULL-NOT:   ttl.compiler_allocated
// FULL:       %[[USER_OUTPUT_RESERVE:.*]] = ttl.cb_reserve %[[USER_OUTPUT_DFB]]
// FULL:       %[[REDUCE_OUTPUT_RESERVE:.*]] = ttl.cb_reserve %[[REDUCE_OUTPUT_DFB]]
// FULL:       %[[INTERMEDIATE_RESERVE:.*]] = ttl.cb_reserve %[[INTERMEDIATE_DFB]]
// FULL:       ttl.compute
// FULL:         ttl.tile_mul
// FULL:         ttl.tile_store {{.*}}, %[[INTERMEDIATE_RESERVE]]
// FULL:       ttl.cb_push %[[INTERMEDIATE_DFB]]
// FULL:       %[[INTERMEDIATE_WAIT:.*]] = ttl.cb_wait %[[INTERMEDIATE_DFB]]
// FULL:       %[[INTERMEDIATE:.*]] = ttl.attach_cb %[[INTERMEDIATE_WAIT]], %[[INTERMEDIATE_DFB]]
// FULL:       ttl.compute ins(%[[INTERMEDIATE]]
// FULL:         ttl.tile_store {{.*}}, %[[USER_OUTPUT_RESERVE]]
// FULL:       ttl.compute ins(%[[INTERMEDIATE]],
// FULL:         ttl.tile_reduce
// FULL:         ttl.tile_store {{.*}}, %[[REDUCE_OUTPUT_RESERVE]]
// FULL-NOT:   ttl.mul
// FULL-NOT:   ttl.reduce
// FULL-NOT:   ttl.store
func.func @nonstore_use_before_store()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %user_output_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
  %reduce_output_dfb = ttl.bind_cb {cb_index = 4, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %rhs_wait = ttl.cb_wait %rhs_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %user_output = ttl.cb_reserve %user_output_dfb
      : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %reduce_output = ttl.cb_reserve %reduce_output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %product = ttl.mul %input, %rhs
      : tensor<1x4x!ttcore.tile<32x32, bf16>>,
        tensor<1x4x!ttcore.tile<32x32, bf16>>
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %result = ttl.reduce %product, %scaler 0 : i32 [0, 1]
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
         tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %product, %user_output
      : tensor<1x4x!ttcore.tile<32x32, bf16>>,
        tensor<1x4x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %reduce_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A view derived from the second wait retains that acquisition's lifetime
// through attach and slice operations. Popping the first wait therefore does
// not require an intermediate DFB for the second view.

// PLAN-LABEL: ComputeOp creation plan @slice_of_second_acquisition
// PLAN:       ttl.exp kind=direct recipe=elementwise legal=true
// PLAN:       ttl.exp kind=direct recipe=elementwise legal=true
// PLAN-NOT:   materialize

// FULL-LABEL: func.func @slice_of_second_acquisition
// FULL:       %[[INPUT_DFB:.*]] = ttl.bind_cb{cb_index = 0
// FULL:       %[[INPUT_WAIT:.*]] = ttl.cb_wait %[[INPUT_DFB]] {num_tiles = 2 : i64}
// FULL:       tensor.extract_slice %[[INPUT_WAIT]][0, 0]
// FULL:       %[[SECOND_TILE:.*]] = tensor.extract_slice %[[INPUT_WAIT]][0, 1]
// FULL:       %[[SECOND_VIEW:.*]] = ttl.attach_cb %[[SECOND_TILE]], %[[INPUT_DFB]]
// FULL:       %[[SECOND_SLICE:.*]] = tensor.extract_slice %[[SECOND_VIEW]]
// FULL:       ttl.compute ins(%[[SECOND_SLICE]]
// FULL:       ttl.cb_pop %[[INPUT_DFB]] {num_tiles = 2 : i64}
// FULL-NOT:   ttl.compiler_allocated
// FULL-NOT:   ttl.exp
// FULL-NOT:   ttl.store
func.func @slice_of_second_acquisition()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 4}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first = ttl.attach_cb %first_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second = ttl.attach_cb %second_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second_slice = tensor.extract_slice %second[0, 0] [1, 1] [1, 1]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        to tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first_output = ttl.cb_reserve %first_output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first_result = ttl.exp %first
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %first_result, %first_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.exp %second_slice
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
