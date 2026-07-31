// Tests whether fused-compute planning can preserve instrumentation ordering.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-formation-plans))' -o /dev/null 2>&1 | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-form-producer-compute))' | FileCheck %s --check-prefix=IR
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-form-producer-compute))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN

// Nested user signposts move into the compute body with the operations they
// observe. A following scope remains outside the compute.

// CHECK-LABEL: Compute formation plan @movable_instrumentation
// CHECK:       ttl.add kind=fused recipe=fused legal=true
// CHECK:       warning=instrumentation changes code generation: matmul-accumulator folding is disabled because the combined hardware operation cannot preserve the observation point between ttl.matmul and ttl.add; the instrumented program uses separate tile operations
// IR-LABEL:    func.func @movable_instrumentation
// IR:          ttl.compute
// IR:            ttl.signpost "ttl_scope"
// IR:            ttl.tile_matmul_block
// IR-NEXT:       ttl.signpost "ttl_math"
// IR-NEXT:       ttl.tile_add
// IR-NEXT:       ttl.signpost "ttl_math" {is_end}
// IR:            ttl.tile_store
// IR-NEXT:       ttl.signpost "ttl_scope" {is_end}
// IR:          ttl.signpost "ttl_push"
// IR-NEXT:     ttl.cb_push
// IR-NEXT:     ttl.signpost "ttl_push" {is_end}
// WARN: warning: instrumentation changes code generation: matmul-accumulator folding is disabled because the combined hardware operation cannot preserve the observation point between ttl.matmul and ttl.add; the instrumented program uses separate tile operations
func.func @movable_instrumentation()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %accumulator_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
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
  %accumulator_wait = ttl.cb_wait %accumulator_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %accumulator = ttl.attach_cb %accumulator_wait, %accumulator_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_scope"
  %product = ttl.matmul %lhs, %rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_math"
  %sum = ttl.add %product, %accumulator
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_math" {is_end}
  ttl.store %sum, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_scope" {is_end}
  ttl.signpost "ttl_push"
  ttl.cb_push %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.signpost "ttl_push" {is_end}
  return
}

// -----

// Two output stores have distinct instrumentation scopes. Formation must keep
// each tile store inside its corresponding scope instead of grouping the tile
// stores between the first begin and final end signposts.

// CHECK-LABEL: Compute formation plan @instrumented_output_stores
// CHECK:       ttl.add kind=fused recipe=fused legal=true
// IR-LABEL:    func.func @instrumented_output_stores
// IR:          ttl.compute
// IR:            ttl.tile_exp
// IR-NEXT:       ttl.tile_add
// IR-NEXT:       ttl.signpost "ttl_store_one"
// IR-NEXT:       ttl.tile_store
// IR-NEXT:       ttl.signpost "ttl_store_one" {is_end}
// IR-NEXT:       ttl.signpost "ttl_store_two"
// IR-NEXT:       ttl.tile_store
// IR-NEXT:       ttl.signpost "ttl_store_two" {is_end}
// IR-NEXT:       ttl.yield
func.func @instrumented_output_stores()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %first_output = ttl.cb_reserve %first_output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %second_output = ttl.cb_reserve %second_output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %inner = ttl.exp %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.add %inner, %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_store_one"
  ttl.store %result, %first_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_store_one" {is_end}
  ttl.signpost "ttl_store_two"
  ttl.store %result, %second_output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_store_two" {is_end}
  return
}

// -----

// Relocating the signpost while leaving the constant outside ttl.compute would
// reverse their observable order. The planner therefore rejects this fusion,
// and the conversion does not form the fused compute.

// CHECK-LABEL: Compute formation plan @intervening_operation
// CHECK:       ttl.add kind=fused recipe=fused legal=false
// CHECK:       rejected=instrumented fusion contains an operation that cannot move into ttl.compute
func.func @intervening_operation()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %accumulator_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
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
  %accumulator_wait = ttl.cb_wait %accumulator_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %accumulator = ttl.attach_cb %accumulator_wait, %accumulator_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_scope"
  %product = ttl.matmul %lhs, %rhs
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %unrelated = arith.constant 0 : i32
  %sum = ttl.add %product, %accumulator
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_scope" {is_end}
  ttl.cb_push %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}
