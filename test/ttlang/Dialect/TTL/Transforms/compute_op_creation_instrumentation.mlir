// Tests whether direct and fused ComputeOp creation preserve instrumentation
// ownership and ordering.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute))' | FileCheck %s --check-prefix=IR
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute))' 2>/dev/null | FileCheck %s --check-prefix=SPLIT

// Nested user signposts move into the compute body with the operations they
// observe. A following scope remains outside the compute.

// CHECK-LABEL: ComputeOp creation plan @movable_instrumentation
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

// Two output stores have distinct instrumentation scopes. Creation must keep
// each tile store inside its corresponding scope instead of grouping the tile
// stores between the first begin and final end signposts.

// CHECK-LABEL: ComputeOp creation plan @instrumented_output_stores
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

// Scalar debug output cannot move into ttl.compute. Materializing the tensor
// value consumed after the print preserves the source observation order. The
// unrelated pure constant does not require another split.

// CHECK-LABEL: ComputeOp creation plan @intervening_operation
// CHECK:       ttl.add kind=fused recipe=fused legal=false
// CHECK:       rejected=creating ttl.compute would move instrumentation across a non-reorderable operation
// CHECK:       reason=compute-op-instrumentation-would-be-reordered
// SPLIT-LABEL: func.func @intervening_operation
// SPLIT:       ttl.compute
// SPLIT:         ttl.tile_matmul_block
// SPLIT:       ttl.dprint "between product and sum"
// SPLIT:       ttl.compute
// SPLIT:         ttl.tile_add
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
  "ttl.dprint"() {fmt = "between product and sum", mode = "scalar"}
      : () -> ()
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

// -----

// Folding a multiply into exp scaling must not move scaling past instrumentation
// that observes DST. Keeping separate tile operations preserves the observation
// point after the multiply and before exp.

// CHECK-LABEL: ComputeOp creation plan @intervening_exp_scale_instrumentation
// CHECK:       ttl.exp kind=fused recipe=fused legal=true
// IR-LABEL:    func.func @intervening_exp_scale_instrumentation
// IR:          ttl.compute
// IR:            %[[SCALED:.*]] = ttl.tile_mul_unary_const
// IR-NEXT:       ttl.dprint "after scale"
// IR-NEXT:       ttl.tile_exp %[[SCALED]]
func.func @intervening_exp_scale_instrumentation()
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
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaled = ttl.mul_unary_const %input, 2.000000e+00
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  "ttl.dprint"() {fmt = "after scale", mode = "dst"} : () -> ()
  %result = ttl.exp %scaled
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}

// -----

// A direct producer owns the signposts that surround its source operation.
// A fused consumer may recompute that source, but must not move or duplicate
// the producer's observable events.

// CHECK-LABEL: ComputeOp creation plan @direct_producer_owns_instrumentation
// CHECK:       ttl.exp kind=direct recipe=elementwise legal=true
// CHECK:       ttl.exp kind=fused recipe=fused legal=true
// IR-LABEL:    func.func @direct_producer_owns_instrumentation
// IR:          ttl.compute
// IR:            ttl.signpost "ttl_source"
// IR-NEXT:       ttl.tile_exp
// IR-NEXT:       ttl.signpost "ttl_source" {is_end}
// IR:            ttl.tile_store
// IR:          ttl.compute
// IR-NOT:        ttl.signpost "ttl_source"
// IR:            ttl.tile_exp
// IR-NEXT:       ttl.tile_exp
// IR:          return
func.func @direct_producer_owns_instrumentation()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %published_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_source"
  %published = ttl.exp %input
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.signpost "ttl_source" {is_end}
  %published_view = ttl.cb_reserve %published_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %published, %published_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.exp %published
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_view = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
