// Verifies that recognized multiply-reduction graphs report one typed reason
// when target selection cannot preserve an absorbed result.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute))' --remarks-filter-missed='ttl-reduction-fusion' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK
// QUIET-NOT:  [Missed] ReductionFusion

// The product publication is an additional use that the fixed target schedule
// cannot yet preserve. Ordinary lowering remains selected and reports its
// retained traffic.
// PLAN-LABEL: ComputeOp creation plan @published_product
// PLAN:       rejected-source {{.*}} ttl.reduce
// PLAN-NEXT:    near-match=multiply-full-scalar-reduction fusion not selected: an additional use of an absorbed result cannot be preserved; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=10240; additional-dst-acquisitions=2
// REMARK:     remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=published_product
// REMARK-SAME: Remark="multiply-full-scalar-reduction fusion not selected: an additional use of an absorbed result cannot be preserved; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=10240; additional-dst-acquisitions=2"
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @published_product()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %product_output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %scalar_output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %product = ttl.mul %input, %input
        : tensor<1x4x!ttcore.tile<32x32, bf16>>,
          tensor<1x4x!ttcore.tile<32x32, bf16>>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %product_output = ttl.cb_reserve %product_output_dfb
        : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.store %product, %product_output
        : tensor<1x4x!ttcore.tile<32x32, bf16>>,
          tensor<1x4x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %product, %scaler 0 : i32 [0, 1]
        : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scalar_output = ttl.cb_reserve %scalar_output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %scalar_output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// A multiply after the reduction has a source-visible rounding point. The
// compound helper cannot combine it with reduction scaling under strict
// floating-point semantics.
// PLAN-LABEL: ComputeOp creation plan @post_reduction_scale
// PLAN:       near-match-source {{.*}} ttl.mul_unary_const
// PLAN-NEXT:    near-match=multiply-full-scalar-reduction fusion not selected: the schedule would change strict floating-point semantics; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=10240; additional-dst-acquisitions=2
// REMARK:     remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=post_reduction_scale
// REMARK-SAME: Remark="multiply-full-scalar-reduction fusion not selected: the selected schedule cannot preserve output publication; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=10240; additional-dst-acquisitions=2"
// REMARK:     remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=post_reduction_scale
// REMARK-SAME: Remark="multiply-full-scalar-reduction fusion not selected: the schedule would change strict floating-point semantics; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=10240; additional-dst-acquisitions=2"
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @post_reduction_scale()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %product = ttl.mul %input, %input
        : tensor<1x4x!ttcore.tile<32x32, bf16>>,
          tensor<1x4x!ttcore.tile<32x32, bf16>>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %product, %scaler 0 : i32 [0, 1]
        : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaled = ttl.mul_unary_const %reduced, 5.000000e-01
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %scaled, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Full-scalar hardware reduction consumes its scaler in two stages. The
// compound schedule does not reinterpret a non-unit source scaler.
// PLAN-LABEL: ComputeOp creation plan @nonunit_reduction_scaler
// PLAN:       rejected-source {{.*}} ttl.reduce
// PLAN-NEXT:    near-match=multiply-full-scalar-reduction fusion not selected: the schedule would change strict floating-point semantics; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=10240; additional-dst-acquisitions=2
// REMARK:     remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=nonunit_reduction_scaler
// REMARK-SAME: Remark="multiply-full-scalar-reduction fusion not selected: the schedule would change strict floating-point semantics; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=10240; additional-dst-acquisitions=2"
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @nonunit_reduction_scaler()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %product = ttl.mul %input, %input
        : tensor<1x4x!ttcore.tile<32x32, bf16>>,
          tensor<1x4x!ttcore.tile<32x32, bf16>>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 5.000000e-01
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %product, %scaler 0 : i32 [0, 1]
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
}

// -----

// Observing DST between the product and reduction cannot move into the fixed
// block schedule. The typed reason explains why instrumentation selects the
// ordinary materialized lowering.
// PLAN-LABEL: ComputeOp creation plan @instrumented_product
// PLAN:       ttl.reduce kind=fused recipe=fusion-graph legal=false
// PLAN:         near-match=multiply-full-scalar-reduction fusion not selected: the selected schedule cannot preserve instrumentation order; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=10240; additional-dst-acquisitions=2
// PLAN-NEXT:    rejected=fixed fusion block cannot preserve instrumentation inside the absorbed expression
// REMARK:     remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=instrumented_product
// REMARK-SAME: Remark="multiply-full-scalar-reduction fusion not selected: the selected schedule cannot preserve instrumentation order; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=10240; additional-dst-acquisitions=2"
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @instrumented_product()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %product = ttl.mul %input, %input
        : tensor<1x4x!ttcore.tile<32x32, bf16>>,
          tensor<1x4x!ttcore.tile<32x32, bf16>>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    "ttl.dprint"() {fmt = "after product", mode = "dst"} : () -> ()
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %product, %scaler 0 : i32 [0, 1]
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
}
