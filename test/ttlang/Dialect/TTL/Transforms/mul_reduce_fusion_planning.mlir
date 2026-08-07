// Verifies multiply/full-scalar-reduction selection at effective DST capacity
// boundaries, with distinct inputs and a non-unit semantic scale.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute))' --remarks-filter-missed='ttl-reduction-fusion' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK
// REMARK-NOT: Function=fp32_half_four_tiles

// Four tiles exactly fit fp32 half-sync DST capacity. A post-reduction scalar
// multiply is represented by the target schedule's semantic scale.
// CHECK-LABEL: ComputeOp creation plan @fp32_half_four_tiles
// CHECK:       ttl.mul_unary_const kind=fused recipe=fusion-graph legal=true inputs=1
// CHECK:       target=multiply-full-scalar-reduction inputs=[0, 0] tiles=4
// CHECK-SAME:  scale=5.000000e-01
// CHECK:       resources dst=4/4 acquisitions=1
module attributes {ttl.target_arch = "blackhole"} {
  func.func @fp32_half_four_tiles()
      attributes {fp32_dest_acc_en = true,
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
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

// A fifth tile exceeds fp32 half-sync DST capacity, so the target schedule is
// not selected.
// CHECK-LABEL: ComputeOp creation plan @fp32_half_five_tiles
// CHECK-NOT:   target=multiply-full-scalar-reduction
// CHECK:       rejected-source {{.*}} ttl.reduce
// CHECK-NEXT:    near-match=multiply-full-scalar-reduction fusion not selected: the reduction requires 5 DST slots, but 4 are available; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=12288; additional-dst-acquisitions=2
// REMARK:      remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=fp32_half_five_tiles
// REMARK-SAME: Remark="multiply-full-scalar-reduction fusion not selected: the reduction requires 5 DST slots, but 4 are available; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=12288; additional-dst-acquisitions=2"
// REMARK-NOT:  Function=bf16_eight_tiles_distinct_inputs
module attributes {ttl.target_arch = "blackhole"} {
  func.func @fp32_half_five_tiles()
      attributes {fp32_dest_acc_en = true,
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 5], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 5], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x5x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 5], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %product = ttl.mul %input, %input
        : tensor<1x5x!ttcore.tile<32x32, bf16>>,
          tensor<1x5x!ttcore.tile<32x32, bf16>>
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %product, %scaler 0 : i32 [0, 1]
        : (tensor<1x5x!ttcore.tile<32x32, bf16>>,
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

// Eight tiles fit bf16 capacity. Distinct operands remain distinct target
// inputs instead of being recognized as a named SumOfSquares sequence.
// CHECK-LABEL: ComputeOp creation plan @bf16_eight_tiles_distinct_inputs
// CHECK:       ttl.reduce kind=fused recipe=fusion-graph legal=true inputs=2
// CHECK:       target=multiply-full-scalar-reduction inputs=[0, 1] tiles=8
// CHECK:       resources dst=8/8 acquisitions=1
module attributes {ttl.target_arch = "blackhole"} {
  func.func @bf16_eight_tiles_distinct_inputs()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>
    %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lhs_wait = ttl.cb_wait %lhs_dfb
        : <[1, 8], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %lhs = ttl.attach_cb %lhs_wait, %lhs_dfb
        : (tensor<1x8x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %rhs_wait = ttl.cb_wait %rhs_dfb
        : <[1, 8], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %rhs = ttl.attach_cb %rhs_wait, %rhs_dfb
        : (tensor<1x8x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %product = ttl.mul %lhs, %rhs
        : tensor<1x8x!ttcore.tile<32x32, bf16>>,
          tensor<1x8x!ttcore.tile<32x32, bf16>>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 2.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %product, %scaler 0 : i32 [0, 1]
        : (tensor<1x8x!ttcore.tile<32x32, bf16>>,
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

// Nine tiles exceed the target helper's upper bound even with full sync.
// CHECK-LABEL: ComputeOp creation plan @bf16_full_nine_tiles
// CHECK-NOT:   target=multiply-full-scalar-reduction
// CHECK:       rejected-source {{.*}} ttl.reduce
// CHECK-NEXT:    near-match=multiply-full-scalar-reduction fusion not selected: the reduction requires 9 DST slots, but 8 are available; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=20480; additional-dst-acquisitions=2
// REMARK:      remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=bf16_full_nine_tiles
// REMARK-SAME: Remark="multiply-full-scalar-reduction fusion not selected: the reduction requires 9 DST slots, but 8 are available; ordinary materialized lowering remains selected; retained-intermediate-dfb-bytes=20480; additional-dst-acquisitions=2"
module attributes {ttl.target_arch = "blackhole"} {
  func.func @bf16_full_nine_tiles()
      attributes {dst_full_sync_en = true,
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 9], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 9], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x9x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x9x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 9], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x9x!ttcore.tile<32x32, bf16>>
    %product = ttl.mul %input, %input
        : tensor<1x9x!ttcore.tile<32x32, bf16>>,
          tensor<1x9x!ttcore.tile<32x32, bf16>>
          -> tensor<1x9x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %product, %scaler 0 : i32 [0, 1]
        : (tensor<1x9x!ttcore.tile<32x32, bf16>>,
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
