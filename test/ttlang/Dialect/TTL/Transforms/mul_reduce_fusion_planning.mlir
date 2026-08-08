// Verifies target-independent multiply/full-scalar-reduction recognition and
// target schedule selection at effective DST capacity boundaries.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-select-compute-pipeline-schedules{matmul-full-fp32=0 reduce-full-fp32=0}))' | FileCheck %s --check-prefix=SELECT
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-select-compute-pipeline-schedules{matmul-full-fp32=0 reduce-full-fp32=0},ttl-lower-compute-pipelines,ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute))' | FileCheck %s --check-prefix=MATERIALIZED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-select-compute-pipeline-schedules{matmul-full-fp32=0 reduce-full-fp32=0}))' --remarks-filter-missed='ttl-reduction-fusion' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK

// Four tiles exactly fit fp32 half-sync DST capacity.
// CHECK-LABEL: ComputeOp creation plan @fp32_half_four_tiles
// CHECK:       ttl.reduce kind=fused recipe=fusion-graph legal=true inputs=1
// CHECK:       target=multiply-full-scalar-reduction inputs=[0, 0] tiles=4
// CHECK-SAME:  scale=1.000000e+00
// CHECK:       resources dst=4 acquisitions=1
// SELECT-LABEL: func.func @fp32_half_four_tiles
// SELECT:       ttl.compute_pipeline
// SELECT-SAME:  selected_schedule = #ttl.compute_pipeline_schedule<retained_scalar>
// REMARK-NOT:   Function=fp32_half_four_tiles
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @fp32_half_four_tiles()
      attributes {dst_full_sync_en = false, fp32_dest_acc_en = true,
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

// A target without the compound capability recognizes the semantic graph and
// selects ordinary materialized execution.
// CHECK-LABEL: ComputeOp creation plan @wormhole_materialized
// CHECK:       target=multiply-full-scalar-reduction inputs=[0, 0] tiles=1
// SELECT-LABEL: func.func @wormhole_materialized
// SELECT:       ttl.compute_pipeline
// SELECT-SAME:  selected_schedule = #ttl.compute_pipeline_schedule<materialized>
// MATERIALIZED-LABEL: func.func @wormhole_materialized
// MATERIALIZED:       ttl.bind_cb{{.*}} {ttl.compiler_allocated}
// MATERIALIZED:       ttl.tile_mul
// MATERIALIZED:       ttl.tile_reduce
// MATERIALIZED-NOT:   ttl.compute_pipeline
// MATERIALIZED-NOT:   ttl.tile_mul_reduce_block
// MATERIALIZED:       return
// REMARK:      remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=wormhole_materialized
// REMARK-SAME: Remark="multiply_full_scalar_reduction fusion not selected: the target does not provide the retained-scalar schedule; ordinary materialized lowering remains selected"
module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @wormhole_materialized()
      attributes {dst_full_sync_en = false, fp32_dest_acc_en = false,
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %product = ttl.mul %input, %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %product, %scaler 0 : i32 [0, 1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
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

// An unspecified target uses the intersection of registered capabilities and
// therefore does not assume a target-specific compound schedule.
// CHECK-LABEL: ComputeOp creation plan @unspecified_target_materialized
// CHECK:       target=multiply-full-scalar-reduction inputs=[0, 0] tiles=1
// SELECT-LABEL: func.func @unspecified_target_materialized
// SELECT:       ttl.compute_pipeline
// SELECT-SAME:  selected_schedule = #ttl.compute_pipeline_schedule<materialized>
// REMARK:      remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=unspecified_target_materialized
// REMARK-SAME: Remark="multiply_full_scalar_reduction fusion not selected: the target does not provide the retained-scalar schedule; ordinary materialized lowering remains selected"
module {
  func.func @unspecified_target_materialized()
      attributes {dst_full_sync_en = false, fp32_dest_acc_en = false,
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %wait, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %product = ttl.mul %input, %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %product, %scaler 0 : i32 [0, 1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
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

// Recognition remains target-independent; configuration selects materialized
// execution because a fifth tile exceeds fp32 double-buffered DST capacity.
// CHECK-LABEL: ComputeOp creation plan @fp32_half_five_tiles
// CHECK:       target=multiply-full-scalar-reduction inputs=[0, 0] tiles=5
// CHECK:       resources dst=5 acquisitions=1
// SELECT-LABEL: func.func @fp32_half_five_tiles
// SELECT:       ttl.compute_pipeline
// SELECT-SAME:  selected_schedule = #ttl.compute_pipeline_schedule<materialized>
// REMARK:      remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=fp32_half_five_tiles
// REMARK-SAME: Remark="multiply_full_scalar_reduction fusion not selected: the reduction requires 5 DST slots, but 4 are available; ordinary materialized lowering remains selected"
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @fp32_half_five_tiles()
      attributes {dst_full_sync_en = false, fp32_dest_acc_en = true,
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
// CHECK:       resources dst=8 acquisitions=1
// SELECT-LABEL: func.func @bf16_eight_tiles_distinct_inputs
// SELECT:       ttl.compute_pipeline
// SELECT-SAME:  selected_schedule = #ttl.compute_pipeline_schedule<retained_scalar>
// REMARK-NOT:   Function=bf16_eight_tiles_distinct_inputs
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
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
    %scaler = ttl.fill 1.000000e+00
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

// Nine tiles form a semantic pipeline but exceed the retained helper's limit.
// CHECK-LABEL: ComputeOp creation plan @bf16_full_nine_tiles
// CHECK:       target=multiply-full-scalar-reduction inputs=[0, 0] tiles=9
// CHECK:       resources dst=9 acquisitions=1
// SELECT-LABEL: func.func @bf16_full_nine_tiles
// SELECT:       ttl.compute_pipeline
// SELECT-SAME:  selected_schedule = #ttl.compute_pipeline_schedule<materialized>
// REMARK:      remark: [Missed] ReductionFusion | Category:ttl-reduction-fusion | Function=bf16_full_nine_tiles
// REMARK-SAME: Remark="multiply_full_scalar_reduction fusion not selected: the reduction requires 9 DST slots, but 8 are available; ordinary materialized lowering remains selected"
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
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
