// Verifies target-independent SumOfSquares pipeline creation and end-to-end
// retained reduction lowering without an intermediate DFB.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute))' | FileCheck %s --check-prefix=COMPUTE
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-select-compute-pipeline-schedules{enable-fpu-binary-ops=0 fp32-dest-acc-en=disabled dst-full-sync-en=disabled matmul-full-fp32=0 reduce-full-fp32=0},ttl-lower-compute-pipelines,ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 fp32-dest-acc-en=disabled dst-full-sync-en=disabled matmul-full-fp32=0 reduce-full-fp32=0},ttl-assign-dst,ttl-subblock-compute-for-dst,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,func.func(ttkernel-combine-pack-tiles),canonicalize,cse,lower-affine)' -o %t.ttkernel.mlir
// RUN: FileCheck %s --input-file=%t.ttkernel.mlir --check-prefix=TTKERNEL
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// The semantic graph selects a capacity-fitting target schedule. Both target
// operands refer to the same graph input for SumOfSquares.
// PLAN-LABEL: ComputeOp creation plan @sum_of_squares
// PLAN:       ttl.reduce kind=fused recipe=fusion-graph legal=true inputs=1 outputs=1 transactions=1
// PLAN:         graph nodes=3 edges=2 stages=1
// PLAN-DAG:       semantic=elementwise-binary
// PLAN-DAG:       semantic=fill
// PLAN-DAG:       semantic=reduction
// PLAN-DAG:       kind=full-tensor carrier=dst
// PLAN-DAG:       kind=full-scalar carrier=recompute
// PLAN:           target=multiply-full-scalar-reduction inputs=[0, 0] tiles=7
// PLAN:           resources dst=7 acquisitions=1 eliminated-intermediate-dfb-bytes=16384
// PLAN:       order=[C0]

// Initial conversion emits the semantic pipeline without an intermediate DFB.
// COMPUTE-LABEL: func.func @sum_of_squares
// COMPUTE-NOT:   ttl.bind_cb{{.*}}ttl.compiler_allocated
// COMPUTE:       ttl.compute_pipeline
// COMPUTE-SAME:  pipeline_kind = #ttl.compute_pipeline_kind<multiply_full_scalar_reduction>
// COMPUTE:         ttl.compute_stage
// COMPUTE:           ttl.mul
// COMPUTE-NEXT:      ttl.fill
// COMPUTE-NEXT:      ttl.reduce
// COMPUTE-NOT:   ttl.tile_mul_reduce_block

// Target lowering retains all seven products in one DST acquisition and
// publishes only the scalar in slot zero.
// TTKERNEL-LABEL: func.func @sum_of_squares
// TTKERNEL-NOT:   ttl.bind_cb
// TTKERNEL:       ttkernel.init_sfpu(%[[INPUT:[a-zA-Z0-9_]+]], %[[OUTPUT:[a-zA-Z0-9_]+]])
// TTKERNEL-NEXT:  ttkernel.tile_regs_acquire
// TTKERNEL-NEXT:  ttkernel.experimental_mul_reduce_block(%[[INPUT]], %[[INPUT]], %[[OUTPUT]]) num_tiles = 7 scale = 1.000000e+00 dtype = <bf16>
// TTKERNEL-NEXT:  ttkernel.tile_regs_commit
// TTKERNEL-NEXT:  ttkernel.tile_regs_wait
// TTKERNEL-NEXT:  ttkernel.pack_tile(%{{.*}}, %[[OUTPUT]], %{{.*}})
// TTKERNEL-NEXT:  ttkernel.tile_regs_release
// TTKERNEL-NOT:   ttkernel.tile_regs_acquire

// C++ translation contains one acquire, one fused helper call, and one pack.
// CPP-LABEL: void kernel_main()
// CPP:       init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
// CPP-NEXT:  tile_regs_acquire();
// CPP-NEXT:  float [[SCALE:[a-zA-Z0-9_]+]] = 1.000000000e+00f;
// CPP-NEXT:  experimental::multiply_full_scalar_reduction_block<7>(get_compile_time_arg_val(0), get_compile_time_arg_val(0), get_compile_time_arg_val(1), [[SCALE]]);
// CPP-NEXT:  tile_regs_commit();
// CPP-NEXT:  tile_regs_wait();
// CPP-NEXT:  pack_tile<true>({{.*}}, get_compile_time_arg_val(1), {{.*}});
// CPP-NEXT:  tile_regs_release();
// CPP-NOT:   tile_regs_acquire();

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @sum_of_squares()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 7], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 7], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x7x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x7x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 7], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x7x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x7x!ttcore.tile<32x32, bf16>>,
          tensor<1x7x!ttcore.tile<32x32, bf16>>
          -> tensor<1x7x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [-2, -1]
        : (tensor<1x7x!ttcore.tile<32x32, bf16>>,
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
