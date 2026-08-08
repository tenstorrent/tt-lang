// Verifies retained row-normalization lowering with one repeated gamma column.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute))' | FileCheck %s --check-prefix=COMPUTE
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-select-compute-pipeline-schedules{enable-fpu-binary-ops=0 fp32-dest-acc-en=disabled dst-full-sync-en=disabled matmul-full-fp32=0 reduce-full-fp32=0},ttl-lower-compute-pipelines,ttl-lower-source-scalar-scopes,ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 fp32-dest-acc-en=disabled dst-full-sync-en=disabled matmul-full-fp32=0 reduce-full-fp32=0},ttl-assign-dst,ttl-subblock-compute-for-dst,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,func.func(ttkernel-combine-pack-tiles),canonicalize,cse,lower-affine)' -o %t.ttkernel.mlir
// RUN: FileCheck %s --input-file=%t.ttkernel.mlir --check-prefix=TTKERNEL
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Recognition retains the 1x1 gamma input as a repeated-column operand.
// PLAN-LABEL: ComputeOp creation plan @row_normalization_repeated_column
// PLAN:       ttl.mul kind=fused recipe=row_normalization legal=true inputs=2 outputs=1 transactions=1
// PLAN:       order=[C0]

// The pipeline accepts distinct 1x3 row and 1x1 gamma input tensors without
// allocating an intermediate dataflow buffer.
// COMPUTE-LABEL: func.func @row_normalization_repeated_column
// COMPUTE-NOT:   ttl.bind_cb{{.*}}ttl.compiler_allocated
// COMPUTE:       ttl.compute_pipeline
// COMPUTE-SAME:  pipeline_kind = #ttl.compute_pipeline_kind<row_normalization>
// COMPUTE:         = ttl.compute_stage
// COMPUTE-SAME:    iterator_types = ["reduction", "reduction"]
// COMPUTE:         = ttl.compute_stage
// COMPUTE-SAME:    iterator_types = ["parallel", "parallel"]
// COMPUTE:         = ttl.compute_stage
// COMPUTE-SAME:    iterator_types = ["parallel", "parallel"]
// COMPUTE:           ttl.block.broadcast
// COMPUTE:           ttl.block.broadcast
// COMPUTE:           ttl.mul

// Target lowering reads gamma tile zero for every output tile and completes
// the row in one DST acquisition.
// TTKERNEL-LABEL: func.func @row_normalization_repeated_column
// TTKERNEL-NOT:   ttl.bind_cb
// TTKERNEL:       ttkernel.tile_regs_acquire
// TTKERNEL-NEXT:  ttkernel.experimental_mul_reduce_block(%[[INPUT:[a-zA-Z0-9_]+]], %[[INPUT]], %{{.*}}) num_tiles = 3
// TTKERNEL:       ttkernel.experimental.binary_dest_reuse_bcast_tiles_init(%[[GAMMA:[a-zA-Z0-9_]+]], <mul>, <col>, <dest_to_srca>)
// TTKERNEL-NEXT:  ttkernel.experimental.binary_dest_reuse_bcast_tiles(%[[GAMMA]], %[[ZERO:[a-zA-Z0-9_]+]], %[[ZERO]], <mul>, <col>, <dest_to_srca>)
// TTKERNEL-NEXT:  ttkernel.experimental.binary_dest_reuse_bcast_tiles(%[[GAMMA]], %[[ZERO]], %{{.*}}, <mul>, <col>, <dest_to_srca>)
// TTKERNEL-NEXT:  ttkernel.experimental.binary_dest_reuse_bcast_tiles(%[[GAMMA]], %[[ZERO]], %{{.*}}, <mul>, <col>, <dest_to_srca>)
// TTKERNEL:       ttkernel.tile_regs_release
// TTKERNEL-NOT:   ttkernel.tile_regs_acquire

// C++ translation retains the same column-broadcast operation and one
// acquisition.
// CPP-LABEL: void kernel_main()
// CPP:       tile_regs_acquire();
// CPP:       experimental::binary_dest_reuse_bcast_tiles_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(get_compile_time_arg_val(1));
// CPP-NEXT:  experimental::binary_dest_reuse_bcast_tiles<EltwiseBinaryType::ELWMUL, BroadcastType::COL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(get_compile_time_arg_val(1), {{.*}}, {{.*}});
// CPP-NEXT:  experimental::binary_dest_reuse_bcast_tiles<EltwiseBinaryType::ELWMUL, BroadcastType::COL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(get_compile_time_arg_val(1), {{.*}}, {{.*}});
// CPP-NEXT:  experimental::binary_dest_reuse_bcast_tiles<EltwiseBinaryType::ELWMUL, BroadcastType::COL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(get_compile_time_arg_val(1), {{.*}}, {{.*}});
// CPP:       tile_regs_release();
// CPP-NOT:   tile_regs_acquire();

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @row_normalization_repeated_column()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 3], !ttcore.tile<16x32, bf16>, 2>
    %gamma_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 3], !ttcore.tile<16x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 3], !ttcore.tile<16x32, bf16>, 2>
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x3x!ttcore.tile<16x32, bf16>>,
           !ttl.cb<[1, 3], !ttcore.tile<16x32, bf16>, 2>)
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
    %gamma_wait = ttl.cb_wait %gamma_dfb
        : <[1, 1], !ttcore.tile<16x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<16x32, bf16>>
    %gamma = ttl.attach_cb %gamma_wait, %gamma_dfb
        : (tensor<1x1x!ttcore.tile<16x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<16x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x3x!ttcore.tile<16x32, bf16>>,
          tensor<1x3x!ttcore.tile<16x32, bf16>>
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<16x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x3x!ttcore.tile<16x32, bf16>>,
           tensor<1x1x!ttcore.tile<16x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<16x32, bf16>>
    %mean_square = ttl.mul_unary_const %reduced, 6.510417e-04
        : tensor<1x1x!ttcore.tile<16x32, bf16>>
          -> tensor<1x1x!ttcore.tile<16x32, bf16>>
    %epsilon = ttl.fill 1.000000e-05
        : tensor<1x1x!ttcore.tile<16x32, bf16>>
    %biased = ttl.add %mean_square, %epsilon
        : tensor<1x1x!ttcore.tile<16x32, bf16>>,
          tensor<1x1x!ttcore.tile<16x32, bf16>>
          -> tensor<1x1x!ttcore.tile<16x32, bf16>>
    %inverse_rms = ttl.rsqrt %biased
        : tensor<1x1x!ttcore.tile<16x32, bf16>>
          -> tensor<1x1x!ttcore.tile<16x32, bf16>>
    %scalar = ttl.block.broadcast %inverse_rms dims = [0, 1], shape = [1, 3]
        : tensor<1x1x!ttcore.tile<16x32, bf16>>
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
    %normalized = ttl.mul %input, %scalar
        : tensor<1x3x!ttcore.tile<16x32, bf16>>,
          tensor<1x3x!ttcore.tile<16x32, bf16>>
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
    %gamma_column = ttl.block.broadcast %gamma dims = [1], shape = [1, 3]
        : tensor<1x1x!ttcore.tile<16x32, bf16>>
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
    %result = ttl.mul %normalized, %gamma_column
        : tensor<1x3x!ttcore.tile<16x32, bf16>>,
          tensor<1x3x!ttcore.tile<16x32, bf16>>
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 3], !ttcore.tile<16x32, bf16>, 2>
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
    ttl.store %result, %output
        : tensor<1x3x!ttcore.tile<16x32, bf16>>,
          tensor<1x3x!ttcore.tile<16x32, bf16>>
    return
  }
}
