// Verifies recognition and end-to-end lowering of a one-row normalization
// sequence without compiler-allocated intermediate dataflow buffers.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute))' | FileCheck %s --check-prefix=COMPUTE
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute),ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0},func.func(ttl-assign-dst,ttl-subblock-compute-for-dst,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits,func.func(ttkernel-combine-pack-tiles),canonicalize,cse,lower-affine)' -o %t.ttkernel.mlir
// RUN: FileCheck %s --input-file=%t.ttkernel.mlir --check-prefix=TTKERNEL
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// The complete expression is recorded as one row-normalization recipe.
// PLAN-LABEL: ComputeOp creation plan @row_normalization_full_gamma
// PLAN:       ttl.mul kind=fused recipe=row_normalization legal=true inputs=2 outputs=1 transactions=1
// PLAN:       order=[C0]

// Producer creation preserves the expression until specialized conversion.
// Conversion emits one compute with no intermediate DFB allocation.
// COMPUTE-LABEL: func.func @row_normalization_full_gamma
// COMPUTE-NOT:   ttl.bind_cb{{.*}}ttl.compiler_allocated
// COMPUTE:       ttl.compute
// COMPUTE:         ttl.tile_row_normalization_block
// COMPUTE-NOT:   ttl.compute

// TTKernel lowering initializes the common unpack/pack configuration and uses
// one DST transaction for all output tiles.
// TTKERNEL-LABEL: func.func @row_normalization_full_gamma
// TTKERNEL-NOT:   ttl.bind_cb
// TTKERNEL:       ttkernel.init_sfpu(%[[INPUT:[a-zA-Z0-9_]+]], %[[OUTPUT:[a-zA-Z0-9_]+]])
// TTKERNEL-NEXT:  ttkernel.tile_regs_acquire
// TTKERNEL-NEXT:  ttkernel.experimental_row_normalization_block(%[[INPUT]], %[[GAMMA:[a-zA-Z0-9_]+]], %[[OUTPUT]]) num_tiles = 3
// TTKERNEL-SAME:  has_gamma = true dtype = <bf16>
// TTKERNEL-NEXT:  ttkernel.tile_regs_commit
// TTKERNEL-NEXT:  ttkernel.tile_regs_wait
// TTKERNEL-NEXT:  ttkernel.pack_tile_block(%{{.*}}, %[[OUTPUT]], %{{.*}})
// TTKERNEL-NEXT:  ttkernel.tile_regs_release
// TTKERNEL-NOT:   ttkernel.tile_regs_acquire

// C++ translation retains the single-acquire and block-pack schedule.
// CPP-LABEL: void kernel_main()
// CPP:       init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
// CPP-NEXT:  tile_regs_acquire();
// CPP-NEXT:  experimental::row_normalization_block<3, true, DataFormat::Float16_b>(
// CPP-SAME:  1020331500U,
// CPP-NEXT:  tile_regs_commit();
// CPP-NEXT:  tile_regs_wait();
// CPP-NEXT:  pack_tile_block({{.*}}, get_compile_time_arg_val(2), {{.*}});
// CPP-NEXT:  tile_regs_release();
// CPP-NOT:   tile_regs_acquire();

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @row_normalization_full_gamma()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 3], !ttcore.tile<16x32, bf16>, 2>
    %gamma_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 3], !ttcore.tile<16x32, bf16>, 2>
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
        : <[1, 3], !ttcore.tile<16x32, bf16>, 2>
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
    %gamma = ttl.attach_cb %gamma_wait, %gamma_dfb
        : (tensor<1x3x!ttcore.tile<16x32, bf16>>,
           !ttl.cb<[1, 3], !ttcore.tile<16x32, bf16>, 2>)
          -> tensor<1x3x!ttcore.tile<16x32, bf16>>
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
    %result = ttl.mul %normalized, %gamma
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
