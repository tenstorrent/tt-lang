// Verifies that final kernel configuration preserves a retained
// row-normalization block's exact DST residency requirement.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-select-compute-pipeline-schedules{enable-fpu-binary-ops=0 fp32-dest-acc-en=auto dst-full-sync-en=auto matmul-full-fp32=0 reduce-full-fp32=0},ttl-lower-compute-pipelines,ttl-lower-source-scalar-scopes,ttl-create-producer-compute,ttl-insert-intermediate-dfbs,convert-ttl-to-compute,ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 fp32-dest-acc-en=auto dst-full-sync-en=auto matmul-full-fp32=0 reduce-full-fp32=0},ttl-assign-dst))' | FileCheck %s

// Five fp32 tiles require full DST synchronization. The fixed block carries
// num_tiles after tensor iteration is scalarized, so final configuration does
// not revert to the four-slot half-sync capacity.
// CHECK-LABEL: func.func @fp32_five_tiles
// CHECK-SAME:  dst_full_sync_en = true
// CHECK-SAME:  fp32_dest_acc_en = true
// CHECK:       ttl.tile_row_normalization_block
// CHECK-SAME:  num_tiles = 5
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @fp32_five_tiles()
      attributes {fp32_dest_acc_en = true,
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 5], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 5], !ttcore.tile<32x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 5], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x5x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 5], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x5x!ttcore.tile<32x32, bf16>>,
          tensor<1x5x!ttcore.tile<32x32, bf16>>
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x5x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mean = ttl.mul_unary_const %reduced, 1.953125e-04
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %epsilon = ttl.fill 1.000000e-05
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %biased = ttl.add %mean, %epsilon
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %inverse = ttl.rsqrt %biased
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %broadcast = ttl.block.broadcast %inverse dims = [0, 1], shape = [1, 5]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %result = ttl.mul %input, %broadcast
        : tensor<1x5x!ttcore.tile<32x32, bf16>>,
          tensor<1x5x!ttcore.tile<32x32, bf16>>
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 5], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x5x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output
        : tensor<1x5x!ttcore.tile<32x32, bf16>>,
          tensor<1x5x!ttcore.tile<32x32, bf16>>
    return
  }
}
