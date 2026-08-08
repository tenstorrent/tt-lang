// Verifies row-normalization planning at DST capacity boundaries for every
// kernel register configuration exercised by the fused LLK.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s

// Four fp32 half-sync tiles exactly fill the effective DST capacity.
// CHECK-LABEL: ComputeOp creation plan @fp32_half_sync_fits
// CHECK:       ttl.mul kind=fused recipe=row_normalization legal=true
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @fp32_half_sync_fits()
      attributes {dst_full_sync_en = false, fp32_dest_acc_en = true,
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x4x!ttcore.tile<32x32, bf16>>,
          tensor<1x4x!ttcore.tile<32x32, bf16>>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x4x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mean = ttl.mul_unary_const %reduced, 2.441406e-04
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
    %broadcast = ttl.block.broadcast %inverse dims = [0, 1], shape = [1, 4]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %result = ttl.mul %input, %broadcast
        : tensor<1x4x!ttcore.tile<32x32, bf16>>,
          tensor<1x4x!ttcore.tile<32x32, bf16>>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output
        : tensor<1x4x!ttcore.tile<32x32, bf16>>,
          tensor<1x4x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// A fifth fp32 half-sync tile requires materialized lowering.
// CHECK-LABEL: ComputeOp creation plan @fp32_half_sync_exceeds_capacity
// CHECK-NOT:   recipe=row_normalization
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @fp32_half_sync_exceeds_capacity()
      attributes {dst_full_sync_en = false, fp32_dest_acc_en = true,
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

// -----

// Full sync retains the helper's eight-tile limit under bf16 accumulation.
// CHECK-LABEL: ComputeOp creation plan @bf16_full_sync_fits
// CHECK:       ttl.mul kind=fused recipe=row_normalization legal=true
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @bf16_full_sync_fits()
      attributes {dst_full_sync_en = true,
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 8], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x8x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x8x!ttcore.tile<32x32, bf16>>,
          tensor<1x8x!ttcore.tile<32x32, bf16>>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x8x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mean = ttl.mul_unary_const %reduced, 1.220703e-04
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
    %broadcast = ttl.block.broadcast %inverse dims = [0, 1], shape = [1, 8]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %result = ttl.mul %input, %broadcast
        : tensor<1x8x!ttcore.tile<32x32, bf16>>,
          tensor<1x8x!ttcore.tile<32x32, bf16>>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 8], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output
        : tensor<1x8x!ttcore.tile<32x32, bf16>>,
          tensor<1x8x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Full sync restores eight usable tiles under fp32 accumulation.
// CHECK-LABEL: ComputeOp creation plan @fp32_full_sync_fits
// CHECK:       ttl.mul kind=fused recipe=row_normalization legal=true
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @fp32_full_sync_fits()
      attributes {dst_full_sync_en = true, fp32_dest_acc_en = true,
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 8], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x8x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x8x!ttcore.tile<32x32, bf16>>,
          tensor<1x8x!ttcore.tile<32x32, bf16>>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x8x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mean = ttl.mul_unary_const %reduced, 1.220703e-04
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
    %broadcast = ttl.block.broadcast %inverse dims = [0, 1], shape = [1, 8]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %result = ttl.mul %input, %broadcast
        : tensor<1x8x!ttcore.tile<32x32, bf16>>,
          tensor<1x8x!ttcore.tile<32x32, bf16>>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 8], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x8x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output
        : tensor<1x8x!ttcore.tile<32x32, bf16>>,
          tensor<1x8x!ttcore.tile<32x32, bf16>>
    return
  }
}
