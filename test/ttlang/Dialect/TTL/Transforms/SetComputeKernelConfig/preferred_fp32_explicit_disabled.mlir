// Verify an explicit 16-bit DST constraint overrides a supported full-fp32
// preference without warning about the requested fallback.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --verify-diagnostics | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @bf16_reduce_explicit_16bit
// CHECK-SAME: fp32_dest_acc_en = false
func.func @bf16_reduce_explicit_16bit(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %scaler: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    attributes {fp32_dest_acc_en = false,
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dst_index = arith.constant 0 : index
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %scaler_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %input_attached = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %scaler_attached = ttl.attach_cb %scaler, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_attached = ttl.attach_cb %output, %output_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %output_view = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute
      ins(%input_attached, %scaler_attached
          : tensor<1x1x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%output_attached : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%input_tile: !ttcore.tile<32x32, bf16>,
         %scaler_tile: !ttcore.tile<32x32, bf16>,
         %output_tile: !ttcore.tile<32x32, bf16>):
      %row = ttl.iter_index 0 : index
      %column = ttl.iter_index 1 : index
      %reduced = ttl.tile_reduce %input_tile, %scaler_tile, %output_tile
          0 : i32 <reduce_dim_col> into dst[%dst_index]
          : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
             !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %reduced, %output_view[%row, %column]
          from dst[%dst_index]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
