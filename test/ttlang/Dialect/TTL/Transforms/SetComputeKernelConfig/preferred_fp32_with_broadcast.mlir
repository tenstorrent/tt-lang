// Verifies that BF16 column broadcast preserves a supported full-fp32
// accumulation preference.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Blackhole BF16 column broadcast supports 32-bit destination elements.
// CHECK-LABEL: func.func @matmul_broadcast
// CHECK-SAME: fp32_dest_acc_en = true
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @matmul_broadcast(
      %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %bias: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %bias_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 3, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lhs_attached = ttl.attach_cb %lhs, %lhs_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs_attached = ttl.attach_cb %rhs, %rhs_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %bias_attached = ttl.attach_cb %bias, %bias_dfb
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
        ins(%lhs_attached, %rhs_attached, %bias_attached
            : tensor<1x1x!ttcore.tile<32x32, bf16>>,
              tensor<1x1x!ttcore.tile<32x32, bf16>>,
              tensor<1x1x!ttcore.tile<32x32, bf16>>)
        outs(%output_attached : tensor<1x1x!ttcore.tile<32x32, bf16>>)
        {indexing_maps = [#map, #map, #map, #map],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%lhs_tile: !ttcore.tile<32x32, bf16>,
         %rhs_tile: !ttcore.tile<32x32, bf16>,
         %bias_tile: !ttcore.tile<32x32, bf16>,
         %output_tile: !ttcore.tile<32x32, bf16>):
      %product = ttl.tile_matmul_block %lhs_tile, %rhs_tile into dst[%c0]
          : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
          -> !ttcore.tile<32x32, bf16>
      %broadcast = ttl.tile_bcast %bias_tile, %output_tile 1 : i32
          into dst[%c0] : (!ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      %sum = ttl.tile_add %product, %broadcast into dst[%c0]
          : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
          -> !ttcore.tile<32x32, bf16>
      %row = ttl.iter_index 0 : index
      %column = ttl.iter_index 1 : index
      ttl.tile_store %sum, %output_view[%row, %column] from dst[%c0]
          : !ttcore.tile<32x32, bf16>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
}
