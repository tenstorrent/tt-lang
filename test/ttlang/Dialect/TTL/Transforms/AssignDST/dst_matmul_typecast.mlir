// Matmul sources use their unpack formats while an explicit conversion selects FP32 packing.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst),canonicalize,cse)' | FileCheck %s

#identity = affine_map<(row, column) -> (row, column)>

// CHECK-LABEL: func.func @matmul_fp32_pack
// CHECK: ttl.tile_matmul_block {{.*}} into dst[%c0]
// CHECK: ttl.tile_typecast {{.*}} into dst[%c0]
// CHECK: ttl.tile_store
func.func @matmul_fp32_pack(
    %activation: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %weight: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %activation_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %weight_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %activation_block = ttl.attach_cb %activation, %activation_dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %weight_block = ttl.attach_cb %weight, %weight_dfb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_view = ttl.cb_reserve %output_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %output_block = ttl.attach_cb %output_view, %output_dfb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%activation_block, %weight_block : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%output_block : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%activation_tile: !ttcore.tile<32x32, bf16>, %weight_tile: !ttcore.tile<32x32, bf16>, %output_tile: !ttcore.tile<32x32, f32>):
      %row = ttl.iter_index 0 : index
      %column = ttl.iter_index 1 : index
      %zero = arith.constant 0 : index
      %product = ttl.tile_matmul_block %activation_tile, %weight_tile into dst[%zero] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
      %converted = ttl.tile_typecast %product into dst[%zero] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
      ttl.tile_store %converted, %output_view[%row, %column] from dst[%zero] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return
}
