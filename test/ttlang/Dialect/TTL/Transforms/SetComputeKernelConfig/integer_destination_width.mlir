// Summary: Verify integer compute types select the required destination width.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: si32 compute values require 32-bit destination registers.
// CHECK-LABEL: func.func @si32_destination
// CHECK-SAME: fp32_dest_acc_en = true
func.func @si32_destination() {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<16x32, si32>, 2>
  %output_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<16x32, si32>, 2>
  %input = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<16x32, si32>, 2> -> tensor<1x1x!ttcore.tile<16x32, si32>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<16x32, si32>>
  %output = ttl.attach_cb %empty, %output_cb : (tensor<1x1x!ttcore.tile<16x32, si32>>, !ttl.cb<[1, 1], !ttcore.tile<16x32, si32>, 2>) -> tensor<1x1x!ttcore.tile<16x32, si32>>
  %output_view = ttl.cb_reserve %output_cb : <[1, 1], !ttcore.tile<16x32, si32>, 2> -> tensor<1x1x!ttcore.tile<16x32, si32>>
  %result = ttl.compute
      ins(%input : tensor<1x1x!ttcore.tile<16x32, si32>>)
      outs(%output : tensor<1x1x!ttcore.tile<16x32, si32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%input_tile: !ttcore.tile<16x32, si32>,
       %output_tile: !ttcore.tile<16x32, si32>):
    %c0 = arith.constant 0 : index
    ttl.tile_store %output_tile, %output_view[%c0, %c0] from dst[%c0] : !ttcore.tile<16x32, si32>, tensor<1x1x!ttcore.tile<16x32, si32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<16x32, si32>>
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: u32 compute values require 32-bit destination registers.
// CHECK-LABEL: func.func @u32_destination
// CHECK-SAME: fp32_dest_acc_en = true
func.func @u32_destination() {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<16x32, u32>, 2>
  %output_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<16x32, u32>, 2>
  %input = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<16x32, u32>, 2> -> tensor<1x1x!ttcore.tile<16x32, u32>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<16x32, u32>>
  %output = ttl.attach_cb %empty, %output_cb : (tensor<1x1x!ttcore.tile<16x32, u32>>, !ttl.cb<[1, 1], !ttcore.tile<16x32, u32>, 2>) -> tensor<1x1x!ttcore.tile<16x32, u32>>
  %output_view = ttl.cb_reserve %output_cb : <[1, 1], !ttcore.tile<16x32, u32>, 2> -> tensor<1x1x!ttcore.tile<16x32, u32>>
  %result = ttl.compute
      ins(%input : tensor<1x1x!ttcore.tile<16x32, u32>>)
      outs(%output : tensor<1x1x!ttcore.tile<16x32, u32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%input_tile: !ttcore.tile<16x32, u32>,
       %output_tile: !ttcore.tile<16x32, u32>):
    %c0 = arith.constant 0 : index
    ttl.tile_store %output_tile, %output_view[%c0, %c0] from dst[%c0] : !ttcore.tile<16x32, u32>, tensor<1x1x!ttcore.tile<16x32, u32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<16x32, u32>>
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: u16 compute values retain 16-bit destination registers.
// CHECK-LABEL: func.func @u16_destination
// CHECK-SAME: fp32_dest_acc_en = false
func.func @u16_destination() {
  %input_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<16x32, u16>, 2>
  %output_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<16x32, u16>, 2>
  %input = ttl.cb_wait %input_cb : <[1, 1], !ttcore.tile<16x32, u16>, 2> -> tensor<1x1x!ttcore.tile<16x32, u16>>
  %empty = tensor.empty() : tensor<1x1x!ttcore.tile<16x32, u16>>
  %output = ttl.attach_cb %empty, %output_cb : (tensor<1x1x!ttcore.tile<16x32, u16>>, !ttl.cb<[1, 1], !ttcore.tile<16x32, u16>, 2>) -> tensor<1x1x!ttcore.tile<16x32, u16>>
  %output_view = ttl.cb_reserve %output_cb : <[1, 1], !ttcore.tile<16x32, u16>, 2> -> tensor<1x1x!ttcore.tile<16x32, u16>>
  %result = ttl.compute
      ins(%input : tensor<1x1x!ttcore.tile<16x32, u16>>)
      outs(%output : tensor<1x1x!ttcore.tile<16x32, u16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%input_tile: !ttcore.tile<16x32, u16>,
       %output_tile: !ttcore.tile<16x32, u16>):
    %c0 = arith.constant 0 : index
    ttl.tile_store %output_tile, %output_view[%c0, %c0] from dst[%c0] : !ttcore.tile<16x32, u16>, tensor<1x1x!ttcore.tile<16x32, u16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<16x32, u16>>
  return
}
