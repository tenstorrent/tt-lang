// RUN: ttlang-opt %s --ttl-lower-to-loops --convert-ttl-to-ttkernel | FileCheck %s

// Test: ttl.copy_tile inside ttl.compute lowers to ttkernel.copy_tile_init + ttkernel.copy_tile.
// The lowering traces src back to the attached CB via tensor.extract (post loop-lowering).

// CHECK-LABEL: func.func @copy_tile_in_compute
// CHECK:       ttkernel.copy_tile_init
// CHECK-NEXT:  ttkernel.copy_tile
// CHECK-NOT:   ttl.copy_tile
func.func @copy_tile_in_compute(
    %t_tensor: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %src_idx: index,
    %dst_idx: index) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %t_attached = ttl.attach_cb %t_tensor, %cb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%t_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%t_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%tile_in: !ttcore.tile<32x32, f32>, %tile_out: !ttcore.tile<32x32, f32>):
    %dst = ttl.copy_tile %tile_in, %src_idx, %dst_idx
        : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst
    ttl.yield %tile_out : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}
