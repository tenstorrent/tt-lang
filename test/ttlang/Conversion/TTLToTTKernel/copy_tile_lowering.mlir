// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-to-loops, ttl-annotate-cb-associations),convert-ttl-to-ttkernel,ttkernel-insert-inits)' | FileCheck %s

// Test: ttl.copy_tile inside ttl.compute lowers to ttkernel.copy_tile_init + ttkernel.copy_tile.
// The lowering traces src back to the attached CB via tensor.extract (post loop-lowering).
// After conversion, attach_cb ops are removed (replaced with their tensor operands).

// CHECK-LABEL: func.func @copy_tile_in_compute
// CHECK:       %[[CB_TTK:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
// CHECK:       ttkernel.cb_reserve_back
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           ttkernel.copy_tile_init(%[[CB_TTK]]) : (!ttkernel.cb<{{.*}}>) -> ()
// CHECK:           ttkernel.copy_tile(%[[CB_TTK]], %{{.*}}, %{{.*}}) : (!ttkernel.cb<{{.*}}>, index, index) -> ()
// CHECK:           ttkernel.pack_tile
// CHECK-NOT:   ttl.copy_tile
// CHECK-NOT:   ttl.tile_store
// CHECK-NOT:   ttl.attach_cb
func.func @copy_tile_in_compute(
    %t_tensor: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %src_idx: index,
    %dst_idx: index) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %t_attached = ttl.attach_cb %t_tensor, %cb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
      -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%t_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%t_attached : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%tile_in: !ttcore.tile<32x32, f32>, %tile_out: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %dst, %dst_tile = ttl.copy_tile %tile_in[%i, %j], %dst_idx
        : !ttcore.tile<32x32, f32>, index -> !ttl.dst, !ttcore.tile<32x32, f32>
    ttl.tile_store %dst_tile, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}
