// Summary: tile_accumulate with add combiner keeps the accumulator and result
// in one DST slot.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config{enable-fpu-binary-ops=0 matmul-full-fp32=0 reduce-full-fp32=0}, func.func(ttl-assign-dst{dst-capacity=4}), canonicalize, cse)' | FileCheck %s

#map_acc_init = affine_map<(d0, d1) -> (d0)>
#map_acc_contrib = affine_map<(d0, d1) -> (d0, d1)>

// The accumulator block argument is copied into DST[0]. The contribution stays
// dataflow-buffer resident. The accumulate op and final store both use DST[0].
// CHECK-LABEL: func.func @accumulate_add_dst_identity
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: ttl.compute
// CHECK-NEXT: ^bb0(%[[INIT_ARG:.*]]: !ttcore.tile<32x32, f32>, %[[CONTRIB_ARG:.*]]: !ttcore.tile<32x32, f32>, %{{.*}}: !ttcore.tile<32x32, f32>):
// CHECK: %{{.*}}, %[[ACC:.*]] = ttl.copy_tile %[[INIT_ARG]][%{{.*}}] into dst[%[[C0]]]
// CHECK-NOT: ttl.copy_tile %[[CONTRIB_ARG]]
// CHECK: %[[NEXT:.*]] = ttl.tile_accumulate %[[ACC]], %[[CONTRIB_ARG]] add into dst[%[[C0]]]
// CHECK: ttl.tile_store %[[NEXT]], %{{.*}}[%{{.*}}] from dst[%[[C0]]]
func.func @accumulate_add_dst_identity(
    %init_arg: tensor<1x!ttcore.tile<32x32, f32>>,
    %contrib_arg: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x!ttcore.tile<32x32, f32>> {
  %out_init = tensor.empty() : tensor<1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %contrib_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %out_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %init_att = ttl.attach_cb %init_arg, %init_cb : (tensor<1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x!ttcore.tile<32x32, f32>>
  %contrib_att = ttl.attach_cb %contrib_arg, %contrib_cb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %out_att = ttl.attach_cb %out_init, %out_cb : (tensor<1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %out_cb : <[1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%init_att, %contrib_att : tensor<1x!ttcore.tile<32x32, f32>>,
                                     tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%out_att : tensor<1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_acc_init, #map_acc_contrib, #map_acc_init],
       iterator_types = ["parallel", "reduction"]} {
  ^bb0(%init_tile: !ttcore.tile<32x32, f32>,
       %contrib_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %next = ttl.tile_accumulate %init_tile, %contrib_tile add into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %next, %out_view[%i] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<1x!ttcore.tile<32x32, f32>>
}
