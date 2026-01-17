// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-lower-to-loops,ttl-unroll-compute-loops))' --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// Test: Unroll 1D loop with ttl.unroll_factor=2
// CHECK-LABEL: @test_unroll
func.func @test_unroll(%a: tensor<4x!ttcore.tile<32x32, f32>>,
                       %b: tensor<4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C2]]
  %result = ttl.compute ins(%a_att, %b_att : tensor<4x!ttcore.tile<32x32, f32>>, tensor<4x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], ttl.unroll_factor = 2 : i32} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<4x!ttcore.tile<32x32, f32>>

  return %result : tensor<4x!ttcore.tile<32x32, f32>>
}
