// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),canonicalize)' | FileCheck %s

// Test ttl.typecast lowers to a ttl.compute with ttl.tile_typecast in the
// body. The compute has block arguments of different element types (input is
// bf16, output is f32) since typecast changes the element data type.

// CHECK-LABEL: func.func @typecast_bf16_to_f32
func.func @typecast_bf16_to_f32(%a: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // CHECK:      %[[RES:.*]] = ttl.compute
  // CHECK-SAME:   ins(%{{.*}} : tensor<2x2x!ttcore.tile<32x32, bf16>>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK:      ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK:        ttl.tile_typecast %[[IN]] into dst[%{{.*}}] {{.*}} : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  // CHECK:        ttl.tile_store
  // CHECK:        ttl.yield
  // CHECK:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.typecast %a_cb
       : (tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
