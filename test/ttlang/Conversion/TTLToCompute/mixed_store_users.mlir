// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),cse,canonicalize)' | FileCheck %s

// Summary: Tests store lowering when a producer has both a direct store user
// and an earlier non-store user. The direct store lowering must not replace
// the producer with a later ttl.compute result unless that replacement
// dominates all remaining uses.

// The max result is consumed by sub before the max store. The max store and
// the fused max+sub store both lower to ttl.compute without violating SSA
// dominance.
// CHECK-LABEL: func.func @mixed_store_and_fused_use
func.func @mixed_store_and_fused_use(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>, %arg1: tensor<1x1x!ttcore.tile<32x32, f32>>, %arg2: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 17, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %arg1_cb = ttl.attach_cb %arg1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %arg2_cb = ttl.attach_cb %arg2, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %max_reserve = ttl.cb_reserve %cb3 : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %sub_reserve = ttl.cb_reserve %cb4 : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // CHECK: %[[MAX_COMPUTE:.*]] = ttl.compute
  // CHECK: ttl.tile_max
  // CHECK: ttl.tile_store
  // CHECK: %[[SUB_COMPUTE:.*]] = ttl.compute
  // CHECK: ttl.tile_max
  // CHECK: ttl.tile_sub
  // CHECK: ttl.tile_store
  // CHECK: return %[[SUB_COMPUTE]]
  %max = ttl.max %arg0_cb, %arg1_cb : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %sub = ttl.sub %arg2_cb, %max : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %max, %max_reserve : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %sub, %sub_reserve : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %sub : tensor<1x1x!ttcore.tile<32x32, f32>>
}
