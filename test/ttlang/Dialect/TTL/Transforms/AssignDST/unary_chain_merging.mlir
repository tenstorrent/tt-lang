// Summary: Test unary operation interval merging - verify live intervals via LLVM_DEBUG.
// RUN: ttlang-opt %s --ttl-assign-dst -debug-only=ttl-assign-dst --split-input-file 2>&1 | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test: Chain of unary ops should all share the same DST register (in-place execution).
// Input pattern:
//   %0 = tile_abs(%in)
//   %1 = tile_exp(%0)
//   %2 = tile_relu(%1)
//   yield %2
//
// Expected live intervals (all merged into one equivalence class):
// - Block arg and all unary results should have the same interval [0, 3]
// - All values get allocated to the same DST register

// Verify Phase 2 merging happens
// CHECK: === Phase 2: Build Live Intervals ===
// CHECK: Phase 2: Merged
// CHECK-SAME: tile_abs
// CHECK: Phase 2: Merged
// CHECK-SAME: tile_exp
// CHECK: Phase 2: Merged
// CHECK-SAME: tile_relu

// Verify merged set interval is computed correctly
// Block args start at -1, so merged interval is [-1, 3]
// CHECK: Merged set interval: [-1, 3] for 4 values

// Verify all values in the chain have the same interval
// CHECK: === Live Intervals ===
// CHECK-DAG: [-1, 3]
// CHECK-DAG: [-1, 3]
// CHECK-DAG: [-1, 3]
// CHECK-DAG: [-1, 3]

// Verify all 4 values are allocated together to DST[0] (only one allocation for merged set)
// CHECK: === Phase 3: Linear Scan Allocation ===
// CHECK: Allocated DST[0]
// CHECK-SAME: (merged set size: 4)
// CHECK-NOT: Allocated DST

func.func @unary_chain_shared_dst(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %abs = ttl.tile_abs %a_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %abs : !ttcore.tile<32x32, f32>
    %relu = ttl.tile_relu %exp : !ttcore.tile<32x32, f32>
    ttl.yield %relu : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test: Binary followed by unary chain - unary merges with binary output.
// Pattern from DST_Allocation.md Example 4:
//   %0 = tile_mul(%in0, %in1)
//   %1 = tile_abs(%0)
//   yield %1
//
// Expected: tile_mul and tile_abs share the same DST (merged interval [0, 2])
// Block args have short intervals [0, 0] since they're only used by tile_mul

// Verify binary→unary merging
// CHECK: === Phase 2: Build Live Intervals ===
// CHECK: Phase 2: Merged
// CHECK-SAME: tile_mul
// CHECK-SAME: tile_abs

// Verify merged set has correct interval
// CHECK: Merged set interval: [0, 2] for 2 values

// Verify live intervals
// CHECK: === Live Intervals ===
// The merged set (tile_mul + tile_abs) should have [0, 2]
// CHECK-DAG: tile_mul{{.*}}: [0, 2]
// CHECK-DAG: tile_abs{{.*}}: [0, 2]
// Block args should have [-1, 0] (start at -1)
// CHECK-DAG: block argument{{.*}}: [-1, 0]
// CHECK-DAG: block argument{{.*}}: [-1, 0]

// Verify allocation: merged set of 2 values allocated to DST[0]
// CHECK: === Phase 3: Linear Scan Allocation ===
// CHECK: Allocated DST[0]
// CHECK-SAME: (merged set size: 1)
// CHECK: Allocated DST[1]
// CHECK-SAME: (merged set size: 1)
// CHECK: Allocated DST[0]
// CHECK-SAME: (merged set size: 2)

func.func @binary_then_unary_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %mul = ttl.tile_mul %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %abs = ttl.tile_abs %mul : !ttcore.tile<32x32, f32>
    ttl.yield %abs : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
