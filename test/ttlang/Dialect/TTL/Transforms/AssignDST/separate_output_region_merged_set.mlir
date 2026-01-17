// Summary: Merged sets containing yielded values should be allocated in output region.
// This test demonstrates the bug where Phase 3 allocates a merged set to the input
// region when the leader value is not yielded, even though one of its merged partners
// IS yielded. The fix ensures Phase 3 skips merged sets if ANY member is yielded.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{separate-output-region=1}))' -debug-only=ttl-assign-dst 2>&1 | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{separate-output-region=1}))' | FileCheck %s --check-prefix=IR

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test case: Binary operation followed by unary operation.
// The binary result (%mul) is NOT yielded directly, but the unary result (%abs) IS yielded.
// Phase 2 merges %mul and %abs (unary operations share DST with their input).
// With the bug: Phase 3 sees %mul is not yielded, allocates the merged set to input region.
// After fix: Phase 3 sees %abs (merged with %mul) is yielded, skips the merged set.
//            Phase 4 allocates the entire merged set to output region.

// Verify Phase 2 merges tile_mul and tile_abs
// CHECK: === Phase 2: Build Live Intervals ===
// CHECK: Phase 2: Merged
// CHECK-SAME: tile_mul
// CHECK-SAME: tile_abs
// CHECK: Merged set interval: [0, 2] for 2 values

// Verify Phase 3 only allocates block arguments (inputs)
// CHECK: === Phase 3: Linear Scan Allocation ===
// CHECK: Phase 3: Allocated DST[0]
// CHECK-SAME: (merged set size: 1)
// CHECK: Phase 3: Allocated DST[1]
// CHECK-SAME: (merged set size: 1)
// Phase 3 should NOT allocate the merged set (size 2) because one member is yielded
// CHECK-NOT: Phase 3: Allocated DST{{.*}} (merged set size: 2)
// CHECK: Phase 3 footprint: 2 registers

// Verify Phase 4 allocates the merged set to output region (starting at DST[2])
// CHECK: === Phase 4: Linear Scan Allocation ===
// CHECK: Phase 4: Allocated DST[2]
// CHECK-SAME: (merged set size: 2)

// Verify final DST assignment: both merged values get output region index
// CHECK: === Final DST Assignment ===
// CHECK-DAG: tile_mul{{.*}} -> DST[2]
// CHECK-DAG: tile_abs{{.*}} -> DST[2]
// CHECK: Max DST usage: 3 / 8 registers

// Verify IR has correct dst_idx attributes
// IR-LABEL: func.func @binary_unary_merged_output
// IR: ttl.compute
// IR: ttl.tile_mul {{.*}} {dst_idx = 2 : i32}
// IR: ttl.tile_abs {{.*}} {dst_idx = 2 : i32}

func.func @binary_unary_merged_output(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
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
    // Binary operation: creates intermediate value (not yielded)
    %mul = ttl.tile_mul %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    // Unary operation: merges with %mul, result IS yielded
    %abs = ttl.tile_abs %mul : !ttcore.tile<32x32, f32>
    ttl.yield %abs : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
