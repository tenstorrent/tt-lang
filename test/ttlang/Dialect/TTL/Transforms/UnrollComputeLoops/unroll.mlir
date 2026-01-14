// Summary: Verify loop unrolling with DST index updates and epilogue handling.
// Tests the full pipeline: ttl.compute → scf.for with unroll_factor → unrolled loops.
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst,ttl-lower-to-loops,ttl-unroll-compute-loops))' --split-input-file | FileCheck %s

#map1d = affine_map<(d0) -> (d0)>

// Purpose: 4 tiles, 2 inputs, footprint=3, capacity=8, unroll=2
// Expect: main loop with step=2, no epilogue (4/2 divides evenly)
// CHECK-LABEL: func.func @test_unroll_exact_divisor
func.func @test_unroll_exact_divisor(%a: tensor<4x!ttcore.tile<32x32, f32>>,
                                     %b: tensor<4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>

  // CHECK: [[FOR:%.+]] = scf.for {{.*}} step %c2
  // CHECK: tensor.extract
  // CHECK-NEXT: tensor.extract
  // CHECK-NEXT: [[ZERO:%.+]] = arith.constant 0
  // CHECK-NEXT: ttl.copy_tile {{.*}}, [[ZERO]] {base_dst_idx = 0
  // CHECK-NEXT: [[ONE:%.+]] = arith.constant 1
  // CHECK-NEXT: ttl.copy_tile {{.*}}, [[ONE]] {base_dst_idx = 1
  // CHECK-NEXT: ttl.tile_add {{.*}} {base_dst_idx = 0 : i32, dst_idx = 0
  // CHECK: [[TWO:%.+]] = arith.constant 2
  // CHECK-NEXT: ttl.copy_tile {{.*}}, [[TWO]] {base_dst_idx = 0
  // CHECK-NEXT: [[THREE:%.+]] = arith.constant 3
  // CHECK-NEXT: ttl.copy_tile {{.*}}, [[THREE]] {base_dst_idx = 1
  // CHECK-NEXT: ttl.tile_add {{.*}} {base_dst_idx = 0 : i32, dst_idx = 2
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<4x!ttcore.tile<32x32, f32>>,
                         tensor<4x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map1d, #map1d, #map1d],
       iterator_types = ["parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<4x!ttcore.tile<32x32, f32>>
}

// -----

#map1d = affine_map<(d0) -> (d0)>

// Purpose: 5 tiles, 2 inputs, footprint=3, capacity=8, unroll=2
// Expect: main loop (4 tiles) + epilogue loop (1 tile)
// CHECK-LABEL: func.func @test_unroll_with_remainder
func.func @test_unroll_with_remainder(%a: tensor<5x!ttcore.tile<32x32, f32>>,
                                      %b: tensor<5x!ttcore.tile<32x32, f32>>)
    -> tensor<5x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<5x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<5x!ttcore.tile<32x32, f32>>, !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>) -> tensor<5x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<5x!ttcore.tile<32x32, f32>>, !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>) -> tensor<5x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<5x!ttcore.tile<32x32, f32>>, !ttl.cb<[5], !ttcore.tile<32x32, f32>, 2>) -> tensor<5x!ttcore.tile<32x32, f32>>

  // CHECK: [[FOR:%.+]] = scf.for {{.*}} step [[STEP:%.+]]
  // CHECK: } {ttl.unroll_factor = 2
  // CHECK-NEXT: [[EXTRACT1:%.+]] = tensor.extract
  // CHECK-NEXT: [[EXTRACT2:%.+]] = tensor.extract
  // CHECK-NEXT: [[COPY1:%.+]] = ttl.copy_tile [[EXTRACT1]], {{.*}} [[ZERO:%.+]] {base_dst_idx = 0
  // CHECK-NEXT: [[COPY2:%.+]] = ttl.copy_tile [[EXTRACT2]], {{.*}} [[ONE:%.+]] {base_dst_idx = 1
  // CHECK-NEXT: [[ADD:%.+]] = ttl.tile_add {{.*}} {base_dst_idx = 0 : i32, dst_idx = 0
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<5x!ttcore.tile<32x32, f32>>,
                         tensor<5x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<5x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map1d, #map1d, #map1d],
       iterator_types = ["parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<5x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<5x!ttcore.tile<32x32, f32>>
}

// -----

#map1d = affine_map<(d0) -> (d0)>

// Purpose: 1 tile (no unroll_factor attribute)
// Expect: unchanged loop with step=1
// CHECK-LABEL: func.func @test_no_unroll
func.func @test_no_unroll(%a: tensor<1x!ttcore.tile<32x32, f32>>,
                          %b: tensor<1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x!ttcore.tile<32x32, f32>>

  // CHECK: scf.for {{.*}} step %c1
  // CHECK-NOT: step %c2
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x!ttcore.tile<32x32, f32>>,
                         tensor<1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map1d, #map1d, #map1d],
       iterator_types = ["parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x!ttcore.tile<32x32, f32>>
}
