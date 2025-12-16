// RUN: ttlang-opt --convert-ttl-to-ttkernel %s | FileCheck %s
// Summary: Basic tests for ttl.compute and tile op lowering to TTKernel.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compute_unary_exp
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.tile_regs_wait
// CHECK: ttkernel.tile_regs_release
func.func @compute_unary_exp(%a: tensor<1x1xf32>, %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %result = ttl.compute ins(%a : tensor<1x1xf32>) outs(%init : tensor<1x1xf32>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: f32, %out_tile: f32):
    %exp = ttl.tile_exp %a_tile {dst_idx = 0 : i32} : f32
    ttl.yield %exp : f32
  } -> tensor<1x1xf32>
  func.return %result : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: func.func @compute_binary_add
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.add_binary_tile_init
// CHECK: ttkernel.add_binary_tile
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.tile_regs_wait
// CHECK: ttkernel.tile_regs_release
func.func @compute_binary_add(%a: tensor<1x1xf32>, %b: tensor<1x1xf32>, %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %result = ttl.compute ins(%a, %b : tensor<1x1xf32>, tensor<1x1xf32>) outs(%init : tensor<1x1xf32>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: f32, %b_tile: f32, %out_tile: f32):
    %sum = ttl.tile_add %a_tile, %b_tile {dst_idx = 0 : i32} : f32
    ttl.yield %sum : f32
  } -> tensor<1x1xf32>
  func.return %result : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: func.func @compute_chain
// CHECK: ttkernel.tile_regs_acquire
// CHECK: ttkernel.add_binary_tile_init
// CHECK: ttkernel.add_binary_tile
// CHECK: ttkernel.exp_tile_init
// CHECK: ttkernel.exp_tile
// CHECK: ttkernel.tile_regs_commit
// CHECK: ttkernel.tile_regs_wait
// CHECK: ttkernel.tile_regs_release
func.func @compute_chain(%a: tensor<1x1xf32>, %b: tensor<1x1xf32>, %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %result = ttl.compute ins(%a, %b : tensor<1x1xf32>, tensor<1x1xf32>) outs(%init : tensor<1x1xf32>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: f32, %b_tile: f32, %out_tile: f32):
    %sum = ttl.tile_add %a_tile, %b_tile {dst_idx = 0 : i32} : f32
    %exp = ttl.tile_exp %sum {dst_idx = 0 : i32} : f32
    ttl.yield %exp : f32
  } -> tensor<1x1xf32>
  func.return %result : tensor<1x1xf32>
}
