// RUN: ttlang-opt %s -split-input-file -verify-diagnostics

// Test: Block argument count mismatch
func.func @compute_wrong_arg_count(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{body block must have 3 arguments (matching inputs + outputs), but got 2}}
  %0 = ttl.compute ins(%a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Indexing maps count mismatch
func.func @compute_wrong_map_count(%a: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{expected 2 indexing maps but got 1}}
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Invalid iterator type
func.func @compute_invalid_iterator(%a: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{iterator_types must contain only 'parallel' or 'reduction'}}
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "sequential"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Missing terminator
func.func @compute_no_terminator(%a: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{body block must have a terminator}}
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Wrong terminator (not ttl.yield)
func.func @compute_wrong_terminator(%a: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{body block must be terminated with ttl.yield}}
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    func.return %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: tile_batch_size shape/values are verified (moved from lowering tests).

#map_invalid = affine_map<(d0, d1) -> (d0, d1)>

// expected-error @+3 {{'ttl.compute' op tile_batch_size size (1) must match number of iterator dimensions (2)}}
func.func @compute_invalid_batch_rank(%a: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_invalid, #map_invalid], iterator_types = ["parallel", "parallel"], tile_batch_size = [2]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map_invalid = affine_map<(d0, d1) -> (d0, d1)>

// expected-error @+3 {{'ttl.compute' op tile_batch_size values must be > 0}}
func.func @compute_invalid_batch_value(%a: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map_invalid, #map_invalid], iterator_types = ["parallel", "parallel"], tile_batch_size = [2, 0]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
