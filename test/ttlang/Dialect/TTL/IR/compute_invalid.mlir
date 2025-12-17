// RUN: ttlang-opt %s -split-input-file -verify-diagnostics

// Test: Block argument count mismatch
func.func @compute_wrong_arg_count(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>, %cba: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>, %cbb: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>, %cbout: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{body block must have 3 arguments (matching inputs + outputs), but got 2}}
  %0 = ttl.compute ins(%a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) in_cbs(%cba, %cbb : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) out_cbs(%cbout : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Indexing maps count mismatch
func.func @compute_wrong_map_count(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %cba: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>, %cbout: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{expected 4 indexing maps but got 1}}
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) in_cbs(%cba : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) out_cbs(%cbout : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Invalid iterator type
func.func @compute_invalid_iterator(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %cba: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>, %cbout: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{iterator_types must contain only 'parallel' or 'reduction'}}
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) in_cbs(%cba : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) out_cbs(%cbout : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "sequential"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Missing terminator
func.func @compute_no_terminator(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %cba: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>, %cbout: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{body block must have a terminator}}
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) in_cbs(%cba : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) out_cbs(%cbout : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Wrong terminator (not ttl.yield)
func.func @compute_wrong_terminator(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %cba: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>, %cbout: !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error@+1 {{body block must be terminated with ttl.yield}}
  %0 = ttl.compute ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>) in_cbs(%cba : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>) out_cbs(%cbout : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    func.return %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: CB count must match input count
func.func @compute_cb_count_mismatch(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                    %cb0: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                                    %cb1: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @+1 {{number of input_cbs (2) must match number of inputs (1)}}
  %0 = ttl.compute
      ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>)
      in_cbs(%cb0, %cb1 : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>)
      out_cbs(%cb0 : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>,
       %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

// Test: CB rank mismatch with tensor
func.func @compute_cb_rank_mismatch(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                   %cb0: !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @+1 {{input_cb[0] shape rank must match input tensor rank for compatibility}}
  %0 = ttl.compute
      ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>)
      in_cbs(%cb0 : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>)
      outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>)
      out_cbs(%cb0 : !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>,
       %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: CB element type mismatch
func.func @compute_cb_elem_mismatch(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                    %cb0: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @+1 {{input_cb[0] element type must match input element type}}
  %0 = ttl.compute
      ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>)
      in_cbs(%cb0 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>)
      out_cbs(%cb0 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>,
       %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----
