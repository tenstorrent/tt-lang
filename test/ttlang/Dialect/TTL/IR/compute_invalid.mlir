// RUN: ttlang-opt %s -split-input-file -verify-diagnostics
// Negative tests for ttl.compute verifier with tensor-only operands and
// CB associations via ttl.attach_cb.

// Test: Block argument count mismatch
func.func @compute_wrong_arg_count(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbb: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{body block must have 3 arguments (matching inputs + outputs), but got 2}}
  %0 = ttl.compute
      ins(%a_att, %b_att : tensor<2x2x!ttcore.tile<32x32, f32>>,
                           tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Indexing maps count mismatch
func.func @compute_wrong_map_count(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{expected 2 indexing maps but got 1}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Invalid iterator type
func.func @compute_invalid_iterator(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{iterator_types must contain only 'parallel' or 'reduction'}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "sequential"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Indexing map is not a projected permutation
func.func @compute_invalid_map_expr(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{input 0 indexing map must be a projected permutation (unique dims or 0 constants)}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0 + d1, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Broadcast constant on non-1 dimension
func.func @compute_broadcast_dim_not_one(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{input 0 broadcast dim 0 must have size 1}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Reduction dimension referenced in output indexing map.
func.func @compute_reduction_in_output(
    %a: tensor<2x3x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x3x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x3x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x3x!ttcore.tile<32x32, f32>>
  // expected-error @below {{output 0 indexing map cannot reference reduction dimension 1}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x3x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x3x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "reduction"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x3x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x3x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Reduction dimension not referenced by any input map. d1 is marked
// "reduction" but both maps broadcast it (constant 0), so no input traverses
// the reduction iterator.
func.func @compute_unreferenced_reduction(
    %a: tensor<2x1x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{reduction dimension 1 must be referenced by at least one input indexing map}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x1x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>,
                        affine_map<(d0, d1) -> (d0, 0)>],
       iterator_types = ["parallel", "reduction"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x1x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x1x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Missing terminator
func.func @compute_no_terminator(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{body block must have a terminator}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Wrong terminator (not ttl.yield)
func.func @compute_wrong_terminator(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{body block must be terminated with ttl.yield}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    func.return %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Missing CB attachment on input tensor
func.func @compute_missing_input_cb(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // %a has no CB attached
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{input 0 must have a circular buffer attached via `ttl.attach_cb` or `ttl.cb_wait`}}
  %0 = ttl.compute
      ins(%a : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Missing CB attachment on output tensor
func.func @compute_missing_output_cb(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // %init has no CB attached
  // expected-error @below {{output 0 must have a circular buffer attached via `ttl.attach_cb` or `ttl.cb_wait`}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: attach_cb element type mismatch
func.func @attach_cb_elem_mismatch(
    %t: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) {
  // expected-error @below {{tensor element type ('!ttcore.tile<32x32, f32>') must match CB element type ('!ttcore.tile<32x32, bf16>')}}
  %att = ttl.attach_cb %t, %cb
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

// Test: attach_cb rank mismatch
// TODO: Re-enable this check once rank validation is revisited.
// See TODO in TTLOps.cpp AttachCBOp::verify() - rank checking disabled for
// TTNN tensors (4D device shape vs 2D CB shard shape).
func.func @attach_cb_rank_mismatch(
    %t: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cb: !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) {
  // TODO: expected error @below {{cb shape rank (1) must match tensor rank (2)}}
  %att = ttl.attach_cb %t, %cb
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

// Test: Multiple different CBs attached to same tensor
func.func @ambiguous_cb_attachment(
    %t: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cb1: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cb2: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cbout = ttl.bind_cb {cb_index = 0, block_count = 2}
           : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Attach same tensor to two different CBs
  %t1 = ttl.attach_cb %t, %cb1
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %t2 = ttl.attach_cb %t, %cb2
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // expected-error @below {{input 0 must have a circular buffer attached via `ttl.attach_cb` or `ttl.cb_wait`}}
  %0 = ttl.compute
      ins(%t : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
      ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

// Test: No inputs (empty ins)
func.func @compute_no_inputs(
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{body must contain at least one ttl.tile_store}}
  %0 = ttl.compute
      ins()
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Compute body missing ttl.tile_store
func.func @compute_missing_tile_store(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{body must contain at least one ttl.tile_store}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Dynamic input shape
func.func @compute_dynamic_input(
    %a: tensor<?x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<?x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<?x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{input 0 must have a static shape}}
  %0 = ttl.compute
      ins(%a_att : tensor<?x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Dynamic output shape
func.func @compute_dynamic_output(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %out: tensor<?x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<?x2x!ttcore.tile<32x32, f32>> {
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_att = ttl.attach_cb %out, %cbout
      : (tensor<?x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<?x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{output 0 must have a static shape}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%out_att : tensor<?x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    ttl.yield
  } -> tensor<?x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<?x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: More iterator dimensions than any tensor rank (catches malformed IR
// where iteration domain doesn't correspond to any actual tensor).
// Iterator count below max tensor rank (1 < 2).
func.func @compute_iterator_below_tensor_rank(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{iterator_types count (1) must be >= maximum tensor rank (2)}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0) -> (d0, d0)>,
                        affine_map<(d0) -> (d0, d0)>],
       iterator_types = ["parallel"]} {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
      ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Result count does not match output count
func.func @compute_result_count_mismatch(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{expected 1 results (one per output) but got 2}}
  %0, %1 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
  func.return %0, %1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: tile_store view not from cb_reserve inside compute body
func.func @compute_tile_store_view_not_from_reserve(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %view: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %exp = ttl.tile_exp %arg0 : !ttcore.tile<32x32, f32>
    // expected-error @below {{'ttl.tile_store' op view must trace to a dataflow buffer}}
    ttl.tile_store %exp, %view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Block argument type does not match operand element type
func.func @compute_block_arg_type_mismatch(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{block argument 0 type '!ttcore.tile<32x32, bf16>' does not match operand element type '!ttcore.tile<32x32, f32>'}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, bf16>, %arg1: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    ttl.tile_store %arg0, %out_view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: formal output CB has no tile_store in the body (multi-output, one missing)
func.func @compute_output_cb_missing_store(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout0: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout1: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0_att = ttl.attach_cb %init0, %cbout0
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1_att = ttl.attach_cb %init1, %cbout1
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view0 = ttl.cb_reserve %cbout0 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{formal output CB has no tile_store in the body}}
  %0, %1 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init0_att, %init1_att : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                    tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %exp = ttl.tile_exp %arg0 : !ttcore.tile<32x32, f32>
    // Only store to cbout0, missing store to cbout1
    ttl.tile_store %exp, %out_view0[%i, %j] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
  func.return %0, %1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: tile_store targets a CB not in the compute's formal outputs (#396)
func.func @compute_tile_store_cb_not_output(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
    %cb_extra: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %extra_view = ttl.cb_reserve %cb_extra : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    ttl.tile_store %arg0, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{stores to CB that is not a formal output of the compute}}
    ttl.tile_store %arg0, %extra_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
