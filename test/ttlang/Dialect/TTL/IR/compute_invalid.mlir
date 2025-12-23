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
    ttl.yield %sum : !ttcore.tile<32x32, f32>
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
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
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
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
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
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
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
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
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
func.func @attach_cb_rank_mismatch(
    %t: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %cb: !ttl.cb<[1], !ttcore.tile<32x32, f32>, 2>) {
  // expected-error @below {{cb shape rank (1) must match tensor rank (2)}}
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
  %cbout = ttl.bind_cb {cb_index = 0, buffer_factor = 2}
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
      ttl.yield %arg0 : !ttcore.tile<32x32, f32>
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
  // expected-error @below {{requires at least one input for SFPU unpacker configuration}}
  %0 = ttl.compute
      ins()
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>):
    ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----



// Test: More iterator dimensions than any tensor rank (catches malformed IR
// where iteration domain doesn't correspond to any actual tensor).
func.func @compute_iterator_exceeds_tensor_rank(
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
  // expected-error @below {{iterator_types count (3) must match maximum tensor rank (2)}}
  %0 = ttl.compute
      ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel", "parallel"]} {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
      ttl.yield %arg0 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
