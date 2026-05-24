// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Summary: Verifier-level rejection cases for ttl.block.broadcast: shape
// rank mismatch, broadcast dim with non-1 input size, non-broadcast dim
// size mismatch, out-of-range dim, duplicate dim, and row-major element
// type. CB-attachment of the input is a pipeline invariant (enforced in
// ttl-convert-ttl-to-compute by the fusion path or the standalone
// LowerBlockBroadcastToCompute pattern), not a structural op invariant:
// ttl-insert-intermediate-dfbs materializes a CB for non-CB-attached
// intermediates (e.g. reduce results feeding broadcast) before lowering.

// Shape size does not match input rank.
func.func @bcast_rank_mismatch(%arg0: tensor<2x1x!ttcore.tile<32x32, f32>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{shape size 3 does not match input rank 2}}
  %r = ttl.block.broadcast %arg0_cb dims = [-1], shape = [2, 1, 4] : tensor<2x1x!ttcore.tile<32x32, f32>> -> tensor<2x1x4x!ttcore.tile<32x32, f32>>
  return
}

// -----

// Broadcast dim has input size != 1.
func.func @bcast_non_unit_broadcast_dim(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{input dim 1 is 2 but must be 1 for broadcast dim 1}}
  %r = ttl.block.broadcast %arg0_cb dims = [-1], shape = [2, 4] : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x4x!ttcore.tile<32x32, f32>>
  return
}

// -----

// Non-broadcast dim does not match shape.
func.func @bcast_non_broadcast_dim_mismatch(%arg0: tensor<2x1x!ttcore.tile<32x32, f32>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{input dim 0 is 2 but must match shape[0] = 4 for non-broadcast dim}}
  %r = ttl.block.broadcast %arg0_cb dims = [-1], shape = [4, 8] : tensor<2x1x!ttcore.tile<32x32, f32>> -> tensor<4x8x!ttcore.tile<32x32, f32>>
  return
}

// -----

// Dim index out of range.
func.func @bcast_dim_out_of_range(%arg0: tensor<2x1x!ttcore.tile<32x32, f32>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{dim 3 is out of range for rank 2}}
  %r = ttl.block.broadcast %arg0_cb dims = [3], shape = [2, 4] : tensor<2x1x!ttcore.tile<32x32, f32>> -> tensor<2x4x!ttcore.tile<32x32, f32>>
  return
}

// -----

// Duplicate dim after normalization.
func.func @bcast_duplicate_dim(%arg0: tensor<1x2x!ttcore.tile<32x32, f32>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{duplicate dim -2}}
  %r = ttl.block.broadcast %arg0_cb dims = [0, -2], shape = [4, 2] : tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<4x2x!ttcore.tile<32x32, f32>>
  return
}

// -----

// Row-major element type (not !ttcore.tile) is rejected.
func.func @bcast_row_major_rejected(%arg0: tensor<2x1xf32>) {
  // expected-error @below {{row-major broadcast is not supported; input element type must be !ttcore.tile}}
  %r = ttl.block.broadcast %arg0 dims = [-1], shape = [2, 4] : tensor<2x1xf32> -> tensor<2x4xf32>
  return
}

// -----

// Zero in shape on a broadcast dim is rejected.
func.func @bcast_zero_shape(%arg0: tensor<2x1x!ttcore.tile<32x32, f32>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{shape[1] = 0 must be positive}}
  %r = ttl.block.broadcast %arg0_cb dims = [-1], shape = [2, 0] : tensor<2x1x!ttcore.tile<32x32, f32>> -> tensor<2x0x!ttcore.tile<32x32, f32>>
  return
}
