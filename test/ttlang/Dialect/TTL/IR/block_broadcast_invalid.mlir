// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Summary: Verifier-level rejection cases for ttl.block.broadcast: shape
// rank mismatch, broadcast dim with non-1 input size, non-broadcast dim
// size mismatch, out-of-range dim, duplicate dim, row-major element type,
// and missing CB attachment on input.

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

// Missing CB attachment on input (validateBlockBroadcastOp in
// ConvertTTLToCompute).
// RUN-VARIANT: this check runs the conversion pass that performs the
// CB-attachment validation, not the op verifier.
// expected-error not surfaced by the op verifier alone; covered by the
// runOnOperation walk in convert-ttl-to-compute. Keep this case in the
// conversion-level invalid file (bcast_lowering_invalid.mlir).
