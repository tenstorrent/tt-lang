// Verifier tests for ttl.tile_store_block op.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// -----

// ntiles exceeds view tensor size.

func.func @ntiles_exceeds_view(%view: tensor<2x2x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  // expected-error @+1 {{ntiles (5) exceeds view tensor size (4)}}
  ttl.tile_store_block %c0, %view, %c5 : index, tensor<2x2x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// ntiles is zero.

func.func @ntiles_zero(%view: tensor<2x2x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{ntiles must be positive, got 0}}
  ttl.tile_store_block %c0, %view, %c0 : index, tensor<2x2x!ttcore.tile<32x32, bf16>>
  func.return
}
