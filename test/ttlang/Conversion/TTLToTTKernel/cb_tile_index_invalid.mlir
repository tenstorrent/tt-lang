// RUN: ttlang-opt --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file %s

// Tests for computeCBTileIndexFromLoops validation in convert-ttl-to-ttkernel.
// The conversion should fail with clear diagnostics when enclosing loops have
// unexpected structure (dynamic bounds, non-zero lower bounds, dynamic step).
//
// Uses unrealized_conversion_cast to create a TTKernel CB type so that
// getCBFromView succeeds, isolating the loop validation logic.

// -----

// Dynamic upper bound: tile_store inside scf.for with dynamic UB.
// computeCBTileIndexFromLoops rejects this because it expects constant
// loop bounds from tile/subblock loop generation.
// expected-error @below {{'ttl.tile_store' op enclosing loop has dynamic upper bound; expected constant bounds from tile/subblock loops}}
module {
  func.func @tile_store_dynamic_upper_bound(
      %tile: !ttcore.tile<32x32, bf16>,
      %ub: index) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %fake_cb = builtin.unrealized_conversion_cast to !ttkernel.cb<16, !ttcore.tile<32x32, bf16>>
    %view = builtin.unrealized_conversion_cast %fake_cb : !ttkernel.cb<16, !ttcore.tile<32x32, bf16>> to tensor<4x4x!ttcore.tile<32x32, bf16>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c0 to %ub step %c1 {
      ttl.tile_store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}

// -----

// Non-zero lower bound: tile_store inside scf.for with lb != 0.
// computeCBTileIndexFromLoops requires all loops to start at 0.
// expected-error @below {{'ttl.tile_store' op enclosing loop has non-zero lower bound (2); expected lb=0 from tile/subblock loops}}
module {
  func.func @tile_store_nonzero_lower_bound(
      %tile: !ttcore.tile<32x32, bf16>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %fake_cb = builtin.unrealized_conversion_cast to !ttkernel.cb<16, !ttcore.tile<32x32, bf16>>
    %view = builtin.unrealized_conversion_cast %fake_cb : !ttkernel.cb<16, !ttcore.tile<32x32, bf16>> to tensor<4x4x!ttcore.tile<32x32, bf16>>
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c2 to %c16 step %c1 {
      ttl.tile_store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}

// -----

// Dynamic step: tile_store inside scf.for with dynamic step.
// computeCBTileIndexFromLoops requires constant step to classify loops.
// expected-error @below {{'ttl.tile_store' op enclosing loop has dynamic step; expected constant step from tile/subblock loops}}
module {
  func.func @tile_store_dynamic_step(
      %tile: !ttcore.tile<32x32, bf16>,
      %step: index) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %fake_cb = builtin.unrealized_conversion_cast to !ttkernel.cb<16, !ttcore.tile<32x32, bf16>>
    %view = builtin.unrealized_conversion_cast %fake_cb : !ttkernel.cb<16, !ttcore.tile<32x32, bf16>> to tensor<4x4x!ttcore.tile<32x32, bf16>>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %step {
      ttl.tile_store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}

// -----

// Dynamic lower bound: tile_store inside scf.for with dynamic LB.
// computeCBTileIndexFromLoops requires constant lower bounds.
// expected-error @below {{'ttl.tile_store' op enclosing loop has dynamic lower bound; expected constant bounds from tile/subblock loops}}
module {
  func.func @tile_store_dynamic_lower_bound(
      %tile: !ttcore.tile<32x32, bf16>,
      %lb: index) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %fake_cb = builtin.unrealized_conversion_cast to !ttkernel.cb<16, !ttcore.tile<32x32, bf16>>
    %view = builtin.unrealized_conversion_cast %fake_cb : !ttkernel.cb<16, !ttcore.tile<32x32, bf16>> to tensor<4x4x!ttcore.tile<32x32, bf16>>
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %lb to %c16 step %c1 {
      ttl.tile_store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}
