// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: ttl.tile_accumulate rejects an unsupported combiner and requires
// the accumulator, contribution, and result tile types to match.

func.func @unsupported_combiner(
    %a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>)
    -> !ttcore.tile<32x32, bf16> {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{expected accumulation combiner}}
  %r = ttl.tile_accumulate %a, %b mul into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
  return %r : !ttcore.tile<32x32, bf16>
}

// -----

func.func @contribution_type_mismatch(
    %a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, bf16> {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{failed to verify that all of}}
  %r = ttl.tile_accumulate %a, %b add into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
  return %r : !ttcore.tile<32x32, bf16>
}

// -----

func.func @accumulator_result_type_mismatch(
    %a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{failed to verify that all of}}
  %r = ttl.tile_accumulate %a, %b add into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  return %r : !ttcore.tile<32x32, f32>
}
