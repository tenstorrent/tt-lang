// RUN: ttlang-opt %s --convert-ttl-to-ttkernel -split-input-file -verify-diagnostics
// Tests for invalid ttl.tile_* op lowering when dst_idx is missing.

// Unary tile op missing dst_idx attribute.
func.func @tile_exp_missing_dst_idx(%a: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  // expected-error @+1 {{failed to legalize operation 'ttl.tile_exp' that was explicitly marked illegal}}
  %exp = ttl.tile_exp %a : !ttcore.tile<32x32, f32>
  func.return %exp : !ttcore.tile<32x32, f32>
}

// -----

// Binary tile op missing dst_idx attribute.
func.func @tile_add_missing_dst_idx(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  // expected-error @+1 {{failed to legalize operation 'ttl.tile_add' that was explicitly marked illegal}}
  %sum = ttl.tile_add %a, %b : !ttcore.tile<32x32, f32>
  func.return %sum : !ttcore.tile<32x32, f32>
}
