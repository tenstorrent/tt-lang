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

// -----

// Binary tile op where lhs operand has no dst_idx.
// Uses unrealized_conversion_cast to create a value without a defining operation that has dst_idx.
func.func @tile_mul_lhs_missing_dst_idx(%idx: index) -> !ttcore.tile<32x32, f32> {
  %b = arith.constant dense<1.0> : tensor<32x32xf32>
  %b_tile = builtin.unrealized_conversion_cast %b : tensor<32x32xf32> to !ttcore.tile<32x32, f32>
  %b_with_idx = ttl.tile_exp %b_tile {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>

  %a = arith.constant dense<2.0> : tensor<32x32xf32>
  %a_tile = builtin.unrealized_conversion_cast %a : tensor<32x32xf32> to !ttcore.tile<32x32, f32>

  // expected-error @+1 {{failed to legalize operation 'ttl.tile_mul' that was explicitly marked illegal}}
  %prod = ttl.tile_mul %a_tile, %b_with_idx {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  func.return %prod : !ttcore.tile<32x32, f32>
}

// -----

// Binary tile op where rhs operand has no dst_idx.
func.func @tile_mul_rhs_missing_dst_idx(%idx: index) -> !ttcore.tile<32x32, f32> {
  %a = arith.constant dense<2.0> : tensor<32x32xf32>
  %a_tile = builtin.unrealized_conversion_cast %a : tensor<32x32xf32> to !ttcore.tile<32x32, f32>
  %a_with_idx = ttl.tile_exp %a_tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>

  %b = arith.constant dense<1.0> : tensor<32x32xf32>
  %b_tile = builtin.unrealized_conversion_cast %b : tensor<32x32xf32> to !ttcore.tile<32x32, f32>

  // expected-error @+1 {{failed to legalize operation 'ttl.tile_mul' that was explicitly marked illegal}}
  %prod = ttl.tile_mul %a_with_idx, %b_tile {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
  func.return %prod : !ttcore.tile<32x32, f32>
}

// -----

// Max tile op where lhs operand has no dst_idx.
func.func @tile_max_lhs_missing_dst_idx(%idx: index) -> !ttcore.tile<32x32, f32> {
  %b = arith.constant dense<1.0> : tensor<32x32xf32>
  %b_tile = builtin.unrealized_conversion_cast %b : tensor<32x32xf32> to !ttcore.tile<32x32, f32>
  %b_with_idx = ttl.tile_exp %b_tile {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>

  %a = arith.constant dense<2.0> : tensor<32x32xf32>
  %a_tile = builtin.unrealized_conversion_cast %a : tensor<32x32xf32> to !ttcore.tile<32x32, f32>

  // expected-error @+1 {{failed to legalize operation 'ttl.tile_max' that was explicitly marked illegal}}
  %max = ttl.tile_max %a_tile, %b_with_idx : !ttcore.tile<32x32, f32>
  func.return %max : !ttcore.tile<32x32, f32>
}

// -----

// Max tile op where rhs operand has no dst_idx.
func.func @tile_max_rhs_missing_dst_idx(%idx: index) -> !ttcore.tile<32x32, f32> {
  %a = arith.constant dense<2.0> : tensor<32x32xf32>
  %a_tile = builtin.unrealized_conversion_cast %a : tensor<32x32xf32> to !ttcore.tile<32x32, f32>
  %a_with_idx = ttl.tile_exp %a_tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>

  %b = arith.constant dense<1.0> : tensor<32x32xf32>
  %b_tile = builtin.unrealized_conversion_cast %b : tensor<32x32xf32> to !ttcore.tile<32x32, f32>

  // expected-error @+1 {{failed to legalize operation 'ttl.tile_max' that was explicitly marked illegal}}
  %max = ttl.tile_max %a_with_idx, %b_tile : !ttcore.tile<32x32, f32>
  func.return %max : !ttcore.tile<32x32, f32>
}
