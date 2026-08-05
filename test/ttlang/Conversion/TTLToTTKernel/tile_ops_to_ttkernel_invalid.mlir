// RUN: ttlang-opt %s --convert-ttl-to-ttkernel -split-input-file -verify-diagnostics
// Tests for invalid ttl.tile_* op lowering: operand tracing and missing CB for FPU.

// Binary tile op where lhs operand has no dst_idx.
// Uses unrealized_conversion_cast to create a value without a defining operation that has dst_idx.
func.func @tile_mul_lhs_missing_dst_idx(%idx: index) -> !ttcore.tile<32x32, f32> {
  %b = arith.constant dense<1.0> : tensor<32x32xf32>
  %b_tile = builtin.unrealized_conversion_cast %b : tensor<32x32xf32> to !ttcore.tile<32x32, f32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b_with_idx = ttl.tile_exp %b_tile into dst[%c1] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>

  %a = arith.constant dense<2.0> : tensor<32x32xf32>
  %a_tile = builtin.unrealized_conversion_cast %a : tensor<32x32xf32> to !ttcore.tile<32x32, f32>

  // expected-error @+1 {{failed to legalize operation 'ttl.tile_mul' that was explicitly marked illegal}}
  %prod = ttl.tile_mul %a_tile, %b_with_idx into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  func.return %prod : !ttcore.tile<32x32, f32>
}

// -----

// Binary tile op where rhs operand has no dst_idx.
func.func @tile_mul_rhs_missing_dst_idx(%idx: index) -> !ttcore.tile<32x32, f32> {
  %a = arith.constant dense<2.0> : tensor<32x32xf32>
  %a_tile = builtin.unrealized_conversion_cast %a : tensor<32x32xf32> to !ttcore.tile<32x32, f32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a_with_idx = ttl.tile_exp %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>

  %b = arith.constant dense<1.0> : tensor<32x32xf32>
  %b_tile = builtin.unrealized_conversion_cast %b : tensor<32x32xf32> to !ttcore.tile<32x32, f32>

  // expected-error @+1 {{failed to legalize operation 'ttl.tile_mul' that was explicitly marked illegal}}
  %prod = ttl.tile_mul %a_with_idx, %b_tile into dst[%c1] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  func.return %prod : !ttcore.tile<32x32, f32>
}

// -----

// Max tile op where lhs operand has no dst_idx.
func.func @tile_max_lhs_missing_dst_idx(%idx: index) -> !ttcore.tile<32x32, f32> {
  %b = arith.constant dense<1.0> : tensor<32x32xf32>
  %b_tile = builtin.unrealized_conversion_cast %b : tensor<32x32xf32> to !ttcore.tile<32x32, f32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b_with_idx = ttl.tile_exp %b_tile into dst[%c1] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>

  %a = arith.constant dense<2.0> : tensor<32x32xf32>
  %a_tile = builtin.unrealized_conversion_cast %a : tensor<32x32xf32> to !ttcore.tile<32x32, f32>

  // expected-error @+1 {{failed to legalize operation 'ttl.tile_max' that was explicitly marked illegal}}
  %max = ttl.tile_max %a_tile, %b_with_idx into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  func.return %max : !ttcore.tile<32x32, f32>
}

// -----

// Max tile op where rhs operand has no dst_idx.
func.func @tile_max_rhs_missing_dst_idx(%idx: index) -> !ttcore.tile<32x32, f32> {
  %a = arith.constant dense<2.0> : tensor<32x32xf32>
  %a_tile = builtin.unrealized_conversion_cast %a : tensor<32x32xf32> to !ttcore.tile<32x32, f32>
  %c0 = arith.constant 0 : index
  %a_with_idx = ttl.tile_exp %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>

  %b = arith.constant dense<1.0> : tensor<32x32xf32>
  %b_tile = builtin.unrealized_conversion_cast %b : tensor<32x32xf32> to !ttcore.tile<32x32, f32>

  // expected-error @+1 {{failed to legalize operation 'ttl.tile_max' that was explicitly marked illegal}}
  %max = ttl.tile_max %a_with_idx, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  func.return %max : !ttcore.tile<32x32, f32>
}

// -----

// Target validation rejects an integer tile operation without an integer LLK.
func.func @tile_exp_u32(%input: !ttcore.tile<16x32, u32>) -> !ttcore.tile<16x32, u32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_exp' op integer tile type !ttcore.tile<16x32, u32> is not supported by this compute primitive}}
  %result = ttl.tile_exp %input into dst[%c0]
      : !ttcore.tile<16x32, u32> -> !ttcore.tile<16x32, u32>
  func.return %result : !ttcore.tile<16x32, u32>
}

// -----

// Target selection rejects architectures without implemented LLK capabilities.
// expected-error @below {{'builtin.module' op Quasar compute LLK capabilities are not implemented by TT-Lang}}
module attributes {ttl.target_arch = "quasar"} {
  func.func @unsupported_target() {
    func.return
  }
}
