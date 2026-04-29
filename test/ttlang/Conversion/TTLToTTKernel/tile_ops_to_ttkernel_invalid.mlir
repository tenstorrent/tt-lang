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

// FPU binary op where operands cannot be traced to CBs.
// The ttl.fpu_binary attribute marks the op for FPU lowering, but the operands
// are plain function arguments with no CB association. The FPU pattern fails
// (cannot find CBs), and the SFPU pattern skips FPU-marked ops, so the op
// becomes illegal.
func.func @fpu_add_no_cb(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>)
    -> !ttcore.tile<32x32, bf16>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{failed to legalize operation 'ttl.tile_add' that was explicitly marked illegal}}
  %sum = ttl.tile_add %a, %b into dst[%c0] {ttl.fpu_binary} : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
  func.return %sum : !ttcore.tile<32x32, bf16>
}
