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
  %prod = ttl.tile_mul %a_tile, %b_with_idx into dst[%c0] {ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
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
  %prod = ttl.tile_mul %a_with_idx, %b_tile into dst[%c1] {ttl.tile_execution_strategy = #ttl.tile_execution_strategy<sfpu>} : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
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

// Integer reduce has no corresponding LLK implementation.
func.func @tile_reduce_u32(
    %input: !ttcore.tile<16x32, u32>,
    %scaler: !ttcore.tile<16x32, u32>,
    %output: !ttcore.tile<16x32, u32>) -> !ttcore.tile<16x32, u32>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_reduce' op integer tile type !ttcore.tile<16x32, u32> is not supported by this compute primitive}}
  %result = ttl.tile_reduce %input, %scaler, %output 0 : i32 <reduce_dim_col>
      into dst[%c0]
      : (!ttcore.tile<16x32, u32>, !ttcore.tile<16x32, u32>,
         !ttcore.tile<16x32, u32>) -> !ttcore.tile<16x32, u32>
  func.return %result : !ttcore.tile<16x32, u32>
}

// -----

// Short-height tile reduction supports only the row dimension.
func.func @tile_reduce_short_height_col(
    %input: !ttcore.tile<8x32, bf16>,
    %scaler: !ttcore.tile<8x32, bf16>,
    %output: !ttcore.tile<8x32, bf16>) -> !ttcore.tile<8x32, bf16>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_reduce' op 8x32 reduction supports only row reduction}}
  %result = ttl.tile_reduce %input, %scaler, %output 0 : i32 <reduce_dim_col>
      into dst[%c0]
      : (!ttcore.tile<8x32, bf16>, !ttcore.tile<8x32, bf16>,
         !ttcore.tile<8x32, bf16>) -> !ttcore.tile<8x32, bf16>
  func.return %result : !ttcore.tile<8x32, bf16>
}

// -----

// Short-height tile reduction requires matching tile types.
func.func @tile_reduce_short_height_mixed_types(
    %input: !ttcore.tile<8x32, f32>,
    %scaler: !ttcore.tile<8x32, bf16>,
    %output: !ttcore.tile<8x32, f32>) -> !ttcore.tile<8x32, f32>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_reduce' op short-height reduction supports only matching 8x32 input, scaler, and result tile types}}
  %result = ttl.tile_reduce %input, %scaler, %output 0 : i32 <reduce_dim_row>
      into dst[%c0]
      : (!ttcore.tile<8x32, f32>, !ttcore.tile<8x32, bf16>,
         !ttcore.tile<8x32, f32>) -> !ttcore.tile<8x32, f32>
  func.return %result : !ttcore.tile<8x32, f32>
}

// -----

// Short-height tile reduction supports only the validated 8x32 geometry.
func.func @tile_reduce_short_height_shape(
    %input: !ttcore.tile<4x32, bf16>,
    %scaler: !ttcore.tile<4x32, bf16>,
    %output: !ttcore.tile<4x32, bf16>) -> !ttcore.tile<4x32, bf16>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_reduce' op short-height reduction supports only matching 8x32 input, scaler, and result tile types}}
  %result = ttl.tile_reduce %input, %scaler, %output 0 : i32 <reduce_dim_row>
      into dst[%c0]
      : (!ttcore.tile<4x32, bf16>, !ttcore.tile<4x32, bf16>,
         !ttcore.tile<4x32, bf16>) -> !ttcore.tile<4x32, bf16>
  func.return %result : !ttcore.tile<4x32, bf16>
}

// -----

// Short-height tile reduction supports only BF16 and FP32 data.
func.func @tile_reduce_short_height_f16(
    %input: !ttcore.tile<8x32, f16>,
    %scaler: !ttcore.tile<8x32, f16>,
    %output: !ttcore.tile<8x32, f16>) -> !ttcore.tile<8x32, f16>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_reduce' op 8x32 reduction supports only bf16 and f32 tiles}}
  %result = ttl.tile_reduce %input, %scaler, %output 0 : i32 <reduce_dim_row>
      into dst[%c0]
      : (!ttcore.tile<8x32, f16>, !ttcore.tile<8x32, f16>,
         !ttcore.tile<8x32, f16>) -> !ttcore.tile<8x32, f16>
  func.return %result : !ttcore.tile<8x32, f16>
}

// -----

// Integer transpose has no corresponding LLK implementation.
func.func @tile_transpose_u32(
    %input: !ttcore.tile<16x16, u32>,
    %output: !ttcore.tile<16x16, u32>) -> !ttcore.tile<16x16, u32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_transpose' op integer tile type !ttcore.tile<16x16, u32> is not supported by this compute primitive}}
  %result = ttl.tile_transpose %input, %output into dst[%c0]
      : (!ttcore.tile<16x16, u32>, !ttcore.tile<16x16, u32>)
        -> !ttcore.tile<16x16, u32>
  func.return %result : !ttcore.tile<16x16, u32>
}

// -----

// Integer matmul has no corresponding LLK implementation.
func.func @tile_matmul_u32(
    %lhs: !ttcore.tile<16x32, u32>,
    %rhs: !ttcore.tile<32x32, u32>) -> !ttcore.tile<16x32, u32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_matmul_block' op integer tile type !ttcore.tile<16x32, u32> is not supported by this compute primitive}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0]
      : !ttcore.tile<16x32, u32>, !ttcore.tile<32x32, u32>
        -> !ttcore.tile<16x32, u32>
  func.return %result : !ttcore.tile<16x32, u32>
}

// -----

// Integer broadcast rejects storage formats not supported by the LLK.
func.func @tile_bcast_u8(
    %input: !ttcore.tile<16x32, u8>,
    %output: !ttcore.tile<16x32, u8>) -> !ttcore.tile<16x32, u8> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_bcast' op integer tile type !ttcore.tile<16x32, u8> is not supported; integer broadcast supports si32, u32, and u16 tiles}}
  %result = ttl.tile_bcast %input, %output 2 : i32 into dst[%c0]
      : (!ttcore.tile<16x32, u8>, !ttcore.tile<16x32, u8>)
        -> !ttcore.tile<16x32, u8>
  func.return %result : !ttcore.tile<16x32, u8>
}

// -----

// Short-height tile broadcast supports only the column orientation.
func.func @tile_bcast_short_height_row(
    %input: !ttcore.tile<8x32, f32>,
    %output: !ttcore.tile<8x32, f32>) -> !ttcore.tile<8x32, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_bcast' op 8x32 broadcast supports only column broadcast}}
  %result = ttl.tile_bcast %input, %output 2 : i32 into dst[%c0]
      : (!ttcore.tile<8x32, f32>, !ttcore.tile<8x32, f32>)
        -> !ttcore.tile<8x32, f32>
  func.return %result : !ttcore.tile<8x32, f32>
}

// -----

// Short-height tile broadcast supports only the validated 8x32 geometry.
func.func @tile_bcast_short_height_shape(
    %input: !ttcore.tile<4x32, f32>,
    %output: !ttcore.tile<4x32, f32>) -> !ttcore.tile<4x32, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_bcast' op short-height broadcast supports only matching 8x32 input and result tile types}}
  %result = ttl.tile_bcast %input, %output 1 : i32 into dst[%c0]
      : (!ttcore.tile<4x32, f32>, !ttcore.tile<4x32, f32>)
        -> !ttcore.tile<4x32, f32>
  func.return %result : !ttcore.tile<4x32, f32>
}

// -----

// Short-height tile broadcast supports only BF16 and FP32 data.
func.func @tile_bcast_short_height_f16(
    %input: !ttcore.tile<8x32, f16>,
    %output: !ttcore.tile<8x32, f16>) -> !ttcore.tile<8x32, f16> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_bcast' op 8x32 broadcast supports only bf16 and f32 tiles}}
  %result = ttl.tile_bcast %input, %output 1 : i32 into dst[%c0]
      : (!ttcore.tile<8x32, f16>, !ttcore.tile<8x32, f16>)
        -> !ttcore.tile<8x32, f16>
  func.return %result : !ttcore.tile<8x32, f16>
}

// -----

// Target selection rejects architectures without implemented LLK capabilities.
// expected-error @below {{'builtin.module' op Quasar compute LLK capabilities are not implemented by TT-Lang}}
module attributes {ttl.target_arch = #ttcore.arch<quasar>} {
  func.func @unsupported_target() {
    func.return
  }
}
