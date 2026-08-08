// RUN: ttlang-opt %s --verify-diagnostics --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))'

// Verifies that compute creation diagnoses storage-valid tile dimensions that
// the current compute LLKs cannot execute.

// Direct compute creation validates the result before modifying the source
// operation.
func.func @direct_unsupported_dimensions(
    %argument: tensor<1x1x!ttcore.tile<8x16, bf16>>) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x16, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x16, bf16>, 1>
  %input = ttl.attach_cb %argument, %input_dfb
      : (tensor<1x1x!ttcore.tile<8x16, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<8x16, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<8x16, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<8x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<8x16, bf16>>
  // expected-error @below {{'ttl.exp' op compute result tile shape 8x16 is not supported by the current compute LLKs; supported shapes are 1x32, 2x32, 4x32, 8x32, 16x16, 16x32, 32x16, and 32x32}}
  %result = ttl.exp %input
      : tensor<1x1x!ttcore.tile<8x16, bf16>>
        -> tensor<1x1x!ttcore.tile<8x16, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<8x16, bf16>>,
        tensor<1x1x!ttcore.tile<8x16, bf16>>
  func.return
}

// -----

// Short-height support is restricted to elementwise, fill, and matmul
// primitives whose LLKs have been validated for these dimensions.
func.func @short_height_reduce_unsupported() {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
  %scaler_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
  %input_wait = ttl.cb_wait %input_dfb
      : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  %input = ttl.attach_cb %input_wait, %input_dfb
      : (tensor<1x1x!ttcore.tile<8x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  %scaler_wait = ttl.cb_wait %scaler_dfb
      : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  %scaler = ttl.attach_cb %scaler_wait, %scaler_dfb
      : (tensor<1x1x!ttcore.tile<8x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  // expected-error @below {{'ttl.reduce' op tile shape 8x32 is not supported by this compute primitive; short-height tiles are supported by elementwise, fill, and matmul compute primitives}}
  %result = ttl.reduce %input, %scaler 0 : i32 [0, 1]
      : (tensor<1x1x!ttcore.tile<8x32, bf16>>,
         tensor<1x1x!ttcore.tile<8x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<8x32, bf16>>,
        tensor<1x1x!ttcore.tile<8x32, bf16>>
  func.return
}

// -----

// Short-height transpose is rejected independently of other operations in the
// function.
func.func @short_height_transpose_unsupported(
    %argument: tensor<1x1x!ttcore.tile<8x32, bf16>>) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
  %input = ttl.attach_cb %argument, %input_dfb
      : (tensor<1x1x!ttcore.tile<8x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  // expected-error @below {{'ttl.transpose' op tile shape 8x32 is not supported by this compute primitive; short-height tiles are supported by elementwise, fill, and matmul compute primitives}}
  %result = ttl.transpose %input
      : tensor<1x1x!ttcore.tile<8x32, bf16>>
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<8x32, bf16>>,
        tensor<1x1x!ttcore.tile<8x32, bf16>>
  func.return
}

// -----

// Short-height broadcast is rejected independently of other operations in the
// function.
func.func @short_height_broadcast_unsupported(
    %argument: tensor<1x1x!ttcore.tile<8x32, bf16>>) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<8x32, bf16>, 1>
  %input = ttl.attach_cb %argument, %input_dfb
      : (tensor<1x1x!ttcore.tile<8x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 2], !ttcore.tile<8x32, bf16>, 1>
        -> tensor<1x2x!ttcore.tile<8x32, bf16>>
  // expected-error @below {{'ttl.block.broadcast' op tile shape 8x32 is not supported by this compute primitive; short-height tiles are supported by elementwise, fill, and matmul compute primitives}}
  %result = ttl.block.broadcast %input dims = [1], shape = [1, 2]
      : tensor<1x1x!ttcore.tile<8x32, bf16>>
        -> tensor<1x2x!ttcore.tile<8x32, bf16>>
  ttl.store %result, %output
      : tensor<1x2x!ttcore.tile<8x32, bf16>>,
        tensor<1x2x!ttcore.tile<8x32, bf16>>
  func.return
}

// -----

// A matmul in the same compute plan does not extend typecast's supported tile
// dimensions.
func.func @fused_matmul_typecast_short_height_unsupported(
    %lhs_argument: tensor<1x1x!ttcore.tile<8x32, bf16>>,
    %rhs_argument: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, f32>, 1>
  %lhs = ttl.attach_cb %lhs_argument, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<8x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  %rhs = ttl.attach_cb %rhs_argument, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<8x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<8x32, f32>>
  %matmul = ttl.matmul %lhs, %rhs
      : tensor<1x1x!ttcore.tile<8x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  // expected-error @below {{'ttl.typecast' op tile shape 8x32 is not supported by this compute primitive; short-height tiles are supported by elementwise, fill, and matmul compute primitives}}
  %result = ttl.typecast %matmul
      : (tensor<1x1x!ttcore.tile<8x32, bf16>>)
        -> tensor<1x1x!ttcore.tile<8x32, f32>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<8x32, f32>>,
        tensor<1x1x!ttcore.tile<8x32, f32>>
  func.return
}

// -----

// BFP storage is valid at sub-tile dimensions, but compute creation retains
// the conservative 32x32 restriction.
func.func @direct_unsupported_bfp_dimensions(
    %argument: tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_bf8>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_bf8>, 1>
  %input = ttl.attach_cb %argument, %input_dfb
      : (tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_bf8>, 1>)
        -> tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<16x32, bfp_bf8>, 1>
        -> tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
  // expected-error @below {{'ttl.exp' op compute result BFP compute tiles require 32x32 dimensions, got 16x32}}
  %result = ttl.exp %input
      : tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
        -> tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>,
        tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
  func.return
}

// -----

// Matmul validates operation-specific LLK restrictions before creating a
// compute region.
func.func @matmul_unsupported_lhs_dimensions(
    %lhs_argument: tensor<1x1x!ttcore.tile<16x16, bf16>>,
    %rhs_argument: tensor<1x1x!ttcore.tile<16x32, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x16, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 1>
  %lhs = ttl.attach_cb %lhs_argument, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<16x16, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x16, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<16x16, bf16>>
  %rhs = ttl.attach_cb %rhs_argument, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<16x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<16x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  // expected-error @below {{'ttl.matmul' op matmul lhs tile dimensions 16x16 are not implemented by the current compute LLKs}}
  %result = ttl.matmul %lhs, %rhs
      : tensor<1x1x!ttcore.tile<16x16, bf16>>,
        tensor<1x1x!ttcore.tile<16x32, bf16>>
        -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<16x32, bf16>>,
        tensor<1x1x!ttcore.tile<16x32, bf16>>
  func.return
}

// -----

// Transposed matmul validates the rhs tile restriction before creating a
// compute region.
func.func @matmul_unsupported_transposed_rhs_dimensions(
    %lhs_argument: tensor<1x1x!ttcore.tile<32x16, bf16>>,
    %rhs_argument: tensor<1x1x!ttcore.tile<32x16, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %lhs = ttl.attach_cb %lhs_argument, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x16, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x16, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x16, bf16>>
  %rhs = ttl.attach_cb %rhs_argument, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x16, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x16, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x16, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.matmul' op matmul transpose_rhs is not implemented for 32x16 rhs tiles}}
  %result = ttl.matmul %lhs, %rhs {transpose_rhs}
      : tensor<1x1x!ttcore.tile<32x16, bf16>>,
        tensor<1x1x!ttcore.tile<32x16, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Transposed matmul validates the unsupported 32x32-by-16x32 tile
// configuration before creating a compute region.
func.func @matmul_unsupported_transposed_tile_configuration(
    %lhs_argument: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs_argument: tensor<1x1x!ttcore.tile<16x32, bf16>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, bf16>, 1>
  %lhs = ttl.attach_cb %lhs_argument, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_argument, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<16x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x16, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x16, bf16>>
  // expected-error @below {{'ttl.matmul' op matmul tile dimensions lhs 32x32 and rhs 16x32 do not support transpose_rhs in the current compute LLKs}}
  %result = ttl.matmul %lhs, %rhs {transpose_rhs}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<16x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x16, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x16, bf16>>,
        tensor<1x1x!ttcore.tile<32x16, bf16>>
  func.return
}

// -----

// Integer add rejects storage formats that the integer LLK does not accept.
func.func @integer_add_unsupported_u8(
    %lhs_argument: tensor<1x1x!ttcore.tile<16x32, u8>>,
    %rhs_argument: tensor<1x1x!ttcore.tile<16x32, u8>>) {
  %lhs_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, u8>, 1>
  %rhs_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, u8>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, u8>, 1>
  %lhs = ttl.attach_cb %lhs_argument, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<16x32, u8>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, u8>, 1>)
        -> tensor<1x1x!ttcore.tile<16x32, u8>>
  %rhs = ttl.attach_cb %rhs_argument, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<16x32, u8>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, u8>, 1>)
        -> tensor<1x1x!ttcore.tile<16x32, u8>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<16x32, u8>, 1>
        -> tensor<1x1x!ttcore.tile<16x32, u8>>
  // expected-error @below {{'ttl.add' op integer tile type !ttcore.tile<16x32, u8> is not supported; integer add, subtract, and multiply support si32, u32, and u16 tiles}}
  %result = ttl.add %lhs, %rhs
      : tensor<1x1x!ttcore.tile<16x32, u8>>,
        tensor<1x1x!ttcore.tile<16x32, u8>>
        -> tensor<1x1x!ttcore.tile<16x32, u8>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<16x32, u8>>,
        tensor<1x1x!ttcore.tile<16x32, u8>>
  func.return
}

// -----

// Integer tiles are rejected for primitives without an integer LLK.
func.func @integer_exp_unsupported_u32(
    %argument: tensor<1x1x!ttcore.tile<16x32, u32>>) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, u32>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, u32>, 1>
  %input = ttl.attach_cb %argument, %input_dfb
      : (tensor<1x1x!ttcore.tile<16x32, u32>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, u32>, 1>)
        -> tensor<1x1x!ttcore.tile<16x32, u32>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<16x32, u32>, 1>
        -> tensor<1x1x!ttcore.tile<16x32, u32>>
  // expected-error @below {{'ttl.exp' op integer tile type !ttcore.tile<16x32, u32> is not supported by this compute primitive}}
  %result = ttl.exp %input
      : tensor<1x1x!ttcore.tile<16x32, u32>>
        -> tensor<1x1x!ttcore.tile<16x32, u32>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<16x32, u32>>,
        tensor<1x1x!ttcore.tile<16x32, u32>>
  func.return
}

// -----

// Typecast validation reports the complete target-independent input/result
// relation before compute creation mutates the source operation.
func.func @integer_typecast_unsupported(
    %argument: tensor<1x1x!ttcore.tile<32x32, si32>>) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, si32>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %input = ttl.attach_cb %argument, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, si32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, si32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, si32>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.typecast' op only supports floating-point tile data types, but got input: !ttcore.tile<32x32, si32>, result: !ttcore.tile<32x32, bf16>}}
  %result = ttl.typecast %input
      : (tensor<1x1x!ttcore.tile<32x32, si32>>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Passthrough rejects storage-valid formats that the unpack/pack operation
// does not preserve on device.
func.func @passthrough_unsupported_u8(
    %argument: tensor<1x1x!ttcore.tile<4x16, u8>>) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<4x16, u8>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<4x16, u8>, 1>
  %input = ttl.attach_cb %argument, %input_dfb
      : (tensor<1x1x!ttcore.tile<4x16, u8>>,
         !ttl.cb<[1, 1], !ttcore.tile<4x16, u8>, 1>)
        -> tensor<1x1x!ttcore.tile<4x16, u8>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<4x16, u8>, 1>
        -> tensor<1x1x!ttcore.tile<4x16, u8>>
  // expected-error @below {{'ttl.store' op cannot lower tensor store to ttl.compute: passthrough store tile type !ttcore.tile<4x16, u8> is not supported; passthrough supports bf16, f16, f32, BFP, si32, u32, and u16 tiles}}
  ttl.store %input, %output
      : tensor<1x1x!ttcore.tile<4x16, u8>>,
        tensor<1x1x!ttcore.tile<4x16, u8>>
  func.return
}

// -----

// BFP passthrough retains the 32x32 unpack/pack restriction.
func.func @passthrough_unsupported_bfp_dimensions(
    %argument: tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>) {
  %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_bf8>, 1>
  %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_bf8>, 1>
  %input = ttl.attach_cb %argument, %input_dfb
      : (tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>,
         !ttl.cb<[1, 1], !ttcore.tile<16x32, bfp_bf8>, 1>)
        -> tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<16x32, bfp_bf8>, 1>
        -> tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
  // expected-error @below {{'ttl.store' op cannot lower tensor store to ttl.compute: passthrough store BFP tiles require 32x32 dimensions, got 16x32}}
  ttl.store %input, %output
      : tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>,
        tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
  func.return
}
