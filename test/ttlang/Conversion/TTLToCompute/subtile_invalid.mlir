// RUN: ttlang-opt %s --verify-diagnostics --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))'

// Verifies that compute creation diagnoses storage-valid tile dimensions that
// the current compute LLKs cannot execute.

// Direct compute creation validates the result before modifying the source
// operation.
func.func @direct_unsupported_dimensions(
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
  // expected-error @below {{'ttl.exp' op compute result tile shape 8x32 is not supported by the current compute LLKs; supported shapes are 16x16, 16x32, 32x16, and 32x32}}
  %result = ttl.exp %input
      : tensor<1x1x!ttcore.tile<8x32, bf16>>
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<8x32, bf16>>,
        tensor<1x1x!ttcore.tile<8x32, bf16>>
  func.return
}

// -----

// Passthrough-store conversion also creates a compute operation and therefore
// applies the same LLK dimension restriction.
func.func @passthrough_unsupported_dimensions(
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
  // expected-error @below {{'ttl.store' op cannot lower tensor store to ttl.compute: passthrough store tile shape 4x16 is not supported by the current compute LLKs; supported shapes are 16x16, 16x32, 32x16, and 32x32}}
  ttl.store %input, %output
      : tensor<1x1x!ttcore.tile<4x16, u8>>,
        tensor<1x1x!ttcore.tile<4x16, u8>>
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
  // expected-error @below {{'ttl.exp' op compute result TT-Lang supports BFP compute tiles only with 32x32 dimensions, got 16x32}}
  %result = ttl.exp %input
      : tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
        -> tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
  ttl.store %result, %output
      : tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>,
        tensor<1x1x!ttcore.tile<16x32, bfp_bf8>>
  func.return
}
