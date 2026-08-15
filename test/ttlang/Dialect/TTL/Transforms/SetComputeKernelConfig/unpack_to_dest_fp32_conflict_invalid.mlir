// Verify that fixed execution requirements using one f32 dataflow buffer are
// rejected when they require incompatible unpack modes.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config)' --split-input-file --verify-diagnostics

// SFPU operations consume f32 input through DST, while tile_bcast requires the
// default unpack mode. Neither operation has an alternative execution strategy.
func.func @f32_dfb_used_by_bcast_and_sfpu(
    %input: tensor<1x1x!ttcore.tile<32x32, f32>>) {
  %input_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %input_attached = ttl.attach_cb %input, %input_dfb
      : (tensor<1x1x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %zero = arith.constant 0 : index
  %input_tile = tensor.extract %input_attached[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
  %output_tile = tensor.extract %output[%zero, %zero]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  // expected-note @below {{operand 0 establishes the conflicting unpack mode}}
  %exponential = ttl.tile_exp %input_tile into dst[%zero]
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // expected-error @below {{'ttl.tile_bcast' op dataflow buffer 1 requires incompatible unpack modes in one kernel}}
  %broadcast = ttl.tile_bcast %input_tile, %output_tile 2 : i32 into dst[%zero]
      : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>)
        -> !ttcore.tile<32x32, f32>
  return
}
