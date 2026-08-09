// Verify that fixed execution requirements using one f32 dataflow buffer are
// rejected when they require incompatible unpack modes.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file --verify-diagnostics

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

// -----

// Purpose: A direct f32 DST recurrence requires kernel-wide
// fp32_dest_acc_en, which is incompatible with the bf16 unary broadcast LLK.
func.func @direct_f32_dst_with_bf16_bcast(
    %f32_input: tensor<1x1x!ttcore.tile<32x32, f32>>,
    %bf16_input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  %f32_tile = tensor.extract %f32_input[%c0, %c0]
      : tensor<1x1x!ttcore.tile<32x32, f32>>
  %bf16_tile = tensor.extract %bf16_input[%c0, %c0]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %bf16_output = tensor.empty()
      : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %bf16_output_tile = tensor.extract %bf16_output[%c0, %c0]
      : tensor<1x1x!ttcore.tile<32x32, bf16>>

  ttl.dst_section {
    %dst, %copied = ttl.copy_tile %f32_tile[%c0, %c0] into dst[%c0]
        : !ttcore.tile<32x32, f32>
          -> !ttl.dst, !ttcore.tile<32x32, f32>
    // expected-error @below {{'ttl.tile_bcast' op cannot share a kernel with a direct f32 DST operation because fp32_dest_acc_en is kernel-wide}}
    %broadcast = ttl.tile_bcast
        %bf16_tile, %bf16_output_tile 2 : i32 into dst[%c0]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
  }
  return
}
