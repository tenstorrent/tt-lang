// Verify that non-f32 broadcasts reject f32 DST mode on every architecture
// and broadcast dimension whose LLK ignores that configuration, including an
// automatic conflict with an f32 SFPU operation.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttl-set-compute-kernel-config)' --split-input-file --verify-diagnostics

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // expected-error @below {{'func.func' op explicit 32-bit destination elements are unsupported by the kernel's tile operations}}
  func.func @wormhole_column(
      %input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      attributes {fp32_dest_acc_en = true} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_attached = ttl.attach_cb %input, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %input_tile = tensor.extract %input_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output_tile = tensor.extract %output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-note @below {{the target does not support 32-bit destination elements for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // expected-error @below {{'func.func' op explicit 32-bit destination elements are unsupported by the kernel's tile operations}}
  func.func @wormhole_row(
      %input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      attributes {fp32_dest_acc_en = true} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_attached = ttl.attach_cb %input, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %input_tile = tensor.extract %input_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output_tile = tensor.extract %output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-note @below {{the target does not support 32-bit destination elements for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 2 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // expected-error @below {{'func.func' op explicit 32-bit destination elements are unsupported by the kernel's tile operations}}
  func.func @wormhole_scalar(
      %input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      attributes {fp32_dest_acc_en = true} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_attached = ttl.attach_cb %input, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %input_tile = tensor.extract %input_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output_tile = tensor.extract %output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-note @below {{the target does not support 32-bit destination elements for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 3 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  // expected-error @below {{'func.func' op explicit 32-bit destination elements are unsupported by the kernel's tile operations}}
  func.func @blackhole_column(
      %input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      attributes {fp32_dest_acc_en = true} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_attached = ttl.attach_cb %input, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %input_tile = tensor.extract %input_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output_tile = tensor.extract %output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-note @below {{the target does not support 32-bit destination elements for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// Verify the default policy reports both requirements when an f32 SFPU
// operation and a non-f32 broadcast require incompatible kernel-wide modes.
module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @automatic_mixed_dst_mode_conflict(
      %f32_input: tensor<1x1x!ttcore.tile<32x32, f32>>,
      %bf16_input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %bf16_output: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %f32_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %bf16_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %f32_attached = ttl.attach_cb %f32_input, %f32_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %bf16_attached = ttl.attach_cb %bf16_input, %bf16_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %f32_tile = tensor.extract %f32_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, f32>>
    %bf16_tile = tensor.extract %bf16_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %bf16_output_tile = tensor.extract %bf16_output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttl.tile_exp' op requires 32-bit destination elements, but no kernel-wide destination width supports all tile operations}}
    %exp = ttl.tile_exp %f32_tile into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // expected-note @below {{the target does not support 32-bit destination elements for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %bf16_tile, %bf16_output_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  // expected-error @below {{'func.func' op explicit 32-bit destination elements are unsupported by the kernel's tile operations}}
  func.func @blackhole_row(
      %input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      attributes {fp32_dest_acc_en = true} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_attached = ttl.attach_cb %input, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %input_tile = tensor.extract %input_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output_tile = tensor.extract %output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-note @below {{the target does not support 32-bit destination elements for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 2 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  // expected-error @below {{'func.func' op explicit 32-bit destination elements are unsupported by the kernel's tile operations}}
  func.func @blackhole_scalar(
      %input: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>>)
      attributes {fp32_dest_acc_en = true} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_attached = ttl.attach_cb %input, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %input_tile = tensor.extract %input_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output_tile = tensor.extract %output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-note @below {{the target does not support 32-bit destination elements for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 3 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// Verify a direct f32 destination copy conflicts with a non-f32 broadcast.
module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @direct_f32_dst_with_bf16_bcast(
      %f32_input: tensor<1x1x!ttcore.tile<32x32, f32>>,
      %bf16_input: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %f32_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %bf16_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %f32_attached = ttl.attach_cb %f32_input, %f32_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %bf16_attached = ttl.attach_cb %bf16_input, %bf16_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %f32_tile = tensor.extract %f32_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, f32>>
    %bf16_tile = tensor.extract %bf16_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %bf16_output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %bf16_output_tile = tensor.extract %bf16_output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>

    ttl.dst_section {
      // expected-error @below {{'ttl.copy_tile' op requires 32-bit destination elements, but no kernel-wide destination width supports all tile operations}}
      %dst, %copied = ttl.copy_tile %f32_tile[%zero, %zero] into dst[%zero]
          : !ttcore.tile<32x32, f32>
            -> !ttl.dst, !ttcore.tile<32x32, f32>
      // expected-note @below {{the target does not support 32-bit destination elements for ttl.tile_bcast with 'bf16' elements}}
      %broadcast = ttl.tile_bcast
          %bf16_tile, %bf16_output_tile 2 : i32 into dst[%zero]
          : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
            -> !ttcore.tile<32x32, bf16>
    }
    return
  }
}
