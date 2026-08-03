// Verify that non-f32 broadcasts reject f32 DST mode on every architecture
// and broadcast dimension whose LLK ignores that configuration.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file --verify-diagnostics

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // expected-error @below {{'func.func' op explicit f32 destination accumulation is unsupported by the kernel's tile operations}}
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
    // expected-note @below {{the target does not support f32 DST mode for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // expected-error @below {{'func.func' op explicit f32 destination accumulation is unsupported by the kernel's tile operations}}
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
    // expected-note @below {{the target does not support f32 DST mode for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 2 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // expected-error @below {{'func.func' op explicit f32 destination accumulation is unsupported by the kernel's tile operations}}
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
    // expected-note @below {{the target does not support f32 DST mode for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 3 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  // expected-error @below {{'func.func' op explicit f32 destination accumulation is unsupported by the kernel's tile operations}}
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
    // expected-note @below {{the target does not support f32 DST mode for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  // expected-error @below {{'func.func' op explicit f32 destination accumulation is unsupported by the kernel's tile operations}}
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
    // expected-note @below {{the target does not support f32 DST mode for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 2 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  // expected-error @below {{'func.func' op explicit f32 destination accumulation is unsupported by the kernel's tile operations}}
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
    // expected-note @below {{the target does not support f32 DST mode for ttl.tile_bcast with 'bf16' elements}}
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 3 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}
