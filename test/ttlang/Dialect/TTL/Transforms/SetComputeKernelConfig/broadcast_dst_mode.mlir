// Verify that every BF16 broadcast kind supports 32-bit destination
// accumulation on Wormhole and Blackhole.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // CHECK-LABEL: func.func @wormhole_column
  // CHECK: ttl.tile_bcast
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
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // CHECK-LABEL: func.func @wormhole_row
  // CHECK: ttl.tile_bcast
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
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 2 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // CHECK-LABEL: func.func @wormhole_scalar
  // CHECK: ttl.tile_bcast
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
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 3 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  // CHECK-LABEL: func.func @blackhole_column
  // CHECK: ttl.tile_bcast
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
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// BF16 broadcast and F32 SFPU execution share one 32-bit destination mode.
module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  // CHECK-LABEL: func.func @automatic_mixed_dst_mode
  // CHECK: ttl.tile_exp
  // CHECK: ttl.tile_bcast
  func.func @automatic_mixed_dst_mode(
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
    %exp = ttl.tile_exp %f32_tile into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    %broadcast = ttl.tile_bcast %bf16_tile, %bf16_output_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  // CHECK-LABEL: func.func @blackhole_row
  // CHECK: ttl.tile_bcast
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
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 2 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  // CHECK-LABEL: func.func @blackhole_scalar
  // CHECK: ttl.tile_bcast
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
    %broadcast = ttl.tile_bcast %input_tile, %output_tile 3 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
          -> !ttcore.tile<32x32, bf16>
    return
  }
}
