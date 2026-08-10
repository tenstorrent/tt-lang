// RUN: ttlang-opt %s --verify-diagnostics --split-input-file --ttl-to-ttkernel-pipeline

// Verifies target capability validation for pre-formed compute regions.

#identity = affine_map<(tile_row, tile_column) -> (tile_row, tile_column)>

// Pre-formed passthrough regions reject formats that unpack and pack do not
// preserve on device.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @passthrough_unsupported_u8()
      attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %placeholder = arith.constant -1 : index
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, u8>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, u8>, 2>
    %input_raw = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, u8>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, u8>>
    %input = ttl.attach_cb %input_raw, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, u8>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, u8>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, u8>>
    %reserved = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, u8>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, u8>>
    %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, u8>>
    %output = ttl.attach_cb %empty, %output_dfb
        : (tensor<1x1x!ttcore.tile<32x32, u8>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, u8>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, u8>>
    %computed = ttl.compute
        ins(%input : tensor<1x1x!ttcore.tile<32x32, u8>>)
        outs(%output : tensor<1x1x!ttcore.tile<32x32, u8>>)
        {indexing_maps = [#identity, #identity],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%input_tile: !ttcore.tile<32x32, u8>,
         %output_tile: !ttcore.tile<32x32, u8>):
      %tile_row = ttl.iter_index 0 : index
      %tile_column = ttl.iter_index 1 : index
      // expected-error @below {{'ttl.tile_store' op tile type !ttcore.tile<32x32, u8> is not supported; passthrough supports bf16, f16, f32, BFP, si32, u32, and u16 tiles}}
      ttl.tile_store %input_tile, %reserved[%tile_row, %tile_column]
          from dst[%placeholder] {ttl.dst_placeholder}
          : !ttcore.tile<32x32, u8>, tensor<1x1x!ttcore.tile<32x32, u8>>
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<32x32, u8>>
    ttl.cb_push %output_dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, u8>, 2>
    ttl.cb_pop %input_dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, u8>, 2>
    func.return
  }
}

// -----

#identity = affine_map<(tile_row, tile_column) -> (tile_row, tile_column)>

// Pre-formed passthrough regions retain the 32x32 BFP restriction.
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @passthrough_unsupported_bfp_dimensions()
      attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                  ttl.kernel_thread = #ttkernel.thread<compute>} {
    %placeholder = arith.constant -1 : index
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<4x32, bfp_bf8>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<4x32, bfp_bf8>, 2>
    %input_raw = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<4x32, bfp_bf8>, 2>
          -> tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>
    %input = ttl.attach_cb %input_raw, %input_dfb
        : (tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>,
           !ttl.cb<[1, 1], !ttcore.tile<4x32, bfp_bf8>, 2>)
          -> tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>
    %reserved = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<4x32, bfp_bf8>, 2>
          -> tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>
    %empty = tensor.empty() : tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>
    %output = ttl.attach_cb %empty, %output_dfb
        : (tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>,
           !ttl.cb<[1, 1], !ttcore.tile<4x32, bfp_bf8>, 2>)
          -> tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>
    %computed = ttl.compute
        ins(%input : tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>)
        outs(%output : tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>)
        {indexing_maps = [#identity, #identity],
         iterator_types = ["parallel", "parallel"]} {
    ^bb0(%input_tile: !ttcore.tile<4x32, bfp_bf8>,
         %output_tile: !ttcore.tile<4x32, bfp_bf8>):
      %tile_row = ttl.iter_index 0 : index
      %tile_column = ttl.iter_index 1 : index
      // expected-error @below {{'ttl.tile_store' op BFP tiles require 32x32 dimensions, got 4x32}}
      ttl.tile_store %input_tile, %reserved[%tile_row, %tile_column]
          from dst[%placeholder] {ttl.dst_placeholder}
          : !ttcore.tile<4x32, bfp_bf8>,
            tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>
      ttl.yield
    } -> tensor<1x1x!ttcore.tile<4x32, bfp_bf8>>
    ttl.cb_push %output_dfb
        : !ttl.cb<[1, 1], !ttcore.tile<4x32, bfp_bf8>, 2>
    ttl.cb_pop %input_dfb
        : !ttl.cb<[1, 1], !ttcore.tile<4x32, bfp_bf8>, 2>
    func.return
  }
}
