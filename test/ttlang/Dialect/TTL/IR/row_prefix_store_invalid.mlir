// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

func.func @source_tile_must_be_full(
    %source: tensor<1x1x!ttcore.tile<16x32, bf16>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttl.store' op row_prefix source must use 32x32 tiles, got 16x32}}
  ttl.store %source, %view {row_prefix}
      : tensor<1x1x!ttcore.tile<16x32, bf16>>, tensor<1x14x!ttcore.tile<1x32, bf16>>
  func.return
}

// -----

func.func @source_must_be_one_tile(
    %source: tensor<1x2x!ttcore.tile<32x32, bf16>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttl.store' op row_prefix source must contain exactly one tile, got 2}}
  ttl.store %source, %view {row_prefix}
      : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x14x!ttcore.tile<1x32, bf16>>
  func.return
}

// -----

func.func @data_types_must_match(
    %source: tensor<1x1x!ttcore.tile<32x32, f32>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttl.store' op row_prefix source and destination data types must match}}
  ttl.store %source, %view {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x14x!ttcore.tile<1x32, bf16>>
  func.return
}

// -----

func.func @unsupported_data_type(
    %source: tensor<1x1x!ttcore.tile<32x32, bfp_bf8>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bfp_bf8>, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 14], !ttcore.tile<1x32, bfp_bf8>, 1>
      -> tensor<1x14x!ttcore.tile<1x32, bfp_bf8>>
  // expected-error @below {{'ttl.store' op row_prefix supports only bf16 and f32 tile data types}}
  ttl.store %source, %view {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x14x!ttcore.tile<1x32, bfp_bf8>>
  func.return
}

// -----

func.func @view_must_be_reserved(
    %source: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %view = ttl.cb_wait %cb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttl.store' op row_prefix requires a ttl.cb_reserve-backed view}}
  ttl.store %source, %view {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x14x!ttcore.tile<1x32, bf16>>
  func.return
}
