// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Summary: Verifies row-prefix stores reject unsupported tensor geometry,
// data types, and producer ownership.

// Row-prefix stores require a tiled source tensor.
func.func @source_must_be_tiled(%source: tensor<32x32xbf16>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttl.store' op row_prefix requires tiled source and destination tensors}}
  ttl.store %source, %view {row_prefix}
      : tensor<32x32xbf16>, tensor<1x14x!ttcore.tile<1x32, bf16>>
  func.return
}

// -----

// Row-prefix stores require a tiled destination tensor.
func.func @destination_must_be_tiled(
    %source: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 14], bf16, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 14], bf16, 1> -> tensor<1x14xbf16>
  // expected-error @below {{'ttl.store' op row_prefix requires tiled source and destination tensors}}
  ttl.store %source, %view {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x14xbf16>
  func.return
}

// -----

// Row-prefix stores require one complete 32x32 source tile.
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

// Row-prefix stores consume exactly one source tile.
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

// Source and destination data types must match.
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

// Row-prefix packing supports the native BF16 and FP32 packer formats.
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

// A high-level row-prefix store writes only into producer-owned storage.
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

// -----

// The compact destination retains the full source row width.
func.func @destination_width_must_match(
    %source: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x16, bf16>, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 14], !ttcore.tile<1x16, bf16>, 1>
      -> tensor<1x14x!ttcore.tile<1x16, bf16>>
  // expected-error @below {{'ttl.store' op row_prefix destination tile width must equal source width 32, got 16}}
  ttl.store %source, %view {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x14x!ttcore.tile<1x16, bf16>>
  func.return
}

// -----

// A row prefix cannot exceed the complete source tile capacity.
func.func @destination_must_fit_source(
    %source: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 33], !ttcore.tile<1x32, bf16>, 1>
  %view = ttl.cb_reserve %cb
      : <[1, 33], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<1x33x!ttcore.tile<1x32, bf16>>
  // expected-error @below {{'ttl.store' op row_prefix destination must contain between 1 and 1024 scalar elements, got 1056}}
  ttl.store %source, %view {row_prefix}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x33x!ttcore.tile<1x32, bf16>>
  func.return
}

// -----

// A lowered row-prefix tile store requires producer-owned storage.
func.func @tile_store_view_must_be_reserved(
    %tile: !ttcore.tile<32x32, bf16>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 14], !ttcore.tile<1x32, bf16>, 1>
  %view = ttl.cb_wait %cb
      : <[1, 14], !ttcore.tile<1x32, bf16>, 1>
      -> tensor<1x14x!ttcore.tile<1x32, bf16>>
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_store' op row_prefix requires a producer-reserved view}}
  ttl.tile_store %tile, %view[%c0, %c0] from dst[%c0] {row_prefix}
      : !ttcore.tile<32x32, bf16>, tensor<1x14x!ttcore.tile<1x32, bf16>>
  func.return
}

// -----

// An unbacked tensor view cannot prove producer ownership.
func.func @tile_store_unbacked_view(
    %tile: !ttcore.tile<32x32, bf16>,
    %view: tensor<1x14x!ttcore.tile<1x32, bf16>>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{'ttl.tile_store' op row_prefix requires a producer-reserved view}}
  ttl.tile_store %tile, %view[%c0, %c0] from dst[%c0] {row_prefix}
      : !ttcore.tile<32x32, bf16>, tensor<1x14x!ttcore.tile<1x32, bf16>>
  func.return
}
