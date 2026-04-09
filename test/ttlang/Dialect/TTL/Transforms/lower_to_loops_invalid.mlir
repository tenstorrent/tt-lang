// Invalid tests for ttl-lower-to-loops dst_section validation.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(func.func(ttl-lower-to-loops))'

#map = affine_map<(d0, d1) -> (d0, d1)>

// ----
// Purpose: A tile stored twice with conflicting dst_index assignments.
func.func @tile_store_conflicting_dst() -> (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %in = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %in_att = ttl.attach_cb %in, %cb0
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init0_att = ttl.attach_cb %init0, %cb1
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init1_att = ttl.attach_cb %init1, %cb2
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %view0 = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %view1 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %result:2 = ttl.compute
      ins(%in_att : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%init0_att, %init1_att : tensor<2x2x!ttcore.tile<32x32, bf16>>,
                                     tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"],
       ttl.full_linearization_strides = array<i64: 2, 1>} {
  ^bb0(%in_tile: !ttcore.tile<32x32, bf16>,
       %out0: !ttcore.tile<32x32, bf16>,
       %out1: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %t0 = ttl.tile_add %in_tile, %in_tile into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %t0, %view0[%i, %j] into dst[%c0]
        : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{tile stored with conflicting dst_index assignments}}
    ttl.tile_store %t0, %view1[%i, %j] into dst[%c1]
        : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> (tensor<2x2x!ttcore.tile<32x32, bf16>>,
        tensor<2x2x!ttcore.tile<32x32, bf16>>)

  func.return %result#0, %result#1
      : tensor<2x2x!ttcore.tile<32x32, bf16>>,
        tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// ----
// Purpose: Two different tiles claim the same dst_index.
func.func @dst_index_reused_by_other_tile() -> (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %in = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %in_att = ttl.attach_cb %in, %cb0
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init0_att = ttl.attach_cb %init0, %cb1
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %init1_att = ttl.attach_cb %init1, %cb2
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %view0 = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %view1 = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %result:2 = ttl.compute
      ins(%in_att : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%init0_att, %init1_att : tensor<2x2x!ttcore.tile<32x32, bf16>>,
                                     tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"],
       ttl.full_linearization_strides = array<i64: 2, 1>} {
  ^bb0(%in_tile: !ttcore.tile<32x32, bf16>,
       %out0: !ttcore.tile<32x32, bf16>,
       %out1: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %t0 = ttl.tile_add %in_tile, %in_tile into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<32x32, bf16>
    %t1 = ttl.tile_exp %in_tile into dst[%c0]
        : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %t0, %view0[%i, %j] into dst[%c0]
        : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{dst_index 0 already used by a different tile}}
    ttl.tile_store %t1, %view1[%i, %j] into dst[%c0]
        : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> (tensor<2x2x!ttcore.tile<32x32, bf16>>,
        tensor<2x2x!ttcore.tile<32x32, bf16>>)

  func.return %result#0, %result#1
      : tensor<2x2x!ttcore.tile<32x32, bf16>>,
        tensor<2x2x!ttcore.tile<32x32, bf16>>
}
