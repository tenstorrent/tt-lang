// Invalid ttl.tile_store placements caught by ttl.compute verifier.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: store uses a view not produced by ttl.cb_reserve.
func.func @view_not_from_reserve(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>, %view: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %result = ttl.compute ins(%arg_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %tok, %tile = ttl.copy_tile %in[%c0], %c0 : !ttcore.tile<32x32, bf16>, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      // expected-error @below {{'ttl.tile_store' op view must be produced by ttl.cb_reserve}}
      ttl.tile_store %tile, %view[%i, %j] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
