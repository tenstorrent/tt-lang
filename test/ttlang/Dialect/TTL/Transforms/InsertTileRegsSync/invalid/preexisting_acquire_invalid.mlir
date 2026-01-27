// Summary: Pre-existing tile_regs_acquire outside loop should emit an error.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-to-loops,ttl-insert-tile-regs-sync))' --split-input-file --verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify error when tile_regs_acquire exists outside the loop.
func.func @preexisting_acquire(%arg0: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %arg_cb = ttl.attach_cb %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %output_cb = ttl.attach_cb %output, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  // Pre-existing acquire outside compute without matching release - should error
  // expected-error @+1 {{tile_regs_acquire outside tile loop without matching release; sync ops must be inside the loop body}}
  ttl.tile_regs_acquire

  %result = ttl.compute ins(%arg_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%output_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %tok, %tile = ttl.copy_tile %in, %c0, %c0 : !ttcore.tile<32x32, bf16>, index, index -> !ttl.dst, !ttcore.tile<32x32, bf16>
      %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield %tile : !ttcore.tile<32x32, bf16>
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
