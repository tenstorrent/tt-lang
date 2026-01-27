// Negative tests for init_sfpu CB selection.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(func.func(ttl-lower-to-loops,ttl-insert-tile-regs-sync))'

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test: Error when tile loop has input CBs in attributes but no tensor.extract
// ops in the loop body trace back to those CBs. This indicates a malformed loop
// where init_sfpu cannot determine the correct CB for hardware configuration.

func.func @no_input_cb_extracted(%arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index

  // Bind input and output CBs
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>

  %a_attached = ttl.attach_cb %arg0, %cb0
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out_attached = ttl.attach_cb %init, %cb1
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // Compute where the body doesn't actually use the input - it just yields a constant tile.
  // This is a pathological case where the loop has an input but doesn't extract from it.
  // expected-error @below {{init_sfpu: no input CB is extracted in the loop body}}
  %result = ttl.compute
      ins(%a_attached : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      outs(%out_attached : tensor<2x2x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%in: !ttcore.tile<32x32, bf16>,
       %out: !ttcore.tile<32x32, bf16>):
    // Don't use %in at all - this means no tensor.extract from the input CB
    // Just yield the output tile (identity on output)
    ttl.yield %out : !ttcore.tile<32x32, bf16>
  } -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, bf16>>
}
