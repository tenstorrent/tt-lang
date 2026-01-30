// RUN: ttlang-opt %s -ttl-insert-tile-regs-sync -verify-diagnostics -split-input-file

// Test: tensor.insert with destination that is not an iter_arg block argument
// should produce an error when auto-store insertion tries to determine the CB.
// This tests malformed IR where the insert destination is a value defined
// outside the loop rather than an iter_arg.

#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>

func.func @insert_into_non_iter_arg()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  %input = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %output = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  // Malformed: %output is defined outside the loop, not an iter_arg
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      %tile = tensor.extract %input[%i, %j] : tensor<2x2x!ttcore.tile<32x32, f32>>
      %idx = affine.apply #map(%i, %j)
      %dst_token, %dst_tile = ttl.copy_tile %tile, %idx, %c0 : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %exp = ttl.tile_exp %dst_tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
      // expected-error @below {{could not determine output CB for tensor.insert; destination must be an iter_arg block argument}}
      %inserted = tensor.insert %exp into %output[%i, %j] : tensor<2x2x!ttcore.tile<32x32, f32>>
    } {ttl.tile_loop, ttl.tile_loop.input_cbs = [0], ttl.tile_loop.output_cbs = [1]}
  } {ttl.tile_loop}

  return
}
