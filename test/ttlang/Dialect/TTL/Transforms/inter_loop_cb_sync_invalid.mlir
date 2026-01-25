// Invalid cases for ttl-insert-inter-loop-cb-sync pass.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --ttl-insert-inter-loop-cb-sync

// Test: Error when loop attribute references a CB index that has no bind_cb.
// This can happen if IR is malformed or manually constructed with invalid
// CB index attributes.

func.func @missing_bind_cb_for_referenced_index()
    -> tensor<1x1x!ttcore.tile<32x32, f32>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  // Only bind CB 0, but loops will reference CB 99 which doesn't exist.
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // First loop: output_cbs references CB 99 (non-existent).
  %r0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %init)
      -> (tensor<1x1x!ttcore.tile<32x32, f32>>) {
    scf.yield %acc : tensor<1x1x!ttcore.tile<32x32, f32>>
  } {ttl.tile_loop, ttl.tile_loop.input_cbs = [0], ttl.tile_loop.output_cbs = [99]}

  // Second loop: input_cbs references CB 99 (matches first loop's output).
  // expected-error @below {{op inter-loop CB sync failed: loop attribute references cb_index 99 but no bind_cb with that index exists}}
  %r1 = scf.for %j = %c0 to %c1 step %c1 iter_args(%acc2 = %r0)
      -> (tensor<1x1x!ttcore.tile<32x32, f32>>) {
    scf.yield %acc2 : tensor<1x1x!ttcore.tile<32x32, f32>>
  } {ttl.tile_loop, ttl.tile_loop.input_cbs = [99], ttl.tile_loop.output_cbs = [0]}

  func.return %r1 : tensor<1x1x!ttcore.tile<32x32, f32>>
}
