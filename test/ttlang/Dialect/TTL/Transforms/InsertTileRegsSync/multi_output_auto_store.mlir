// RUN: ttlang-opt %s -ttl-insert-tile-regs-sync | FileCheck %s

// Test: Multiple outputs without explicit stores get auto-inserted stores.
// Each output maps to its corresponding CB via iter_arg index.

#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>

// CHECK-LABEL: func.func @multi_output_auto_store
func.func @multi_output_auto_store()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>

  %input = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %out0 = ttl.attach_cb %init0, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out1 = ttl.attach_cb %init1, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.init_sfpu
  // Reserve goes BEFORE outermost loop (auto-inserted for missing stores)
  // CHECK: %[[VIEW1:.*]] = ttl.cb_reserve %{{.*}} :
  // CHECK: %[[VIEW2:.*]] = ttl.cb_reserve %{{.*}} :
  // CHECK: scf.for
  %r:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg0 = %out0, %arg1 = %out1) -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
    // CHECK: scf.for
    %inner:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0, %arg3 = %arg1) -> (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) {
      // CHECK: ttl.tile_regs_acquire
      %tile = tensor.extract %input[%i, %j] : tensor<2x2x!ttcore.tile<32x32, f32>>
      %idx = affine.apply #map(%i, %j)
      %dst_token, %dst_tile = ttl.copy_tile %tile, %idx, %c0 : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %exp = ttl.tile_exp %dst_tile {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
      %neg = ttl.tile_neg %dst_tile {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
      %ins0 = tensor.insert %exp into %arg2[%i, %j] : tensor<2x2x!ttcore.tile<32x32, f32>>
      %ins1 = tensor.insert %neg into %arg3[%i, %j] : tensor<2x2x!ttcore.tile<32x32, f32>>
      // CHECK: ttl.tile_regs_commit
      // CHECK: ttl.tile_regs_wait
      // Stores go INSIDE the loop (pack each tile)
      // CHECK: ttl.store %{{.*}}, %[[VIEW1]]
      // CHECK: ttl.store %{{.*}}, %[[VIEW2]]
      // CHECK: ttl.tile_regs_release
      scf.yield %ins0, %ins1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
    } {ttl.tile_loop, ttl.tile_loop.input_cbs = [0], ttl.tile_loop.output_cbs = [1, 2]}
    scf.yield %inner#0, %inner#1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  } {ttl.tile_loop}
  // Push goes AFTER outermost loop (signals all tiles ready)
  // CHECK: ttl.cb_push
  // CHECK: ttl.cb_push

  return
}
