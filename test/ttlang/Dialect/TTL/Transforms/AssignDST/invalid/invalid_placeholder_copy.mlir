// Summary: Test that the pass detects unreplaced placeholder copy_tile operations.
// The sentinel value 9223372036854775807 (int64_t max) indicates a placeholder
// copy_tile that was not properly replaced during DST assignment.
//
// This test simulates a scenario where a placeholder copy_tile's src is NOT a
// block argument (it's the result of another operation), so the pass cannot
// replace it with a proper copy. This should trigger the post-pass verification.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}))' --verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @invalid_placeholder_copy(%i0: tensor<1x1x!ttcore.tile<32x32, f32>>,
                                     %i1: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %t0 = ttl.attach_cb %i0, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %t1 = ttl.attach_cb %i1, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %t_init = ttl.attach_cb %init, %cbout : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
    ins(%t0, %t1 : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>)
    outs(%t_init : tensor<1x1x!ttcore.tile<32x32, f32>>)
    {indexing_maps = [#map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%x: !ttcore.tile<32x32, f32>, %y: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // First compute an intermediate result
    %c0 = arith.constant 0 : index
    %add = ttl.tile_add %x, %y into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    // Pre-existing copy_tile with placeholder sentinel indices
    // The src is NOT a block argument (it's %add), so the pass cannot replace it
    %placeholder_dst = arith.constant 9223372036854775807 : index
    // expected-error @+1 {{placeholder copy_tile not replaced with proper copy}}
    %dst_token, %copied = ttl.copy_tile %add[] into dst[%placeholder_dst]
        {ttl.placeholder_copy}
        : !ttcore.tile<32x32, f32> -> !ttl.dst, !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %copied into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}
