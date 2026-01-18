// Summary: Test that the pass detects unreplaced placeholder copy_tile operations.
// The sentinel value 9223372036854775807 (int64_t max) indicates a placeholder
// copy_tile that was not properly replaced during DST assignment.
//
// This test simulates a scenario where a placeholder copy_tile's src is NOT a
// block argument (it's the result of another operation), so the pass cannot
// replace it with a proper copy. This should trigger the post-pass verification.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}))' --verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @invalid_placeholder_copy(%i0: tensor<32x32xf32>, %i1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>

  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

  %t0 = ttl.attach_cb %i0, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t1 = ttl.attach_cb %i1, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute
    ins(%t0, %t1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%x: !ttcore.tile<32x32, f32>, %y: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
    // First compute an intermediate result
    %add = ttl.tile_add %x, %y : !ttcore.tile<32x32, f32>
    // Pre-existing copy_tile with placeholder sentinel indices
    // The src is NOT a block argument (it's %add), so the pass cannot replace it
    %placeholder_src = arith.constant 9223372036854775807 : index
    %placeholder_dst = arith.constant 9223372036854775807 : index
    // expected-error @+1 {{placeholder copy_tile not replaced with proper copy (src_index has sentinel value 9223372036854775807)}}
    %dst_token, %copied = ttl.copy_tile %add, %placeholder_src, %placeholder_dst
        : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %copied : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>

  func.return %res : tensor<32x32xf32>
}
