// Negative test: mixed f32 and non-f32 tile arguments in compute.
// RUN: ttlang-opt %s --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(func.func(ttl-assign-dst))'

#idx_map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: Mixed f32/bf16 tile arguments should be rejected.
func.func @mixed_f32_bf16(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                          %b: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cba
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cbb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_cb = ttl.attach_cb %init, %cbout
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // expected-error @+1 {{mixed f32 and non-f32 tile arguments}}
  %res = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#idx_map, #idx_map, #idx_map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_arg: !ttcore.tile<32x32, f32>, %b_arg: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, f32>):
      %c0 = arith.constant 0 : index
      %dtok0, %dtile0 = ttl.copy_tile %a_arg, %c0, %c0 : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %dtok1, %dtile1 = ttl.copy_tile %a_arg, %c0, %c0 : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
      %add = ttl.tile_add %dtile0, %dtile1 : !ttcore.tile<32x32, f32>
      ttl.yield %add : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}
