// Tests invalid ttl.compute inputs rejected by ttl-lower-to-loops before the
// pass rewrites the compute body.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-lower-to-loops))' --verify-diagnostics --split-input-file

#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>

// Invalid block-matmul compute: the block contains two tile_matmul_block ops.
func.func @two_block_matmuls(
    %a0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %a1: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b1: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cbout = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a0_att = ttl.attach_cb %a0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b0_att = ttl.attach_cb %b0, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a1_att = ttl.attach_cb %a1, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b1_att = ttl.attach_cb %b1, %cb3 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{invalid block matmul compute}}
  %0 = ttl.compute ins(%a0_att, %b0_att, %a1_att, %b1_att : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%init_att : tensor<1x1x!ttcore.tile<32x32, bf16>>) {indexing_maps = [#lhs, #rhs, #lhs, #rhs, #out], iterator_types = ["parallel", "parallel", "reduction"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, bf16>, %arg1: !ttcore.tile<32x32, bf16>, %arg2: !ttcore.tile<32x32, bf16>, %arg3: !ttcore.tile<32x32, bf16>, %arg4: !ttcore.tile<32x32, bf16>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %mm0 = ttl.tile_matmul_block %arg0, %arg1 into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %mm1 = ttl.tile_matmul_block %arg2, %arg3 into dst[%c0] : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    ttl.tile_store %mm1, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  func.return %0 : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
