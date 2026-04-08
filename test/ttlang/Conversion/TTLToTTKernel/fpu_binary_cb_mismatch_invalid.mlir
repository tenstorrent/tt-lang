// RUN: ttlang-opt %s --convert-ttl-to-ttkernel -split-input-file -verify-diagnostics

// FPU binary op where lhs and rhs CBs have different tile counts.
// The lhs CB is [2, 2] (4 tiles) while the rhs CB is [1, 2] (2 tiles).
// The FPU lowering pattern rejects this because it uses the same linearized
// tile index for both operands, which requires matching CB shapes.
func.func @fpu_add_cb_shape_mismatch()
    attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>

  %lhs_ready = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %lhs = ttl.attach_cb %lhs_ready, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %rhs_ready = ttl.cb_wait %cb2 : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %rhs = ttl.attach_cb %rhs_ready, %cb2 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>

  %c0 = arith.constant 0 : index
  %lhs_tile = tensor.extract %lhs[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, bf16>>
  %rhs_tile = tensor.extract %rhs[%c0, %c0] : tensor<1x2x!ttcore.tile<32x32, bf16>>

  // expected-error @+1 {{failed to legalize operation 'ttl.tile_add' that was explicitly marked illegal}}
  %sum = ttl.tile_add %lhs_tile, %rhs_tile {"ttl.fpu_binary", dst_idx = 0 : i32} : !ttcore.tile<32x32, bf16>

  %out_view = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %out = tensor.insert %sum into %out_view[%c0, %c0] : tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  return
}
