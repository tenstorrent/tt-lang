// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Negative test: multiple multi-tile matmuls in a single fusion chain.

// RUN: ttlang-opt --convert-ttl-to-compute --verify-diagnostics %s

// Two multi-tile matmuls added together: matmul1([4,2]@[2,4]) + matmul2([4,2]@[2,4]).
// Each matmul has a non-trivial K dimension that requires 3D promotion, but
// multiple matmuls in a single chain would need multiple reduction dimensions.

func.func @multi_matmul_multitile() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %a1_cb = ttl.bind_cb{cb_index = 0, block_count = 2} : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  %b1_cb = ttl.bind_cb{cb_index = 1, block_count = 2} : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %a2_cb = ttl.bind_cb{cb_index = 2, block_count = 2} : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  %b2_cb = ttl.bind_cb{cb_index = 3, block_count = 2} : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb{cb_index = 4, block_count = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  %a1_t = ttl.cb_wait %a1_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  %a1 = ttl.attach_cb %a1_t, %a1_cb : (tensor<4x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  %b1_t = ttl.cb_wait %b1_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %b1 = ttl.attach_cb %b1_t, %b1_cb : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %a2_t = ttl.cb_wait %a2_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  %a2 = ttl.attach_cb %a2_t, %a2_cb : (tensor<4x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  %b2_t = ttl.cb_wait %b2_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %b2 = ttl.attach_cb %b2_t, %b2_cb : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %out_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %mm1 = ttl.matmul %a1, %b1 : tensor<4x2x!ttcore.tile<32x32, bf16>>, tensor<2x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %mm2 = ttl.matmul %a2, %b2 : tensor<4x2x!ttcore.tile<32x32, bf16>>, tensor<2x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  // expected-error @+1 {{fusion with multiple multi-tile matmuls is not supported}}
  %sum = ttl.add %mm1, %mm2 : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %reserve : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %out_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %a1_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %b1_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %a2_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %b2_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  return
}
