// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests that ConvertTTLToCompute fuses prev + a @ b into a 3-operand
// tile_matmul_block for multi-tile block shapes. See tt-lang#460.

// RUN: ttlang-opt --convert-ttl-to-compute --split-input-file %s | FileCheck %s

// Multi-tile fused matmul+add: [4,2] @ [2,4] + [4,4] -> [4,4].
// The matmul's K dimension (2) does not match the output (4), so
// broadcast-based fusion cannot handle this; the matmul pattern must
// absorb the add and build a 3D iteration space with reduction on K.

// CHECK-LABEL: func.func @matmul_add_multitile
// CHECK: ttl.compute
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: ttl.tile_matmul_block %[[LHS:.*]], %[[RHS:.*]], %[[ACC:.*]] :
// CHECK-NOT: ttl.matmul
// CHECK-NOT: ttl.add
func.func @matmul_add_multitile() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %a_cb = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  %b_cb = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %acc_cb = ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c3 step %c1 {
    %a_t = ttl.cb_wait %a_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x2x!ttcore.tile<32x32, bf16>>
    %a = ttl.attach_cb %a_t, %a_cb : (tensor<4x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x2x!ttcore.tile<32x32, bf16>>
    %b_t = ttl.cb_wait %b_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
    %b = ttl.attach_cb %b_t, %b_cb : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
    %prev_t = ttl.cb_wait %acc_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %prev = ttl.attach_cb %prev_t, %acc_cb : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %reserve = ttl.cb_reserve %acc_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %out = ttl.attach_cb %reserve, %acc_cb : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %mm = ttl.matmul %a, %b : tensor<4x2x!ttcore.tile<32x32, bf16>>, tensor<2x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    %sum = ttl.add %prev, %mm : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.store %sum, %reserve : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %acc_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %acc_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %b_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %a_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  }
  return
}

// -----

// Commuted add: matmul result on LHS, accumulator on RHS.

// CHECK-LABEL: func.func @matmul_add_commuted
// CHECK: ttl.compute
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: ttl.tile_matmul_block %{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-NOT: ttl.matmul
// CHECK-NOT: ttl.add
func.func @matmul_add_commuted() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %a_cb = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  %b_cb = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %acc_cb = ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  %a_t = ttl.cb_wait %a_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_t, %a_cb : (tensor<4x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  %b_t = ttl.cb_wait %b_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %b_t, %b_cb : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %prev_t = ttl.cb_wait %acc_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %prev = ttl.attach_cb %prev_t, %acc_cb : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %acc_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %reserve, %acc_cb : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<4x2x!ttcore.tile<32x32, bf16>>, tensor<2x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  // Matmul result is LHS of the add (commuted order).
  %sum = ttl.add %mm, %prev : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  ttl.store %sum, %reserve : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %acc_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %acc_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %b_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %a_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Standalone multi-tile matmul (no add): should still lower as before.

// CHECK-LABEL: func.func @matmul_standalone_multitile
// CHECK: ttl.compute
// CHECK: ttl.tile_matmul_block
// CHECK-NOT: ttl.matmul
func.func @matmul_standalone_multitile() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %a_cb = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  %b_cb = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  %a_t = ttl.cb_wait %a_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_t, %a_cb : (tensor<4x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x2x!ttcore.tile<32x32, bf16>>
  %b_t = ttl.cb_wait %b_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %b_t, %b_cb : (tensor<2x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %out_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %out = ttl.attach_cb %reserve, %out_cb : (tensor<4x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  %mm = ttl.matmul %a, %b : tensor<4x2x!ttcore.tile<32x32, bf16>>, tensor<2x4x!ttcore.tile<32x32, bf16>> -> tensor<4x4x!ttcore.tile<32x32, bf16>>
  ttl.store %mm, %reserve : tensor<4x4x!ttcore.tile<32x32, bf16>>, tensor<4x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %out_cb : <[4, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %a_cb : <[4, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %b_cb : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  return
}
