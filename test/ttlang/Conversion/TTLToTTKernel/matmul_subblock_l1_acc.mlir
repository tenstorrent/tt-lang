// Matmul with subblocking AND L1 accumulation. Output 3x3 bf16 = 9 tiles
// exceeds bf16 DST capacity (8), triggering subblocking. The user K loop
// with {accumulate} triggers L1 acc annotation and pack_reconfig_l1_acc
// guard insertion.

// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module( \
// RUN:     func.func(ttl-annotate-l1-acc-loops, convert-ttl-to-compute, \
// RUN:       ttl-assign-dst{enable-fpu-binary-ops=0}, \
// RUN:       ttl-subblock-compute-for-dst, ttl-lower-matmul-block, \
// RUN:       ttl-lower-to-loops{dst-accumulation=1}, ttl-schedule-operations, \
// RUN:       ttl-annotate-cb-associations), \
// RUN:     convert-ttl-to-ttkernel, ttkernel-insert-inits, \
// RUN:     ttkernel-insert-l1-accumulation, canonicalize, cse)' \
// RUN:   --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @matmul_3x3_k_loop
// Disable before the K loop.
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[C0_I32]])
// K loop with subblock loops inside.
// CHECK: scf.for %[[K_IV:.*]] = %[[K_LB:.*]] to
// Subblock loop: acquire, matmul, 3 pack_tiles (3x1 subblock), release.
// CHECK:   scf.for
// CHECK:     ttkernel.tile_regs_acquire
// CHECK:     ttkernel.matmul_block
// CHECK-COUNT-3: ttkernel.pack_tile
// CHECK:     ttkernel.tile_regs_release
// CHECK:   }
// Enable after first K iteration.
// CHECK:   arith.cmpi eq, %[[K_IV]], %[[K_LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc(%[[C1_I32]])
// CHECK: }
// Disable after push.
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[C0_I32]])
func.func @matmul_3x3_k_loop(
    %arg0: tensor<3x2x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<2x3x!ttcore.tile<32x32, bf16>>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[3, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<3x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x2x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[3, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  scf.for %k = %c0 to %c2 step %c1 {
    %mm = ttl.matmul %a, %b : tensor<3x2x!ttcore.tile<32x32, bf16>>, tensor<2x3x!ttcore.tile<32x32, bf16>> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    ttl.store %mm, %reserve {accumulate} : tensor<3x3x!ttcore.tile<32x32, bf16>>, tensor<3x3x!ttcore.tile<32x32, bf16>>
  }
  ttl.cb_push %cb2 : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
  func.return %reserve : tensor<3x3x!ttcore.tile<32x32, bf16>>
}

// -----

// 8x8 output (64 tiles >> DST capacity 8) with K=4: heavily subblocked.
// Verifies that multiple levels of subblock loops all sit inside the
// K loop's L1 acc guards.

// CHECK-LABEL: func.func @matmul_8x8_k4
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// Disable before K loop.
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[C0_I32]])
// K loop -> subblock row loop -> acquire, matmul K loop, 8x pack, release.
// CHECK: scf.for %[[K_IV:.*]] = %[[K_LB:.*]] to
// CHECK:   scf.for
// CHECK:     ttkernel.tile_regs_acquire
// CHECK:     scf.for
// CHECK:       ttkernel.matmul_block
// CHECK-COUNT-8: ttkernel.pack_tile
// CHECK:     ttkernel.tile_regs_release
// CHECK:   }
// Enable after first K iteration.
// CHECK:   arith.cmpi eq, %[[K_IV]], %[[K_LB]]
// CHECK:   scf.if
// CHECK:     ttkernel.pack_reconfig_l1_acc(%[[C1_I32]])
// CHECK: }
// Disable after push.
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.pack_reconfig_l1_acc(%[[C0_I32]])
func.func @matmul_8x8_k4(
    %arg0: tensor<8x8x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<8x8x!ttcore.tile<32x32, bf16>>) -> tensor<8x8x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[8, 8], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[8, 8], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[8, 8], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<8x8x!ttcore.tile<32x32, bf16>>, !ttl.cb<[8, 8], !ttcore.tile<32x32, bf16>, 2>) -> tensor<8x8x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<8x8x!ttcore.tile<32x32, bf16>>, !ttl.cb<[8, 8], !ttcore.tile<32x32, bf16>, 2>) -> tensor<8x8x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[8, 8], !ttcore.tile<32x32, bf16>, 2> -> tensor<8x8x!ttcore.tile<32x32, bf16>>
  scf.for %k = %c0 to %c4 step %c1 {
    %mm = ttl.matmul %a, %b : tensor<8x8x!ttcore.tile<32x32, bf16>>, tensor<8x8x!ttcore.tile<32x32, bf16>> -> tensor<8x8x!ttcore.tile<32x32, bf16>>
    ttl.store %mm, %reserve {accumulate} : tensor<8x8x!ttcore.tile<32x32, bf16>>, tensor<8x8x!ttcore.tile<32x32, bf16>>
  }
  ttl.cb_push %cb2 : <[8, 8], !ttcore.tile<32x32, bf16>, 2>
  func.return %reserve : tensor<8x8x!ttcore.tile<32x32, bf16>>
}
