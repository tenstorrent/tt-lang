// Test: DST subblocking with non-divisible tile counts adjusts subblock size.
// A 3x3 tensor has 9 tiles. With DST capacity=8 and 1 unary op (dstPerIteration=1),
// the initial unroll_factor=8, but no subblock of size 8 evenly divides 9.
// Multi-dim tiling finds tileSizes=[1,3] (product=3), producing 3 subblocks
// of 3 tiles each with constant loop bounds. Loop on dim 0 (0 to 3 step 1).

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-set-compute-kernel-config,ttl-assign-dst,ttl-subblock-compute-for-dst))' | FileCheck %s --check-prefix=SUBBLOCK

// SUBBLOCK-LABEL: func.func @remainder_3x3
// Verify outer loop with inner ttl.compute containing linearized_index offset.
// Stride 3 for dim 0: arith.muli(iv, 3) then arith.addi.
// SUBBLOCK:        scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// SUBBLOCK:          ttl.compute
// SUBBLOCK:            ttl.linearized_index
// SUBBLOCK:            arith.muli %[[IV]],
// SUBBLOCK-NEXT:       arith.addi

module {
  func.func @remainder_3x3() attributes {ttl.base_cta_index = 0 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb_in = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
    %cb_out = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %cb_in : <[3, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %in = ttl.attach_cb %wait, %cb_in : (tensor<3x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %res = ttl.cb_reserve %cb_out : <[3, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %out = ttl.attach_cb %res, %cb_out : (tensor<3x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %result = ttl.relu %in : tensor<3x3x!ttcore.tile<32x32, bf16>> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %res : tensor<3x3x!ttcore.tile<32x32, bf16>>, tensor<3x3x!ttcore.tile<32x32, bf16>>
    %out2 = ttl.attach_cb %result, %cb_out : (tensor<3x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb_out : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb_in : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
