// Test: DST subblocking with non-divisible tile counts adjusts unroll_factor.
// A 3x3 tensor has 9 tiles. With DST capacity=8 and 1 FPU op (dstPerIteration=1),
// the initial unroll_factor=8, but 9 % 8 != 0. The subblock pass adjusts
// unroll_factor down to 3 (largest divisor of 9 that is <= 8), producing
// 3 subblocks of 3 tiles each with constant loop bounds.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttcore-register-device,func.func(convert-ttl-to-compute,ttl-set-compute-kernel-config,ttl-assign-dst,ttl-subblock-compute-for-dst))' | FileCheck %s --check-prefix=SUBBLOCK
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(ttcore-register-device,func.func(convert-ttl-to-compute,ttl-set-compute-kernel-config,ttl-assign-dst,ttl-subblock-compute-for-dst,ttl-insert-tile-regs-sync,ttl-lower-to-loops,ttl-annotate-cb-associations),convert-ttl-to-ttkernel,canonicalize,cse,lower-affine,convert-ttkernel-to-emitc,symbol-dce)' | FileCheck %s --check-prefix=EMITC

// SUBBLOCK-LABEL: func.func @remainder_3x3
// Verify outer loop: 0 to 9 step 3 (adjusted from 8 to 3).
// SUBBLOCK:        scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// SUBBLOCK:          ttl.compute
// SUBBLOCK:            ttl.linearized_index
// SUBBLOCK-NEXT:       arith.addi {{.*}}, %[[IV]]

// EMITC-LABEL: func.func @remainder_3x3

module {
  func.func @remainder_3x3() attributes {ttl.base_cta_index = 0 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb_in = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
    %cb_out = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %cb_in : <[3, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %in = ttl.attach_cb %wait, %cb_in : (tensor<3x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %res = ttl.cb_reserve %cb_out : <[3, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %out = ttl.attach_cb %res, %cb_out : (tensor<3x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    %result = ttl.relu %in : tensor<3x3x!ttcore.tile<32x32, bf16>> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %out : tensor<3x3x!ttcore.tile<32x32, bf16>>, tensor<3x3x!ttcore.tile<32x32, bf16>>
    %out2 = ttl.attach_cb %result, %cb_out : (tensor<3x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb_out : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb_in : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
