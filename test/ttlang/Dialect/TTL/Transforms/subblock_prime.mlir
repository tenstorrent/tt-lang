// Tests ttl-subblock-compute-for-dst remainder peeling for the prime-fallback
// case: when the divisor heuristic cannot find ANY subblock (every parallel dim
// forced to size 1 because the block dims are prime and larger than the DST
// budget), the rescue raises a dim to the largest power of two <= the budget and
// the pass tile-and-peels the remainder. Cases the heuristic can already handle
// (e.g. 3x3 -> (1,3)) are covered by subblock_remainder.mlir and are unchanged.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-set-compute-kernel-config,ttl-assign-dst,ttl-subblock-compute-for-dst),canonicalize,cse)' --split-input-file | FileCheck %s

// 11x1 block, bf16 (DST capacity 8). Heuristic yields (1,1) -- 11 is prime and
// > 8. Rescue raises the row dim to 8, giving subblock (8,1). Since 8 does not
// divide 11, the row dim peels into an 8x1 main block + a 3x1 remainder, both
// loop-free (q == 1, no step loop).
//
// CHECK-LABEL: func.func @prime_relu_11x1
// Main 8x1 subblock at row offset 0:
// CHECK:       tensor.extract_slice %{{.*}}[0, 0] [8, 1] [1, 1]
// CHECK:       ttl.compute
// CHECK-SAME:    ttl.full_linearization_strides = array<i64: 1, 1>
// Remainder 3x1 subblock at row offset 8:
// CHECK:       tensor.extract_slice %{{.*}}[8, 0] [3, 1] [1, 1]
// CHECK:       ttl.compute
// CHECK-SAME:    ttl.full_linearization_strides = array<i64: 1, 1>

module {
  func.func @prime_relu_11x1() attributes {ttl.base_cta_index = 0 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb_in = ttl.bind_cb{cb_index = 0, block_count = 2} : <[11, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb_out = ttl.bind_cb{cb_index = 1, block_count = 2} : <[11, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %cb_in : <[11, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<11x1x!ttcore.tile<32x32, bf16>>
    %in = ttl.attach_cb %wait, %cb_in : (tensor<11x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[11, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<11x1x!ttcore.tile<32x32, bf16>>
    %res = ttl.cb_reserve %cb_out : <[11, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<11x1x!ttcore.tile<32x32, bf16>>
    %out = ttl.attach_cb %res, %cb_out : (tensor<11x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[11, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<11x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.relu %in : tensor<11x1x!ttcore.tile<32x32, bf16>> -> tensor<11x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %res : tensor<11x1x!ttcore.tile<32x32, bf16>>, tensor<11x1x!ttcore.tile<32x32, bf16>>
    %out2 = ttl.attach_cb %result, %cb_out : (tensor<11x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[11, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<11x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb_out : <[11, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb_in : <[11, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// 17x1 block, bf16. Heuristic yields (1,1) -- 17 is prime and > 8. Rescue raises
// the row dim to 8, giving subblock (8,1). 17 = 2*8 + 1, so the main region
// [0,16) is a step-8 loop (two 8x1 subblocks) and a 1x1 remainder peels at row
// offset 16.
//
// CHECK-LABEL: func.func @prime_relu_17x1
// Step-8 main loop over the row dim with the subblock-loop annotation:
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:         ttl.compute
// CHECK:       {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 1 : index}
// Loop-free 1x1 remainder at row offset 16:
// CHECK:       tensor.extract_slice %{{.*}}[16, 0] [1, 1] [1, 1]
// CHECK:       ttl.compute

module {
  func.func @prime_relu_17x1() attributes {ttl.base_cta_index = 0 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb_in = ttl.bind_cb{cb_index = 0, block_count = 2} : <[17, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb_out = ttl.bind_cb{cb_index = 1, block_count = 2} : <[17, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %cb_in : <[17, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<17x1x!ttcore.tile<32x32, bf16>>
    %in = ttl.attach_cb %wait, %cb_in : (tensor<17x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[17, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<17x1x!ttcore.tile<32x32, bf16>>
    %res = ttl.cb_reserve %cb_out : <[17, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<17x1x!ttcore.tile<32x32, bf16>>
    %out = ttl.attach_cb %res, %cb_out : (tensor<17x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[17, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<17x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.relu %in : tensor<17x1x!ttcore.tile<32x32, bf16>> -> tensor<17x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %res : tensor<17x1x!ttcore.tile<32x32, bf16>>, tensor<17x1x!ttcore.tile<32x32, bf16>>
    %out2 = ttl.attach_cb %result, %cb_out : (tensor<17x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[17, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<17x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb_out : <[17, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb_in : <[17, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
