// Tests ttl-subblock-compute-for-dst remainder peeling. When the divisor
// heuristic yields one tile because no parallel dimension has a divisor
// greater than one within the DST budget, the rescue
// raises one dimension to the budget and the pass tile-and-peels the leftover
// tiles. Cases the heuristic already handles (e.g. 3x3 -> (1,3)) are covered by
// subblock_remainder.mlir and are unchanged.
//
// The budget is pinned per RUN line rather than left to the dtype default, so a
// change to the default capacity cannot silently void these expectations.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),ttl-set-compute-kernel-config,func.func(ttl-assign-dst{dst-capacity=8},ttl-subblock-compute-for-dst),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=DST8
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),ttl-set-compute-kernel-config,func.func(ttl-assign-dst{dst-capacity=4},ttl-subblock-compute-for-dst),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=DST4
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),ttl-set-compute-kernel-config,func.func(ttl-assign-dst{dst-capacity=6},ttl-subblock-compute-for-dst),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=DST6
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),ttl-set-compute-kernel-config,func.func(ttl-assign-dst{dst-capacity=8},ttl-subblock-compute-for-dst{subblock-sync=false}),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=SYNC --implicit-check-not=ttl.cb_reserve --implicit-check-not=ttl.cb_push
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),ttl-set-compute-kernel-config,func.func(ttl-assign-dst{dst-capacity=8},ttl-subblock-compute-for-dst{subblock-sync=true}),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=SYNC --implicit-check-not=ttl.cb_reserve --implicit-check-not=ttl.cb_push

// Peeling must retain exactly one whole-block reservation and publication in
// either mode, including when the main region contains a subblock loop.
// SYNC-LABEL: func.func @prime_relu_11x1
// SYNC: %[[OUT11:.*]] = ttl.bind_cb{cb_index = 1,
// SYNC: ttl.cb_reserve %[[OUT11]] : <[11, 1],
// SYNC: ttl.cb_push %[[OUT11]] : <[11, 1],
// SYNC: return
// SYNC-LABEL: func.func @prime_relu_17x1
// SYNC: %[[OUT17:.*]] = ttl.bind_cb{cb_index = 1,
// SYNC: ttl.cb_reserve %[[OUT17]] : <[17, 1],
// SYNC: ttl.cb_push %[[OUT17]] : <[17, 1],
// SYNC: return
// SYNC-LABEL: func.func @prime_relu_7x1_f32
// SYNC: %[[OUT7:.*]] = ttl.bind_cb{cb_index = 1,
// SYNC: ttl.cb_reserve %[[OUT7]] : <[7, 1],
// SYNC: ttl.cb_push %[[OUT7]] : <[7, 1],
// SYNC: return
// SYNC-LABEL: func.func @prime_relu_11x11
// SYNC: %[[OUT2D:.*]] = ttl.bind_cb{cb_index = 1,
// SYNC: ttl.cb_reserve %[[OUT2D]] : <[11, 11],
// SYNC: ttl.cb_push %[[OUT2D]] : <[11, 11],
// SYNC: return

// 11x1 block at budget 8. The heuristic yields (1,1) -- 11 is prime and > 8.
// The rescue raises the row dim to 8, which does not divide 11, so the row dim
// peels into an 8x1 main block and a 3x1 remainder, both loop-free (q == 1).
//
// DST8-LABEL: func.func @prime_relu_11x1
// DST8:       tensor.extract_slice %{{.*}}[0, 0] [8, 1] [1, 1]
// DST8:       ttl.compute
// DST8-SAME:    ttl.full_linearization_strides = array<i64: 1, 1>
// DST8:       tensor.extract_slice %{{.*}}[8, 0] [3, 1] [1, 1]
// DST8:       ttl.compute
// DST8-SAME:    ttl.full_linearization_strides = array<i64: 1, 1>

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
    ttl.cb_push %cb_out : <[11, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb_in : <[11, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// 17x1 block at budget 8. 17 = 2*8 + 1, so the main region [0,16) is a step-8
// loop carrying the subblock annotations and a 1x1 remainder peels at row 16.
//
// DST8-LABEL: func.func @prime_relu_17x1
// DST8:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// DST8:         ttl.compute
// DST8:       {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 1 : index}
// DST8:       tensor.extract_slice %{{.*}}[16, 0] [1, 1] [1, 1]
// DST8:       ttl.compute

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
    ttl.cb_push %cb_out : <[17, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb_in : <[17, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Issue #703's reproducer: a 7x1 f32 block at budget 4 rescues to (4,1) and
// peels into a 4x1 main block plus a 3x1 remainder, instead of degrading to one
// tile at a time.
//
// DST4-LABEL: func.func @prime_relu_7x1_f32
// DST4:       tensor.extract_slice %{{.*}}[0, 0] [4, 1] [1, 1]
// DST4:       ttl.compute
// DST4:       tensor.extract_slice %{{.*}}[4, 0] [3, 1] [1, 1]
// DST4:       ttl.compute

module {
  func.func @prime_relu_7x1_f32() attributes {ttl.base_cta_index = 0 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb_in = ttl.bind_cb{cb_index = 0, block_count = 2} : <[7, 1], !ttcore.tile<32x32, f32>, 2>
    %cb_out = ttl.bind_cb{cb_index = 1, block_count = 2} : <[7, 1], !ttcore.tile<32x32, f32>, 2>
    %wait = ttl.cb_wait %cb_in : <[7, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<7x1x!ttcore.tile<32x32, f32>>
    %in = ttl.attach_cb %wait, %cb_in : (tensor<7x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[7, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<7x1x!ttcore.tile<32x32, f32>>
    %res = ttl.cb_reserve %cb_out : <[7, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<7x1x!ttcore.tile<32x32, f32>>
    %out = ttl.attach_cb %res, %cb_out : (tensor<7x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[7, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<7x1x!ttcore.tile<32x32, f32>>
    %result = ttl.relu %in : tensor<7x1x!ttcore.tile<32x32, f32>> -> tensor<7x1x!ttcore.tile<32x32, f32>>
    ttl.store %result, %res : tensor<7x1x!ttcore.tile<32x32, f32>>, tensor<7x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb_out : <[7, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.cb_pop %cb_in : <[7, 1], !ttcore.tile<32x32, f32>, 2>
    return
  }
}

// -----

// Both dims prime and larger than the budget, at a budget that is not a power
// of two. The rescue gives the whole budget to the innermost usable dimension,
// so the column dim takes 6 -- not the largest power of two below it -- and the
// row dim keeps size 1, walked by a step-1 loop. Exactly one dimension peels.
//
// DST6-LABEL: func.func @prime_relu_11x11
// DST6:       scf.for
// DST6:         tensor.extract_slice %{{.*}}[%{{.*}}, 0] [1, 6] [1, 1]
// DST6:         ttl.compute
// DST6:       scf.for
// DST6:         tensor.extract_slice %{{.*}}[%{{.*}}, 6] [1, 5] [1, 1]
// DST6:         ttl.compute
// DST6-NOT:   [2, 4]

module {
  func.func @prime_relu_11x11() attributes {ttl.base_cta_index = 0 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb_in = ttl.bind_cb{cb_index = 0, block_count = 2} : <[11, 11], !ttcore.tile<32x32, bf16>, 2>
    %cb_out = ttl.bind_cb{cb_index = 1, block_count = 2} : <[11, 11], !ttcore.tile<32x32, bf16>, 2>
    %wait = ttl.cb_wait %cb_in : <[11, 11], !ttcore.tile<32x32, bf16>, 2> -> tensor<11x11x!ttcore.tile<32x32, bf16>>
    %in = ttl.attach_cb %wait, %cb_in : (tensor<11x11x!ttcore.tile<32x32, bf16>>, !ttl.cb<[11, 11], !ttcore.tile<32x32, bf16>, 2>) -> tensor<11x11x!ttcore.tile<32x32, bf16>>
    %res = ttl.cb_reserve %cb_out : <[11, 11], !ttcore.tile<32x32, bf16>, 2> -> tensor<11x11x!ttcore.tile<32x32, bf16>>
    %out = ttl.attach_cb %res, %cb_out : (tensor<11x11x!ttcore.tile<32x32, bf16>>, !ttl.cb<[11, 11], !ttcore.tile<32x32, bf16>, 2>) -> tensor<11x11x!ttcore.tile<32x32, bf16>>
    %result = ttl.relu %in : tensor<11x11x!ttcore.tile<32x32, bf16>> -> tensor<11x11x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %res : tensor<11x11x!ttcore.tile<32x32, bf16>>, tensor<11x11x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb_out : <[11, 11], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb_in : <[11, 11], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
