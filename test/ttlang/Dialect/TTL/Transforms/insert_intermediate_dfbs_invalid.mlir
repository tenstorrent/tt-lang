// Negative test: exceeding the conservative DFB-index capacity.
// The insert pass creates compiler-allocated DFBs without checking the limit;
// the finalize pass enforces target capacity after index reuse.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(func.func(ttl-insert-intermediate-dfbs,ttl-insert-cb-sync),ttl-finalize-dfb-indices{reuse-user-dfbs=false})'

// -----

// All 32 CB indices (0-31) are occupied by user DFBs. The add result
// feeds reduce, which requires a compiler-allocated DFB at index 32.
// After reuse (only one intermediate, nothing to reuse), the total DFB
// count is 33, exceeding the 32-index capacity used without target metadata.

// expected-error @below {{need 33 unspilled DFB indices, exceeding the conservative 32-DFB-index capacity used when target metadata is absent (1 compiler-allocated after proven reuse)}}
module {
func.func @exceeds_max_cb_count()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, block_count = 2} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb5 = ttl.bind_cb {cb_index = 5, block_count = 2} {dfb_id = 5 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb6 = ttl.bind_cb {cb_index = 6, block_count = 2} {dfb_id = 6 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb7 = ttl.bind_cb {cb_index = 7, block_count = 2} {dfb_id = 7 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb8 = ttl.bind_cb {cb_index = 8, block_count = 2} {dfb_id = 8 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb9 = ttl.bind_cb {cb_index = 9, block_count = 2} {dfb_id = 9 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb10 = ttl.bind_cb {cb_index = 10, block_count = 2} {dfb_id = 10 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb11 = ttl.bind_cb {cb_index = 11, block_count = 2} {dfb_id = 11 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb12 = ttl.bind_cb {cb_index = 12, block_count = 2} {dfb_id = 12 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb13 = ttl.bind_cb {cb_index = 13, block_count = 2} {dfb_id = 13 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb14 = ttl.bind_cb {cb_index = 14, block_count = 2} {dfb_id = 14 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb15 = ttl.bind_cb {cb_index = 15, block_count = 2} {dfb_id = 15 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb16 = ttl.bind_cb {cb_index = 16, block_count = 2} {dfb_id = 16 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb17 = ttl.bind_cb {cb_index = 17, block_count = 2} {dfb_id = 17 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb18 = ttl.bind_cb {cb_index = 18, block_count = 2} {dfb_id = 18 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb19 = ttl.bind_cb {cb_index = 19, block_count = 2} {dfb_id = 19 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb20 = ttl.bind_cb {cb_index = 20, block_count = 2} {dfb_id = 20 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb21 = ttl.bind_cb {cb_index = 21, block_count = 2} {dfb_id = 21 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb22 = ttl.bind_cb {cb_index = 22, block_count = 2} {dfb_id = 22 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb23 = ttl.bind_cb {cb_index = 23, block_count = 2} {dfb_id = 23 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb24 = ttl.bind_cb {cb_index = 24, block_count = 2} {dfb_id = 24 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb25 = ttl.bind_cb {cb_index = 25, block_count = 2} {dfb_id = 25 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb26 = ttl.bind_cb {cb_index = 26, block_count = 2} {dfb_id = 26 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb27 = ttl.bind_cb {cb_index = 27, block_count = 2} {dfb_id = 27 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb28 = ttl.bind_cb {cb_index = 28, block_count = 2} {dfb_id = 28 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb29 = ttl.bind_cb {cb_index = 29, block_count = 2} {dfb_id = 29 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb30 = ttl.bind_cb {cb_index = 30, block_count = 2} {dfb_id = 30 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb31 = ttl.bind_cb {cb_index = 31, block_count = 2} {dfb_id = 31 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_wait = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %a = ttl.attach_cb %a_wait, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_wait = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %b_wait, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %s_wait = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %s = ttl.attach_cb %s_wait, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %add = ttl.add %a, %b : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %reduced = ttl.reduce %add, %s 0 : i32 [1] : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
}
