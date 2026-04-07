// RUN: ttlang-opt --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' --verify-diagnostics --split-input-file %s

// Summary: Verify that elementwise ops feeding directly into reduce emit a
// clear error instead of a cryptic legalization failure (see issue #474).

// Binary elementwise (mul) feeding scalar reduce on multi-tile input.
// The mul result is not stored to a dataflow buffer, so reduce cannot
// consume it directly.
module {
  func.func @mul_into_scalar_reduce() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %cb3 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    %a_wait = ttl.cb_wait %cb0 : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %a_cb = ttl.attach_cb %a_wait, %cb0 : (tensor<1x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %b_wait = ttl.cb_wait %cb1 : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %b_cb = ttl.attach_cb %b_wait, %cb1 : (tensor<1x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    %sc_wait = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sc_cb = ttl.attach_cb %sc_wait, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reserve = ttl.cb_reserve %cb3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %mul = ttl.mul %a_cb, %b_cb : tensor<1x4x!ttcore.tile<32x32, bf16>>, tensor<1x4x!ttcore.tile<32x32, bf16>> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{elementwise operations feeding into reduce cannot be fused yet; store the intermediate result to a dataflow buffer before passing it to reduce (see issue #474)}}
    %result = ttl.reduce %mul, %sc_cb 0 : i32 [0, 1] : (tensor<1x4x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %reserve : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb3 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb2 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.cb_pop %cb1 : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb0 : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}

// -----

// Unary elementwise (abs) feeding row reduce.
module {
  func.func @abs_into_row_reduce() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>

    %inp_wait = ttl.cb_wait %cb0 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %inp_cb = ttl.attach_cb %inp_wait, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %sc_wait = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sc_cb = ttl.attach_cb %sc_wait, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reserve = ttl.cb_reserve %cb2 : <[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>

    %abs = ttl.abs %inp_cb : tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{elementwise operations feeding into reduce cannot be fused yet; store the intermediate result to a dataflow buffer before passing it to reduce (see issue #474)}}
    %result = ttl.reduce %abs, %sc_cb 0 : i32 [1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %reserve : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb2 : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %cb1 : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.cb_pop %cb0 : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}
