// RUN: ttlang-opt --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' --verify-diagnostics --split-input-file %s

// Summary: Invalid broadcast lowering tests. Verify that the compiler emits
// errors for broadcast dims incompatible with the producing reduce. The
// op-level verifier rejects structural errors (input/shape size mismatch,
// row-major, missing CB) before the lowering even runs; the cases here cover
// the reduce-dim/derived-bcast-type cross-check that requires SSA tracing.

// Scalar reduce (REDUCE_SCALAR) feeding a ROW broadcast (dims=[-2]).
// ROW unpack reads row 0, but REDUCE_SCALAR only has valid data at [0,0].
module {
  func.func @bcast_row_after_scalar_reduce() {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>

    %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [0, 1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %red_res : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttl.block.broadcast' op broadcast dims are incompatible with the producing reduce; need scalar broadcast (dims=[-2, -1])}}
    %bcast = ttl.block.broadcast %red_a dims = [-2], shape = [2, 1] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.store %bcast, %out_res : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Scalar reduce (REDUCE_SCALAR) feeding a COL broadcast (dims=[-1]).
// COL unpack reads column 0, but REDUCE_SCALAR only has valid data at [0,0].
module {
  func.func @bcast_col_after_scalar_reduce() {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>

    %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [0, 1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %red_res : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttl.block.broadcast' op broadcast dims are incompatible with the producing reduce; need scalar broadcast (dims=[-2, -1])}}
    %bcast = ttl.block.broadcast %red_a dims = [-1], shape = [1, 2] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.store %bcast, %out_res : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
