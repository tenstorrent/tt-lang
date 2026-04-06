// RUN: ttlang-opt --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' --verify-diagnostics --split-input-file %s

// Summary: Invalid broadcast lowering tests. Verify that the compiler emits
// errors for broadcast dims incompatible with the producing reduce.

// Scalar reduce (dims=[0,1]) feeding a ROW broadcast (dims=[0]).
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
    %out_init = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
    %out_a = ttl.attach_cb %out_init, %out_cb : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttl.bcast' op broadcast dims are incompatible with the producing reduce; need scalar broadcast (dims=[0, 1])}}
    %bcast = ttl.bcast %red_a, %out_a 2 : i32 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.store %bcast, %out_res : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Scalar reduce (dims=[0,1]) feeding a COL broadcast (dims=[1]).
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
    %out_init = tensor.empty() : tensor<1x2x!ttcore.tile<32x32, bf16>>
    %out_a = ttl.attach_cb %out_init, %out_cb : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttl.bcast' op broadcast dims are incompatible with the producing reduce; need scalar broadcast (dims=[0, 1])}}
    %bcast = ttl.bcast %red_a, %out_a 1 : i32 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.store %bcast, %out_res : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Col reduce (dims=[0]) feeding a COL broadcast (dims=[1]) with no expansion.
// COL unpack reads column 0, but REDUCE_COL leaves valid data in row 0.
module {
  func.func @bcast_col_after_col_reduce() {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
    %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>

    %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [0] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %red_res : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %red_cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>

    %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    %out_init = tensor.empty() : tensor<1x2x!ttcore.tile<32x32, bf16>>
    %out_a = ttl.attach_cb %out_init, %out_cb : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttl.bcast' op broadcast dims are incompatible with the producing reduce; need row broadcast (dims=[0])}}
    %bcast = ttl.bcast %red_a, %out_a 1 : i32 : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.store %bcast, %out_res : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Row reduce (dims=[1]) feeding a ROW broadcast (dims=[0]) with no expansion.
// ROW unpack reads row 0, but REDUCE_ROW leaves valid data in column 0.
module {
  func.func @bcast_row_after_row_reduce() {
    %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
    %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
    %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>

    %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %red_res : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %red_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>

    %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    %out_init = tensor.empty() : tensor<2x1x!ttcore.tile<32x32, bf16>>
    %out_a = ttl.attach_cb %out_init, %out_cb : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{'ttl.bcast' op broadcast dims are incompatible with the producing reduce; need column broadcast (dims=[1])}}
    %bcast = ttl.bcast %red_a, %out_a 2 : i32 : (tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    ttl.store %bcast, %out_res : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
