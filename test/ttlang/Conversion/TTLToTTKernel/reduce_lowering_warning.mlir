// Summary: Tests diagnostics for reduce tile op lowering to TTKernel.

// RUN: ttlang-opt %s --verify-diagnostics \
// RUN:   -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{reduce-full-fp32=true})'

// Blackhole REDUCE_ROW disables full-fp32 lowering because of issue #533.
module attributes {ttl.target_arch = "blackhole"} {
  func.func @blackhole_reduce_sum_dim1_warning() attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %inp = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %inp_cb = ttl.attach_cb %inp, %cb0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler_cb = ttl.attach_cb %scaler, %cb1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %empty = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %out_cb = ttl.attach_cb %empty, %cb2 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %in_tile = tensor.extract %inp_cb[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %sc_tile = tensor.extract %scaler_cb[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %out_tile = tensor.extract %out_cb[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-warning @below {{full-fp32 row reduce is disabled on Blackhole because of issue #533; using non-full-fp32 reduce lowering}}
    %red = ttl.tile_reduce %in_tile, %sc_tile, %out_tile 0 : i32 <reduce_dim_row> into dst[%c0] {ttl.reduce_output_cb_index = 2 : index} : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    func.return
  }
}
