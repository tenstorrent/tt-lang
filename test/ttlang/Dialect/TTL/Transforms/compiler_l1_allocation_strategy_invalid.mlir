// Verifies that an unknown compiler-managed SRAM allocation strategy is rejected.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=unknown})' --verify-diagnostics

// expected-error @below {{unknown compiler-l1 allocation strategy 'unknown'; expected multi-order-decreasing, first-fit-decreasing, best-fit-decreasing, or exact}}
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @unknown_strategy()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %produced = ttl.cb_reserve %storage
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %consumed = ttl.cb_wait %storage
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
