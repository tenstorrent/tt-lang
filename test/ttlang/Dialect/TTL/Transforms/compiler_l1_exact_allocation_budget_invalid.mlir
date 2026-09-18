// Verifies exact placement proves that two overlapping regions exceed the SRAM budget.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=exact l1-budget-override=12352})'

// The 64-byte control prefix leaves six tiles for two regions requiring seven.
// expected-error @below {{'builtin.module' op compiler-l1 exact placement proves that no allocation fits SRAM budget 12352 bytes}}
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @insufficient_budget()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %storage_0 = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %storage_1 = ttl.bind_cb {cb_index = 1, block_count = 4} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %produced_0 = ttl.cb_reserve %storage_0
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %produced_1 = ttl.cb_reserve %storage_1
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %consumed_0 = ttl.cb_wait %storage_0
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %consumed_1 = ttl.cb_wait %storage_1
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    return
  }
}
