// Verifies both memory models reject a zero minimum-arena search limit.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-sram sram-allocation-strategy=minimum-arena sram-minimum-arena-search-limit=0})'
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=metal-cb sram-allocation-strategy=minimum-arena sram-minimum-arena-search-limit=0})'

// expected-error @below {{'builtin.module' op minimum-arena search limit must be positive}}
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @zero_limit()
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
