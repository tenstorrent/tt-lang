// Verifies exact compiler-managed SRAM placement reports bounded and proven failures precisely.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=exact l1-exact-allocation-search-limit=1})'

// A bounded search reports its feasible incumbent without claiming optimality.
// expected-error @below {{'builtin.module' op compiler-l1 exact allocation examined 1 work items and reached the 1-item limit after finding a feasible 22592-byte arena without proving it is minimal}}
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @search_limit()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32} {
    %storage_0 = ttl.bind_cb {cb_index = 0, block_count = 3} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %storage_1 = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %storage_2 = ttl.bind_cb {cb_index = 2, block_count = 4} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %storage_3 = ttl.bind_cb {cb_index = 3, block_count = 4} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %produced_2 = ttl.cb_reserve %storage_2
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %produced_3 = ttl.cb_reserve %storage_3
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %consumed_2 = ttl.cb_wait %storage_2
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_2 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %produced_0 = ttl.cb_reserve %storage_0
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %produced_1 = ttl.cb_reserve %storage_1
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %consumed_0 = ttl.cb_wait %storage_0
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_0 : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %consumed_1 = ttl.cb_wait %storage_1
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_1 : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %consumed_3 = ttl.cb_wait %storage_3
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %storage_3 : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    return
  }
}
