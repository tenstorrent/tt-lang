// The post-allocation validator includes control state and honors its own override.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1},ttl-validate-cb-budget{l1-budget-override=2111})'
// expected-error @below {{combined DFB and runtime resources require 2112 L1 bytes but the budget is 2111}}
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @boundary() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
