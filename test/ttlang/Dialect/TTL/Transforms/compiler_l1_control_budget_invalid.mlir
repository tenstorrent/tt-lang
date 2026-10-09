// A control record alone can exceed the selected SRAM budget.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-sram l1-budget-override=7})'

// expected-error @below {{compiler-sram control records exceed the available L1 budget}}
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @control_record_exceeds_budget() attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>, ttl.noc_index = 0 : i32} {
    %storage = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
