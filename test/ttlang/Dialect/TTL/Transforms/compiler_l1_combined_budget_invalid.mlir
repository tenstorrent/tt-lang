// Verifies conversion counts reset scratch with the compiler-managed arena.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-sram reuse-user-dfbs=true l1-budget-override=2112},ttl-validate-cb-budget{l1-budget-override=2112},convert-ttl-to-ttkernel{l1-budget-override=2112})'

// expected-error @below {{combined DFB and runtime resources require 2176 L1 bytes but the budget is 2112}}
module attributes {
  ttl.launch_grid = [1, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_budget">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %slot = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_budget">, <kind = data_movement, identity = "reader", operation = "reset_budget">, <kind = data_movement, identity = "writer", operation = "reset_budget">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_budget">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_budget">, <kind = data_movement, identity = "reader", operation = "reset_budget">, <kind = data_movement, identity = "writer", operation = "reset_budget">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_budget">,
                  ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_budget">, <kind = data_movement, identity = "reader", operation = "reset_budget">, <kind = data_movement, identity = "writer", operation = "reset_budget">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}
