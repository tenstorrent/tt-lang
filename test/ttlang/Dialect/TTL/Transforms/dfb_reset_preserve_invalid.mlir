// RUN: ttlang-opt %s --verify-diagnostics --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})'

module attributes {
  ttl.launch_grid = [1, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "inconsistent_preservation">} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "inconsistent_preservation">, <kind = data_movement, identity = "reader", operation = "inconsistent_preservation">, <kind = data_movement, identity = "writer", operation = "inconsistent_preservation">]> preserve %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "inconsistent_preservation">,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "inconsistent_preservation">, <kind = data_movement, identity = "reader", operation = "inconsistent_preservation">, <kind = data_movement, identity = "writer", operation = "inconsistent_preservation">]> preserve %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "inconsistent_preservation">,
                  ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{'ttl.reset_all_dfbs' op synchronized DFB reset participants must declare identical target sets}}
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "inconsistent_preservation">, <kind = data_movement, identity = "reader", operation = "inconsistent_preservation">, <kind = data_movement, identity = "writer", operation = "inconsistent_preservation">]>
    return
  }
}
