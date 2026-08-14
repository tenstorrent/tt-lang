// Tests that DFB static-configuration analysis ignores non-compute kernels.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// A NOC-only module does not require a compute-target environment. Quasar
// allocation analysis therefore remains independent of unsupported compute
// configuration and launch APIs.

// CHECK-LABEL: func.func @noc_only
// CHECK: %[[DFB:.*]] = ttl.bind_cb{cb_index = 0, block_count = 1}

module attributes {ttl.target_arch = #ttcore.arch<quasar>} {
  func.func @noc_only()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 1 : i32,
                  ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
