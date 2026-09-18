// Tests allocation reporting when launch-node-dependent IR has no launch grid.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: DFB allocation liveness report
// CHECK: DFB logical_id=0 bounded=0 compiler_created=0
// CHECK-SAME: domain=unknown
// CHECK: access 0 effect=reserve tiles=1 sequence=0 domain=unknown
// CHECK: access 3 effect=pop tiles=1 sequence=0 domain=unknown
// CHECK-NOT: {{^  node }}
// CHECK-NOT: {{^  diagnostic_nodes }}
// CHECK: DFB allocation liveness report end

module {
  func.func @node_dependent_kernel()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %reserved = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %available = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}
