// Summary: Verify dead-code elimination preserves DFB synchronization whose
// returned view is unused.
// RUN: ttlang-opt %s --canonicalize | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' | FileCheck %s

// An unused view does not make the blocking wait removable.
// CHECK-LABEL: func.func @preserve_unused_wait
// CHECK: %[[DFB:.*]] = ttl.bind_cb
// CHECK: %[[WAITED:.*]] = ttl.cb_wait %[[DFB]]
// CHECK-NEXT: ttl.cb_pop %[[DFB]]

module {
  func.func @preserve_unused_wait()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}
