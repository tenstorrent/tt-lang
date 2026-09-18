// Tests that unsafe allocation-group policy does not change ungrouped reuse.
// RUN: not ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})' --verify-diagnostics -o /dev/null

// Unknown launch-node membership remains the decisive conflict for ungrouped
// DFBs even when their backing differs. Storage precedence is specific to
// explicit groups because unsafe policy must not weaken that physical invariant.

// CHECK: DFB conflict lhs=0 rhs=1 reason=unknown-launch-node-domain node=none

module {
  func.func @ungrouped_unknown_domain()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    // expected-error @below {{tensor-backed physical DFB requires an exact non-empty launch-node domain}}
    %tensor_backed = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %scratch = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
