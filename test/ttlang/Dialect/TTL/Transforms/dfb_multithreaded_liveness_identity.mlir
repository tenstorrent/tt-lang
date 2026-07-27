// Tests that explicit and legacy fallback logical DFB identities cannot
// collide.
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// The explicit DFB uses logical id 0 at provisional index 3. The unrelated
// untagged DFB uses provisional index 0 and must receive a distinct logical id.

// CHECK-LABEL: func.func @tagged_thread
// CHECK: ttl.bind_cb{cb_index = 0, {{.*}} {dfb_id = 0 : index}
// CHECK-LABEL: func.func @untagged_thread
// CHECK: ttl.bind_cb{cb_index = 1, {{.*}} {dfb_id = 1 : index}

module {
  func.func @tagged_thread()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.base_cta_index = 4 : i32, ttl.crta_indices = []} {
    %tagged = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %tagged : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %tagged : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited = ttl.cb_wait %tagged : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %tagged : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @untagged_thread()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 4 : i32, ttl.crta_indices = []} {
    %untagged = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved = ttl.cb_reserve %untagged : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %untagged : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited = ttl.cb_wait %untagged : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %untagged : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
