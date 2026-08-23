// Tests external-call DFB uses in concurrent-kernel lifetime analysis.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// A descriptor reference keeps A live through the external call without a
// runtime DFB argument. The unsummarized call may change A's protocol state,
// so A remains unbounded and cannot share with B. The index query after the pop
// does not access physical storage and does not extend A's lifetime.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}, {{.*}}dfb_index = 1 : i32{{.*}}, {{.*}}dfb_index = 2 : i32{{.*}}]}
// CHECK-LABEL: func.func @external_use_before_release_dm
// CHECK-SAME: ttl.base_cta_index = 3 : i32
// CHECK-DAG: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-DAG: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// CHECK-DAG: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index}
// CHECK-LABEL: func.func @external_use_before_release_compute
// CHECK-SAME: ttl.base_cta_index = 3 : i32
// CHECK-DAG: %[[A:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-DAG: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// CHECK-DAG: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index}
// CHECK: ttl.opaque_call "inspect_dfb" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%[[A]] : !ttl.cb<{{.*}}>) () {header = "inspect_dfb.hpp"}
// CHECK: ttl.cb_pop %[[A]]
// CHECK: ttl.get_dfb_id %[[A]]

func.func @external_use_before_release_dm()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %a_view = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_push %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %ack_view = ttl.cb_wait %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_pop %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b_view = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  return
}

func.func @external_use_before_release_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %a_view = ttl.cb_wait %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.opaque_call "inspect_dfb" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%a : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) () {header = "inspect_dfb.hpp"} : () -> ()
  ttl.cb_pop %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %a_index = ttl.get_dfb_id %a : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %ack_view = ttl.cb_reserve %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_push %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b_view = ttl.cb_wait %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  return
}

// -----

// A direct DFB operand after A's pop is an access after the proposed terminal
// event. A is unbounded and cannot share a physical index with B.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}, {{.*}}dfb_index = 1 : i32{{.*}}, {{.*}}dfb_index = 2 : i32{{.*}}]}
// CHECK-LABEL: func.func @external_use_after_release_dm
// CHECK-SAME: ttl.base_cta_index = 3 : i32
// CHECK-DAG: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-DAG: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// CHECK-DAG: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index}
// CHECK-LABEL: func.func @external_use_after_release_compute
// CHECK-SAME: ttl.base_cta_index = 3 : i32
// CHECK-DAG: %[[A:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-DAG: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// CHECK-DAG: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index}
// CHECK: ttl.cb_pop %[[A]]
// CHECK: ttl.opaque_call "inspect_dfb" (%[[A]]) {header = "inspect_dfb.hpp"}

func.func @external_use_after_release_dm()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %a_view = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_push %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %ack_view = ttl.cb_wait %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_pop %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b_view = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  return
}

func.func @external_use_after_release_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %a_view = ttl.cb_wait %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_pop %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  ttl.opaque_call "inspect_dfb" (%a) {header = "inspect_dfb.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>) -> ()
  %ack_view = ttl.cb_reserve %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_push %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b_view = ttl.cb_wait %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  return
}
