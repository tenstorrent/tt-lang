// Tests conservative DFB reuse across concurrently executing kernel functions.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=DEFAULT

// An acknowledgment DFB orders both thread frontiers between A and B. A and B
// may share physical index 0; the acknowledgment requires physical index 1.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 1 : i32, {{.*}}}]}
// REUSE-LABEL: func.func @synchronized_dm
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: %[[DM_A:.*]] = ttl.bind_cb{cb_index = 0,
// REUSE: %[[DM_ACK:.*]] = ttl.bind_cb{cb_index = 1,
// REUSE: %[[DM_B:.*]] = ttl.bind_cb{cb_index = 0,
// REUSE: ttl.cb_reserve %[[DM_A]]
// REUSE: ttl.cb_push %[[DM_A]]
// REUSE: ttl.cb_wait %[[DM_ACK]]
// REUSE: ttl.cb_pop %[[DM_ACK]]
// REUSE: ttl.cb_reserve %[[DM_B]]
// REUSE: ttl.cb_push %[[DM_B]]
// REUSE-LABEL: func.func @synchronized_compute
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: %[[COMPUTE_A:.*]] = ttl.bind_cb{cb_index = 0,
// REUSE: %[[COMPUTE_ACK:.*]] = ttl.bind_cb{cb_index = 1,
// REUSE: %[[COMPUTE_B:.*]] = ttl.bind_cb{cb_index = 0,

// DEFAULT-NOT: ttl.dfb_allocations
// DEFAULT-LABEL: func.func @synchronized_dm
// DEFAULT-SAME: ttl.base_cta_index = 3 : i32
// DEFAULT: ttl.bind_cb{cb_index = 0,
// DEFAULT: ttl.bind_cb{cb_index = 1,
// DEFAULT: ttl.bind_cb{cb_index = 2,
// DEFAULT-LABEL: func.func @synchronized_compute
// DEFAULT-SAME: ttl.base_cta_index = 3 : i32
// DEFAULT: ttl.bind_cb{cb_index = 0,
// DEFAULT: ttl.bind_cb{cb_index = 1,
// DEFAULT: ttl.bind_cb{cb_index = 2,

func.func @synchronized_dm()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack_view = ttl.cb_wait %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_view = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @synchronized_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack_view = ttl.cb_reserve %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_view = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Source order in both threads does not order A's consumer completion before
// B's producer entry. The two DFBs must retain separate physical indices.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 1 : i32, {{.*}}}]}
// REUSE-LABEL: func.func @unordered_dm
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 0,
// REUSE: ttl.bind_cb{cb_index = 1,
// REUSE-LABEL: func.func @unordered_compute
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 0,
// REUSE: ttl.bind_cb{cb_index = 1,

func.func @unordered_dm()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_view = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @unordered_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_view = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}
