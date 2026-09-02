// Tests conservative DFB reuse across concurrently executing kernel functions.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=DISABLED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true},ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=REPEAT

// The acknowledgment DFB orders B's reserve and wait after A's pop. A and B may
// share physical index 0; the acknowledgment requires physical index 1.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 0 : i32}, {block_count = 2 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 1 : i32}]}
// REUSE-LABEL: func.func @synchronized_dm
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: %[[DM_A:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE: %[[DM_ACK:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// REUSE: %[[DM_B:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}
// REUSE: ttl.cb_reserve %[[DM_A]]
// REUSE: ttl.cb_push %[[DM_A]]
// REUSE: ttl.cb_wait %[[DM_ACK]]
// REUSE: ttl.cb_pop %[[DM_ACK]]
// REUSE: ttl.cb_reserve %[[DM_B]]
// REUSE: ttl.cb_push %[[DM_B]]
// REUSE-LABEL: func.func @synchronized_compute
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: %[[COMPUTE_A:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE: %[[COMPUTE_ACK:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// REUSE: %[[COMPUTE_B:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}

// A repeated finalization preserves logical identities after A and B receive
// the same physical index.
// REPEAT-LABEL: func.func @synchronized_dm
// REPEAT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REPEAT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// REPEAT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}
// REPEAT-LABEL: func.func @synchronized_compute
// REPEAT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REPEAT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}
// REPEAT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}

// DISABLED: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 0 : i32}, {block_count = 2 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 1 : i32}, {block_count = 2 : i32, dfb_index = 2 : i32, element_type = !ttcore.tile<1x16, bf16>, num_tiles = 1 : i32, page_size = 32 : i32, storage_index = 2 : i32}]}
// DISABLED-LABEL: func.func @synchronized_dm
// DISABLED-SAME: ttl.base_cta_index = 3 : i32
// DISABLED: ttl.bind_cb{cb_index = 0,
// DISABLED: ttl.bind_cb{cb_index = 1,
// DISABLED: ttl.bind_cb{cb_index = 2,
// DISABLED-LABEL: func.func @synchronized_compute
// DISABLED-SAME: ttl.base_cta_index = 3 : i32
// DISABLED: ttl.bind_cb{cb_index = 0,
// DISABLED: ttl.bind_cb{cb_index = 1,
// DISABLED: ttl.bind_cb{cb_index = 2,

func.func @synchronized_dm()
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

func.func @synchronized_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %a_view = ttl.cb_wait %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_pop %a : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %ack_view = ttl.cb_reserve %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_push %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  %b_view = ttl.cb_wait %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2> -> tensor<1x1x!ttcore.tile<1x16, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
  return
}

// -----

// A relay through the second data-movement kernel orders the second DFB's
// reserve and wait after the first DFB's pop. Both DFBs use the same producer
// and consumer kernels and may share physical index 0.

// REUSE-LABEL: func.func @three_kernel_producer
// REUSE-SAME: ttl.base_cta_index = 3 : i32
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 2 : index}
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 3 : index}
// REUSE-LABEL: func.func @three_kernel_consumer
// REUSE-SAME: ttl.base_cta_index = 3 : i32
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 1 : index}
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 3 : index}
// REUSE-LABEL: func.func @three_kernel_relay
// REUSE-SAME: ttl.base_cta_index = 3 : i32
// REUSE: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 1 : index}
// REUSE: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 2 : index}

func.func @three_kernel_producer()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 4 : i32, ttl.crta_indices = []} {
  %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %relay_to_producer = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_dfb = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_reserved = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %producer_ack = ttl.cb_wait %relay_to_producer : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %relay_to_producer : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_reserved = ttl.cb_reserve %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @three_kernel_consumer()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 4 : i32, ttl.crta_indices = []} {
  %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %consumer_to_relay = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_dfb = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %first_waited = ttl.cb_wait %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %relay_reserved = ttl.cb_reserve %consumer_to_relay : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %consumer_to_relay : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %second_waited = ttl.cb_wait %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @three_kernel_relay()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 4 : i32, ttl.crta_indices = []} {
  %consumer_to_relay = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %relay_to_producer = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %consumer_ack = ttl.cb_wait %consumer_to_relay : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %consumer_to_relay : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %producer_ack = ttl.cb_reserve %relay_to_producer : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %relay_to_producer : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Source order in both kernels does not order A's consumer completion before
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
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_view = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @unordered_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_view = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// B's consumer wait enters before A's terminal pop. The acknowledgment orders
// B's producer entry after A's pop, but does not order the wait entry after
// A's pop. A, the acknowledgment, and B require separate physical indices.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 1 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 2 : i32, {{.*}}}]}
// REUSE-LABEL: func.func @early_wait_producer
// REUSE-SAME: ttl.base_cta_index = 3 : i32
// REUSE: ttl.bind_cb{cb_index = 0,
// REUSE: ttl.bind_cb{cb_index = 1,
// REUSE: ttl.bind_cb{cb_index = 2,
// REUSE-LABEL: func.func @early_wait_compute
// REUSE-SAME: ttl.base_cta_index = 3 : i32
// REUSE-LABEL: func.func @early_wait_consumer
// REUSE-SAME: ttl.base_cta_index = 3 : i32

func.func @early_wait_producer()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack_view = ttl.cb_wait %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_view = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @early_wait_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack_view = ttl.cb_reserve %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @early_wait_consumer()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_view = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// A missing pop leaves A unbounded. Source order alone cannot permit A and B
// to share a physical index.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 1 : i32, {{.*}}}]}
// REUSE-LABEL: func.func @unbalanced_dm
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 0,
// REUSE: ttl.bind_cb{cb_index = 1,
// REUSE-LABEL: func.func @unbalanced_compute
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 0,
// REUSE: ttl.bind_cb{cb_index = 1,

func.func @unbalanced_dm()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_view = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @unbalanced_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_view = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_view = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Sequential DFBs with different exact types cannot share an index. Immutable
// declaration order determines the physical assignment, not logical numbering.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 1 : i32, {{.*}}}]}
// REUSE-LABEL: func.func @different_types
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 0, {{.*}}} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
// REUSE: ttl.bind_cb{cb_index = 1, {{.*}}} : <[1, 1], !ttcore.tile<32x32, f32>, 2>

func.func @different_types()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %b = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_producer = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a_consumer = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_pop %a : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %b_producer = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_consumer = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Declaration-only compiler-created DFBs are valid. They receive distinct
// identities even when their provisional cb_index values match.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 1 : i32, {{.*}}}]}
// REUSE-LABEL: func.func @first_compiler_dfb
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 0, {{.*}}} {dfb_id = 0 : index, ttl.compiler_allocated}
// REUSE-LABEL: func.func @second_compiler_dfb
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 1, {{.*}}} {dfb_id = 1 : index, ttl.compiler_allocated}

func.func @first_compiler_dfb()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
  %compiler_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @second_compiler_dfb()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
  %compiler_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Repeated lifecycle operations cannot be matched by static operation site, so
// the loop DFB remains unbounded even though B starts after the loop completes.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 1 : i32, {{.*}}}]}
// REUSE-LABEL: func.func @loop_lifecycle
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 0,
// REUSE: ttl.bind_cb{cb_index = 1,

func.func @loop_lifecycle()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %loop_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %lower_bound = arith.constant 0 : index
  %upper_bound = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %lower_bound to %upper_bound step %step {
    %producer_view = ttl.cb_reserve %loop_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %loop_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %consumer_view = ttl.cb_wait %loop_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %loop_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  %b_producer = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_consumer = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Ordered lifetimes cannot share when the producer pointer changes from NOC0
// to Pack. Lifecycle completion does not transfer write-pointer ownership.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 1 : i32, {{.*}}}]}
// REUSE-LABEL: func.func @producer_transition_dm
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 0, {{.*}}} {dfb_id = 0 : index}
// REUSE-LABEL: func.func @producer_transition_compute
// REUSE-SAME: ttl.base_cta_index = 2 : i32
// REUSE: ttl.bind_cb{cb_index = 0, {{.*}}} {dfb_id = 0 : index}
// REUSE: ttl.bind_cb{cb_index = 1, {{.*}}} {dfb_id = 1 : index, ttl.compiler_allocated}

func.func @producer_transition_dm()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %input = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_producer = ttl.cb_reserve %input : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %input : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @producer_transition_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %input = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %intermediate = ttl.bind_cb {cb_index = 1, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %input_consumer = ttl.cb_wait %input : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %input : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %intermediate_producer = ttl.cb_reserve %intermediate : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %intermediate : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %intermediate_consumer = ttl.cb_wait %intermediate : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %intermediate : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Ordered lifetimes cannot share when the consumer pointer changes from Unpack
// to NOC0. The acknowledgment orders B's reserve and wait after A's pop.

// REUSE: module attributes {ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 1 : i32, {{.*}}}, {block_count = 2 : i32, dfb_index = 2 : i32, {{.*}}}]}
// REUSE-LABEL: func.func @consumer_transition_compute
// REUSE-SAME: ttl.base_cta_index = 3 : i32
// REUSE: ttl.bind_cb{cb_index = 0, {{.*}}} {dfb_id = 0 : index}
// REUSE: ttl.bind_cb{cb_index = 1, {{.*}}} {dfb_id = 1 : index}
// REUSE: ttl.bind_cb{cb_index = 2, {{.*}}} {dfb_id = 2 : index}
// REUSE-LABEL: func.func @consumer_transition_dm
// REUSE-SAME: ttl.base_cta_index = 3 : i32
// REUSE: ttl.bind_cb{cb_index = 1, {{.*}}} {dfb_id = 1 : index}
// REUSE: ttl.bind_cb{cb_index = 2, {{.*}}} {dfb_id = 2 : index}

func.func @consumer_transition_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %a = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_producer = ttl.cb_reserve %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %a_consumer = ttl.cb_wait %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %a : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack_producer = ttl.cb_reserve %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_producer = ttl.cb_reserve %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @consumer_transition_dm()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                ttl.noc_index = 0 : i32,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ack_consumer = ttl.cb_wait %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %ack : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %b_consumer = ttl.cb_wait %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %b : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}
