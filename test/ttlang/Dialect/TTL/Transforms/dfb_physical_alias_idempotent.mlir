// Tests preservation of a proven physical DFB alias across finalization.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true},ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s

// A proven alias remains valid when a second finalization runs with user reuse
// disabled.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}, {{.*}}dfb_index = 1 : i32{{.*}}]}
// CHECK-LABEL: func.func @safe_alias_producer
// CHECK: ttl.bind_cb{cb_index = [[DATA_INDEX:[0-9]+]], block_count = 2} {dfb_id = 0 : index}
// CHECK: ttl.bind_cb{cb_index = [[ACK_INDEX:[0-9]+]], block_count = 2} {dfb_id = 1 : index}
// CHECK: ttl.bind_cb{cb_index = [[DATA_INDEX]], block_count = 2} {dfb_id = 2 : index}

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @safe_alias_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_block = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack_block = ttl.cb_wait %ack
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_block = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }

  func.func @safe_alias_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_block = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %ack_block = ttl.cb_reserve %ack
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %ack : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_block = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}
