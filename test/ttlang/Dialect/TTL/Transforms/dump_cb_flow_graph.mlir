// Verify that the CB flow dump keeps logical DFBs separate after physical
// index reuse.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-dump-cb-flow-graph{output=%t.json})'
// RUN: FileCheck %s --input-file=%t.json

// Distinct logical DFBs at one physical index require distinct JSON entries.

// CHECK-DAG: {"cb_index":0,"consumers":[],"dfb_id":10
// CHECK-DAG: {"cb_index":0,"consumers":[],"dfb_id":11
module attributes {
  ttl.dfb_allocations = [{block_count = 2 : i32, dfb_index = 0 : i32,
                          element_type = !ttcore.tile<32x32, bf16>,
                          num_tiles = 1 : i32, page_size = 2048 : i32}]
} {
  func.func @first_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }

  func.func @second_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %block = ttl.cb_reserve %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}
