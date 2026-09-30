// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @duplicate_preserved_dfb()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{'ttl.reset_all_dfbs' op preserved DFBs must be distinct}}
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "duplicate_preserved_dfb">, <kind = data_movement, identity = "reader", operation = "duplicate_preserved_dfb">, <kind = data_movement, identity = "writer", operation = "duplicate_preserved_dfb">]> preserve %dfb, %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
