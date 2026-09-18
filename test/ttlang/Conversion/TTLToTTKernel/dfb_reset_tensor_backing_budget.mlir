// Verifies reset scratch does not double-count tensor-backed DFB storage.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false l1-budget-override=64})' -o /dev/null

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @tensor_backed_reset()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
    return
  }
}
