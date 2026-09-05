// Verifies diagnostics for invalid PipeNet destination-count operations.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

#records = #ttl.pipenet_records<net 7 name "mismatched" pipes [
  <srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
   dstEndX = 1, dstEndY = 0>
]>

// The operation and record table must identify the same PipeNet.
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @mismatched_pipe_net_id()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{PipeNet 7, but pipe_net_id is 3}}
    %count = ttl.pipenet_destination_count {
        pipe_net_id = 3 : i64, records = #records} : index
    func.return
  }
}
