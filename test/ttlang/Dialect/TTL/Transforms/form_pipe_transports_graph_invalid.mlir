// Summary: Verify PipeTransport formation rejects transfer schedules whose
// launch-node execution cannot be determined.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics \
// RUN:   --ttl-form-pipe-transports

// A runtime condition does not prove that the source executes this send.
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unanalyzable_send_guard(%offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    %core_x = ttl.core_x : index
    %sum = arith.addi %core_x, %offset : index
    %zero = arith.constant 0 : index
    %condition = arith.cmpi eq, %sum, %zero : index
    scf.if %condition {
      // expected-error @below {{cannot determine whether the pipe source executes this send}}
      %send = ttl.pipe_transfer.send %transfer, %source
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
