// RUN: ttlang-opt %s --split-input-file -ttl-verify-pipenet-guards | FileCheck %s

// Summary: Verifies that ttl-verify-pipenet-guards does not report a false
// wait-for cycle when pipe events appear in mutually exclusive scf.if branches.

// -----

// A runtime-flag scf.if has a locally valid loopback schedule in each branch.
// No ProgramOrder edge should cross sibling branches, so no cycle is formed.
//
// CHECK-LABEL: func.func @if_branch_no_false_cycle
// CHECK-NOT: pipe schedule contains a wait-for cycle
// CHECK-NOT: receive wait occurs before the send that completes it
// CHECK-NOT: pipe send occurs before the receiver publishes a destination address

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @if_branch_no_false_cycle(%runtime_flag: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.if %runtime_flag {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.wait %recv : !ttl.transfer_handle
    } else {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.wait %recv : !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// Nested scf.if: the fix extends to multiple nesting levels.
//
// CHECK-LABEL: func.func @nested_if_no_false_cycle
// CHECK-NOT: pipe schedule contains a wait-for cycle
// CHECK-NOT: receive wait occurs before the send that completes it
// CHECK-NOT: pipe send occurs before the receiver publishes a destination address

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @nested_if_no_false_cycle(%flag_a: i1, %flag_b: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.if %flag_a {
      scf.if %flag_b {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %recv = ttl.copy %pipe, %recv_view
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.transfer_handle
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.wait %recv : !ttl.transfer_handle
      } else {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %recv = ttl.copy %pipe, %recv_view
            : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.transfer_handle
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.wait %recv : !ttl.transfer_handle
      }
    }
    func.return
  }
}
