// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-guards | FileCheck %s

// Summary: Regression test for issue #630. ttl-verify-pipenet-guards must
// not propagate program-order edges across mutually exclusive scf.if
// branches and emit a false "receive wait occurs before the send that
// completes it" diagnostic.
//
// `--verify-diagnostics` with no expected-error annotations asserts the
// verifier emits zero diagnostics. The FileCheck capture-and-reuse of the
// transfer-handle SSA values pins the in-branch send-wait/recv-wait pairing
// so a regression that reordered or merged the per-branch schedules would
// fail the SSA-name reuse.

// -----

// A runtime-flag scf.if has a locally valid loopback schedule in each branch.
// Without the fix, the verifier reports a wait-for cycle by transitively
// ordering the then-branch recv wait before the else-branch send.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @if_branch_no_false_cycle
  // CHECK: scf.if %{{.*}} {
  // CHECK:   %[[T_RECV_A:.*]] = ttl.copy %{{.*}}, %{{.*}}{{.*}}-> !ttl.transfer_handle
  // CHECK:   %[[T_SEND_A:.*]] = ttl.copy %{{.*}}, %{{.*}}{{.*}}-> !ttl.transfer_handle<write>
  // CHECK:   ttl.wait %[[T_SEND_A]]
  // CHECK:   ttl.wait %[[T_RECV_A]]
  // CHECK: } else {
  // CHECK:   %[[T_RECV_B:.*]] = ttl.copy %{{.*}}, %{{.*}}{{.*}}-> !ttl.transfer_handle
  // CHECK:   %[[T_SEND_B:.*]] = ttl.copy %{{.*}}, %{{.*}}{{.*}}-> !ttl.transfer_handle<write>
  // CHECK:   ttl.wait %[[T_SEND_B]]
  // CHECK:   ttl.wait %[[T_RECV_B]]
  // CHECK: }
  func.func @if_branch_no_false_cycle(%runtime_flag: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "loopback_net"}
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

// Nested scf.if: the inner if-else carries the loopback pair in each
// branch. The branch-frontier rule must apply at every nesting level.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @nested_if_no_false_cycle
  // CHECK: scf.if %{{.*}} {
  // CHECK:   scf.if %{{.*}} {
  // CHECK:     %[[T_RECV_A:.*]] = ttl.copy %{{.*}}, %{{.*}}{{.*}}-> !ttl.transfer_handle
  // CHECK:     %[[T_SEND_A:.*]] = ttl.copy %{{.*}}, %{{.*}}{{.*}}-> !ttl.transfer_handle<write>
  // CHECK:     ttl.wait %[[T_SEND_A]]
  // CHECK:     ttl.wait %[[T_RECV_A]]
  // CHECK:   } else {
  // CHECK:     %[[T_RECV_B:.*]] = ttl.copy %{{.*}}, %{{.*}}{{.*}}-> !ttl.transfer_handle
  // CHECK:     %[[T_SEND_B:.*]] = ttl.copy %{{.*}}, %{{.*}}{{.*}}-> !ttl.transfer_handle<write>
  // CHECK:     ttl.wait %[[T_SEND_B]]
  // CHECK:     ttl.wait %[[T_RECV_B]]
  // CHECK:   }
  // CHECK: }
  func.func @nested_if_no_false_cycle(%flag_a: i1, %flag_b: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "loopback_net"}
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
