// RUN: ttlang-opt %s --split-input-file -ttl-verify-pipenet-guards | FileCheck %s

// Summary: Verifies that ttl-verify-pipenet-guards accepts role-contained
// PipeNet work and erases ttl.pipenet_scope after successful verification.

// A copy into a pipe is valid only on the source node. A copy out of a pipe is
// valid only on destination nodes. Existing ttl.if_src/ttl.if_dst regions
// provide those execution domains.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @copy_roles_valid
  // CHECK: ttl.copy
  // CHECK: ttl.copy
  func.func @copy_roles_valid() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.copy %pipe, %cb
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// A ttl.pipenet_scope is accepted when the surrounding predicate is contained
// in the declared role domain. The verifier erases the scope.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @scope_erased_after_verification
  // CHECK-NOT: ttl.pipenet_scope
  // CHECK: ttl.if_src
  // CHECK: func.return
  func.func @scope_erased_after_verification() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %x = ttl.core_x : index
    %c1 = arith.constant 1 : index
    %is_src = arith.cmpi slt, %x, %c1 : index
    scf.if %is_src {
      ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64], ttl.pipe_net_roles = [0 : i64]} {
        ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        }
      }
    }
    func.return
  }
}

// -----

// DFB waits are accepted when every waiting node is covered by a producer
// domain for the same DFB index.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @producer
  // CHECK: ttl.cb_push
  // CHECK-LABEL: func.func @consumer
  // CHECK: ttl.cb_wait
  func.func @producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 4, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 4, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %view = ttl.cb_wait %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    func.return
  }
}
