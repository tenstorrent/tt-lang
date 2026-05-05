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
  // CHECK: return
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

// -----

// ttl.is_src is recognized structurally: the verifier doesn't fall back to
// per-node arith analysis.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @is_src_predicate
  // CHECK: ttl.is_src
  func.func @is_src_predicate() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cond = ttl.is_src {pipe_net_id = 0 : i64}
    scf.if %cond {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// ttl.is_dst recognition.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @is_dst_predicate
  // CHECK: ttl.is_dst
  func.func @is_dst_predicate() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cond = ttl.is_dst {pipe_net_id = 0 : i64}
    scf.if %cond {
      %recv = ttl.copy %pipe, %cb
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// Two PipeNets with disjoint active sets: each ttl.copy validates against
// its own pipe's role, not the union.

module attributes {ttl.launch_grid = [4 : i64, 4 : i64]} {
  // CHECK-LABEL: func.func @two_pipenets_disjoint
  func.func @two_pipenets_disjoint() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pa = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 3) net 0
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>
    %pb = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 1
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 1>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pa : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0> {
      %ra = ttl.copy %pa, %cb
          : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle
    }
    ttl.if_dst %pb : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 1> {
      %rb = ttl.copy %pb, %cb
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 1>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// Loops do not narrow the execution domain. A user guard outside an scf.for
// still covers a pipe-coupled op inside the loop body.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @scf_for_no_predicate
  func.func @scf_for_no_predicate() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %cond = ttl.is_src {pipe_net_id = 0 : i64}
    scf.if %cond {
      scf.for %i = %c0 to %c4 step %c1 {
        %send = ttl.copy %cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// affine.if user guard whose IntegerSet implies the source role.

#srcSet = affine_set<(d0) : (d0 == 0)>
module attributes {ttl.launch_grid = [4 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_guard
  func.func @affine_if_guard() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #srcSet(%x) {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// ttl.is_active in a scope spanning both src and dst roles.

module attributes {ttl.launch_grid = [4 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @is_active_scope
  // CHECK: ttl.is_active
  func.func @is_active_scope() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cond = ttl.is_active {pipe_net_id = 0 : i64}
    scf.if %cond {
      ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64, 0 : i64], ttl.pipe_net_roles = [0 : i64, 1 : i64]} {
      }
    }
    func.return
  }
}

// -----

// Nested `is_active` predicates intersect: the inner block runs only on
// nodes active in BOTH PipeNets. Used for relay-style threads that touch
// two nets in the same body.

module attributes {ttl.launch_grid = [4 : i64, 4 : i64]} {
  // CHECK-LABEL: func.func @nested_is_active_intersect
  func.func @nested_is_active_intersect() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pa = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 3) net 0
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>
    %pb = ttl.create_pipe src(0, 1) dst(1, 1) to(3, 1) net 1
        : !ttl.pipe<src(0, 1) dst(1, 1) to(3, 1) net 1>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %a_active = ttl.is_active {pipe_net_id = 0 : i64}
    scf.if %a_active {
      %b_active = ttl.is_active {pipe_net_id = 1 : i64}
      scf.if %b_active {
        // Reachable on the intersection: net_a active set is column 0 rows
        // 0..3, net_b active set is row 1 cols 0..3. Their intersection is
        // (0, 1), which is in net_b's source role.
        ttl.if_src %pb : !ttl.pipe<src(0, 1) dst(1, 1) to(3, 1) net 1> {
          %send = ttl.copy %cb, %pb
              : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                 !ttl.pipe<src(0, 1) dst(1, 1) to(3, 1) net 1>)
              -> !ttl.transfer_handle<write>
        }
      }
    }
    func.return
  }
}
