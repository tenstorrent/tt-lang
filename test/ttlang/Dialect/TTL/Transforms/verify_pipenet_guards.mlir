// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,ttl-erase-pipenet-scopes)' | FileCheck %s

// Summary: Verifies that ttl-verify-pipenet-guards accepts role-contained
// PipeNet work, and that ttl-erase-pipenet-scopes inlines and erases the
// `ttl.pipenet_scope` markers so downstream lowering sees a scope-free IR.

// A copy into a pipe is valid only on the source node. A copy out of a pipe is
// valid only on destination nodes. Existing ttl.if_src/ttl.if_dst regions
// provide those execution domains.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @copy_roles_valid
  // CHECK: %[[PIPE0:.*]] = ttl.create_pipe
  // CHECK: %[[CB0:.*]] = ttl.bind_cb
  // CHECK: ttl.if_src %[[PIPE0]]
  // CHECK: ttl.copy %[[CB0]], %[[PIPE0]]
  // CHECK: ttl.if_dst %[[PIPE0]]
  // CHECK: %[[RECV0:.*]] = ttl.cb_reserve %[[CB0]]
  // CHECK: ttl.copy %[[PIPE0]], %[[RECV0]]
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
      %recv_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
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
  // CHECK-NOT: ttl.pipenet_scope
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

// `getCBIndex` traces through a chain of `unrealized_conversion_cast` ops
// before reaching the `bind_cb`. Without iteration, only the first cast
// is followed and the producer domain is silently lost, causing a false
// "no other thread fills" diagnostic on the consumer's `cb_wait`.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @producer_chained_cb_casts
  // CHECK-LABEL: func.func @consumer_chained_cb_casts
  func.func @producer_chained_cb_casts() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 5, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cast1 = builtin.unrealized_conversion_cast %cb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
        to !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cast2 = builtin.unrealized_conversion_cast %cast1
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
        to !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      ttl.cb_push %cast2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @consumer_chained_cb_casts() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 5, block_count = 2}
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
  // CHECK: ttl.copy
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
  // CHECK: %[[RECV1:.*]] = ttl.cb_reserve
  // CHECK: ttl.copy {{.*}}, %[[RECV1]]
  func.func @is_dst_predicate() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cond = ttl.is_dst {pipe_net_id = 0 : i64}
    scf.if %cond {
      %recv_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
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
  // CHECK: %[[RECV_A:.*]] = ttl.cb_reserve
  // CHECK: ttl.copy {{.*}}, %[[RECV_A]]
  // CHECK: %[[RECV_B:.*]] = ttl.cb_reserve
  // CHECK: ttl.copy {{.*}}, %[[RECV_B]]
  func.func @two_pipenets_disjoint() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pa = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 3) net 0
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>
    %pb = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 1
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 1>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pa : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0> {
      %recv_a_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %ra = ttl.copy %pa, %recv_a_dst
          : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
    }
    ttl.if_dst %pb : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 1> {
      %recv_b_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %rb = ttl.copy %pb, %recv_b_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
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
  // CHECK: scf.for
  // CHECK: ttl.copy
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
  // CHECK: affine.if
  // CHECK: ttl.copy
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
  // CHECK-NOT: ttl.pipenet_scope
  // CHECK: return
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
  // CHECK: ttl.is_active
  // CHECK: ttl.is_active
  // CHECK: ttl.copy
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

// -----

// `arith.cmpi ne`: the guard `x != 1` is true on coords {0}, which is the
// pipe source.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @cmpi_ne_guard
  // CHECK: arith.cmpi ne
  // CHECK: ttl.copy
  func.func @cmpi_ne_guard() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %c1 = arith.constant 1 : index
    %cond = arith.cmpi ne, %x, %c1 : index
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

// `arith.cmpi sle`: `x <= 0` is true on {0}.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @cmpi_sle_guard
  // CHECK: arith.cmpi sle
  // CHECK: ttl.copy
  func.func @cmpi_sle_guard() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %cond = arith.cmpi sle, %x, %c0 : index
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

// `arith.cmpi sgt`: `x > 0` is true on dst nodes.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @cmpi_sgt_guard
  // CHECK: arith.cmpi sgt
  // CHECK: %[[RECV2:.*]] = ttl.cb_reserve
  // CHECK: ttl.copy {{.*}}, %[[RECV2]]
  func.func @cmpi_sgt_guard() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %cond = arith.cmpi sgt, %x, %c0 : index
    scf.if %cond {
      %recv_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// `arith.cmpi sge`: `x >= 1` is true on dst nodes.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @cmpi_sge_guard
  // CHECK: arith.cmpi sge
  // CHECK: %[[RECV3:.*]] = ttl.cb_reserve
  // CHECK: ttl.copy {{.*}}, %[[RECV3]]
  func.func @cmpi_sge_guard() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %c1 = arith.constant 1 : index
    %cond = arith.cmpi sge, %x, %c1 : index
    scf.if %cond {
      %recv_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// Conjunction of two coord predicates narrows by intersection.

module attributes {ttl.launch_grid = [4 : i64, 4 : i64]} {
  // CHECK-LABEL: func.func @andi_two_coords
  // CHECK: arith.andi
  // CHECK: ttl.copy
  func.func @andi_two_coords() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %y = ttl.core_y : index
    %c0 = arith.constant 0 : index
    %x_eq = arith.cmpi eq, %x, %c0 : index
    %y_eq = arith.cmpi eq, %y, %c0 : index
    %cond = arith.andi %x_eq, %y_eq : i1
    scf.if %cond {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Disjunction over two coord predicates: each clause covers a distinct
// pipe role; the joint then-domain covers both src and dst.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @ori_two_coords
  // CHECK: arith.ori
  // CHECK: ttl.is_src
  // CHECK: ttl.copy
  func.func @ori_two_coords() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %is_a = arith.cmpi eq, %x, %c0 : index
    %is_b = arith.cmpi eq, %x, %c1 : index
    %cond = arith.ori %is_a, %is_b : i1
    // The disjunction covers all launched nodes; the inner scf.if then-region
    // domain equals the launch grid. A copy needs a tighter guard nested
    // below; here we exercise only that the disjunction itself parses and
    // does not narrow incorrectly.
    scf.if %cond {
      %inner = ttl.is_src {pipe_net_id = 0 : i64}
      scf.if %inner {
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

// `arith.xori`: when both operands are coord-comparisons, the predicate
// reads as "exactly one of A, B is true". Here `x == 0` xor `false` reduces
// to `x == 0`, the source.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @xori_guard
  // CHECK: arith.xori
  // CHECK: ttl.is_src
  // CHECK: ttl.copy
  func.func @xori_guard() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %is_a = arith.cmpi eq, %x, %c0 : index
    %is_b = arith.cmpi eq, %x, %c1 : index
    %cond = arith.xori %is_a, %is_b : i1
    // A xor B is true on the symmetric difference; here that's all coords
    // {0, 1}. As with the disjunction case, narrow further before the copy.
    scf.if %cond {
      %inner = ttl.is_src {pipe_net_id = 0 : i64}
      scf.if %inner {
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

// `arith.subi` and `arith.index_cast` inside the predicate are both
// recognized by `evalIndex`.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @evalindex_subi_indexcast
  // CHECK: arith.index_cast
  // CHECK: arith.subi
  // CHECK: ttl.copy
  func.func @evalindex_subi_indexcast() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %x_i32 = arith.index_cast %x : index to i32
    %c1_i32 = arith.constant 1 : i32
    %diff = arith.subi %c1_i32, %x_i32 : i32
    %c1_again = arith.constant 1 : i32
    %cond = arith.cmpi eq, %diff, %c1_again : i32
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

// `affine.if` constraint with `Mul`: `2 * d0 == 0` is true at `d0 = 0`.

#mulSet = affine_set<(d0) : (2 * d0 == 0)>
module attributes {ttl.launch_grid = [4 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_mul
  // CHECK: affine.if
  // CHECK: ttl.copy
  func.func @affine_if_mul() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #mulSet(%x) {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `affine.if` constraint with `Mod`: `d0 mod 4 == 0` is true on {0} only,
// matching the pipe source. The constraint's precision matters: a broken
// Mod evaluator that always returned 0 would widen to {0..3}, where coord
// {1, 2, 3} are outside src and the copy would be rejected.

#modSet = affine_set<(d0) : (d0 mod 4 == 0)>
module attributes {ttl.launch_grid = [4 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_mod
  // CHECK: affine.if
  // CHECK: ttl.copy
  func.func @affine_if_mod() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #modSet(%x) {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `affine.if` constraint with `FloorDiv` (non-zero divisor): `d0 floordiv 2 == 0`
// is true on coords {0, 1}. The pipe source is at coord 0; nest a tighter
// guard before the source copy.

#floordivSet = affine_set<(d0) : (d0 floordiv 2 == 0)>
module attributes {ttl.launch_grid = [4 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_floordiv
  // CHECK: affine.if
  // CHECK: arith.cmpi
  // CHECK: ttl.copy
  func.func @affine_if_floordiv() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #floordivSet(%x) {
      %c0 = arith.constant 0 : index
      %is_src = arith.cmpi eq, %x, %c0 : index
      scf.if %is_src {
        %send = ttl.copy %cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>)
            -> !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// `affine.if` constraint with `CeilDiv`: `d0 ceildiv 2 == 0` is true on {0}.

#ceildivSet = affine_set<(d0) : (d0 ceildiv 2 == 0)>
module attributes {ttl.launch_grid = [4 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_ceildiv
  // CHECK: affine.if
  // CHECK: ttl.copy
  func.func @affine_if_ceildiv() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #ceildivSet(%x) {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `affine.if` with an `AffineSymbolExpr` operand. The symbol resolves to a
// runtime constant; per-coord evaluation needs the symbol value alongside
// the dim value. Here `s0 = 0`, so `d0 - s0 == 0` reduces to `d0 == 0`.

#symSet = affine_set<(d0)[s0] : (d0 - s0 == 0)>
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_symbol
  // CHECK: affine.if
  // CHECK: ttl.copy
  func.func @affine_if_symbol() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %s0 = arith.constant 0 : index
    affine.if #symSet(%x)[%s0] {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Multi-block kernel-thread function. The DataFlow framework propagates the
// domain across `cf.cond_br` / `cf.br` via `BranchOpInterface` without any
// special-case verifier code; the user guard at the head of bb1 still covers
// the pipe-coupled op there.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @multi_block
  // CHECK: cf.cond_br
  // CHECK: ttl.is_src
  // CHECK: ttl.copy
  func.func @multi_block(%flag: i1) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    cf.cond_br %flag, ^bb1, ^bb2
  ^bb1:
    %cond = ttl.is_src {pipe_net_id = 0 : i64}
    scf.if %cond {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    cf.br ^bb2
  ^bb2:
    func.return
  }
}

// -----

// Mixed-predicate guard: `is_src(net) and core_y == 0`. The first operand
// resolves through `PipeNetPredicateOpInterface`; the second through per-coord
// `arith.cmpi` evaluation. The conjunction's then-domain is the intersection
// of {(0,0)} (src) and {(x,0) for x in 0..2}, which is {(0,0)}.

module attributes {ttl.launch_grid = [3 : i64, 2 : i64]} {
  // CHECK-LABEL: func.func @mixed_predicate_andi
  // CHECK: ttl.is_src
  // CHECK: arith.andi
  // CHECK: ttl.copy
  func.func @mixed_predicate_andi() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %is_src = ttl.is_src {pipe_net_id = 0 : i64}
    %y = ttl.core_y : index
    %c0 = arith.constant 0 : index
    %y_eq = arith.cmpi eq, %y, %c0 : index
    %cond = arith.andi %is_src, %y_eq : i1
    scf.if %cond {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `andi` else-branch domain: in `if a and b`, the else covers
// `a.else ∪ (a.then ∩ b.else)`. Picking `a = (x == 0)`, `b = (y == 0)` on
// a 2x2 grid yields then = {(0,0)} (src), else = {(0,1), (1,0), (1,1)}.
// A loopback pipe (dst spans the entire grid) covers both branches: the
// else-side pipe receive must validate against the else-domain being a subset
// of the pipe destination.

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  // CHECK-LABEL: func.func @andi_else_domain
  // CHECK: arith.andi
  // CHECK: ttl.copy
  // CHECK: %[[RECV4:.*]] = ttl.cb_reserve
  // CHECK: ttl.copy {{.*}}, %[[RECV4]]
  func.func @andi_else_domain() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 1) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 1) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %y = ttl.core_y : index
    %c0 = arith.constant 0 : index
    %x_eq = arith.cmpi eq, %x, %c0 : index
    %y_eq = arith.cmpi eq, %y, %c0 : index
    %is_origin = arith.andi %x_eq, %y_eq : i1
    scf.if %is_origin {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(1, 1) net 0>)
          -> !ttl.transfer_handle<write>
    } else {
      %recv_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 1) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// `affine.if` else-branch: pipe-coupled op in the else region must validate
// against the negation of the IntegerSet domain. The set covers {0} (src);
// the else region runs on {1} (dst), where a pipe-to-DFB copy is valid.

#srcOnly = affine_set<(d0) : (d0 == 0)>
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_else_branch
  // CHECK: affine.if
  // CHECK: ttl.copy
  // CHECK: %[[RECV5:.*]] = ttl.cb_reserve
  // CHECK: ttl.copy {{.*}}, %[[RECV5]]
  func.func @affine_if_else_branch() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #srcOnly(%x) {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    } else {
      %recv_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// `scf.while` adds no predicate to its body. The user guard outside still
// covers the pipe-coupled op inside the after-region.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @scf_while_no_predicate
  // CHECK: scf.while
  // CHECK: ttl.copy
  func.func @scf_while_no_predicate() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %cond = ttl.is_src {pipe_net_id = 0 : i64}
    scf.if %cond {
      %final = scf.while (%i = %c0) : (index) -> index {
        %lt = arith.cmpi slt, %i, %c4 : index
        scf.condition(%lt) %i : index
      } do {
      ^bb0(%i: index):
        %send = ttl.copy %cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        %next = arith.addi %i, %c1 : index
        scf.yield %next : index
      }
    }
    func.return
  }
}

// -----

// `scf.execute_region` adds no predicate. User guard outside still covers.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @scf_execute_region_no_predicate
  // CHECK: scf.execute_region
  // CHECK: ttl.copy
  func.func @scf_execute_region_no_predicate() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cond = ttl.is_src {pipe_net_id = 0 : i64}
    scf.if %cond {
      scf.execute_region {
        %send = ttl.copy %cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        scf.yield
      }
    }
    func.return
  }
}

// -----

// `scf.while` at function top level, with the pipe-coupled op behind an
// `if_src` inside the body.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @scf_while_top_level
  // CHECK: scf.while
  // CHECK: ttl.if_src
  // CHECK: ttl.copy
  func.func @scf_while_top_level() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %r = scf.while (%i = %c0) : (index) -> index {
      %lt = arith.cmpi slt, %i, %c4 : index
      scf.condition(%lt) %i : index
    } do {
    ^bb0(%i: index):
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %send = ttl.copy %cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
      }
      %next = arith.addi %i, %c1 : index
      scf.yield %next : index
    }
    func.return
  }
}

// -----

// `affine.for` adds no predicate. User guard outside still covers.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_for_no_predicate
  // CHECK: affine.for
  // CHECK: ttl.copy
  func.func @affine_for_no_predicate() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cond = ttl.is_src {pipe_net_id = 0 : i64}
    scf.if %cond {
      affine.for %i = 0 to 4 {
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

// Caller narrows with `if_src` then `func.call`s a helper whose body
// performs the `ttl.copy(cb, pipe)`. Helper relies on the caller-side
// guard flowing through the call edge.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func private @send_helper
  // CHECK: ttl.copy
  // CHECK-LABEL: func.func @kernel_caller_guards
  // CHECK: ttl.if_src
  // CHECK: func.call @send_helper
  func.func private @send_helper(%cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                                  %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    %x = ttl.copy %cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }

  func.func @kernel_caller_guards() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      func.call @send_helper(%cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    }
    func.return
  }
}

// -----

// `affine.if` constraint with a pure inequality (`>= 0`): `d0 - 1 >= 0`
// narrows to {1, 2}, matching the destination range. Exercises the
// inequality branch of `set.isEq(i) ? v != 0 : v < 0` on a constraint that
// the verifier should accept.

#dstRange = affine_set<(d0) : (d0 - 1 >= 0)>
module attributes {ttl.launch_grid = [3 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_inequality
  // CHECK: affine.if
  // CHECK: %[[RECV6:.*]] = ttl.cb_reserve
  // CHECK: ttl.copy {{.*}}, %[[RECV6]]
  func.func @affine_if_inequality() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #dstRange(%x) {
      %recv_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// `affine.if` IntegerSet with two AND'd constraints: `d0 >= 0` and
// `0 - d0 >= 0` together imply `d0 == 0`. Exercises the per-coord
// constraint loop that breaks on the first failing constraint.

#twoConstraints = affine_set<(d0) : (d0 >= 0, 0 - d0 >= 0)>
module attributes {ttl.launch_grid = [4 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_multi_constraint
  // CHECK: affine.if
  // CHECK: ttl.copy
  func.func @affine_if_multi_constraint() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #twoConstraints(%x) {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `affine.if` with a 2-D IntegerSet using both `core_x` and `core_y`. The
// constraint `(d0 == 0, d1 == 0)` narrows to {(0, 0)}, the pipe source.
// Exercises operand substitution for a multi-dimensional set on a non-1D
// launch grid.

#originSet = affine_set<(d0, d1) : (d0 == 0, d1 == 0)>
module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  // CHECK-LABEL: func.func @affine_if_multi_dim
  // CHECK: affine.if
  // CHECK: ttl.copy
  func.func @affine_if_multi_dim() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 1) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %y = ttl.core_y : index
    affine.if #originSet(%x, %y) {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `affine.if` else-branch where the predicate is an inequality, not an
// equality: `d0 - 1 >= 0` is true on {1, 2}, false on {0}. The else region
// runs on {0} (the pipe source) and validates the send; the then region
// runs on {1, 2} (the pipe destination range) and validates the receive.

#dstHalf = affine_set<(d0) : (d0 - 1 >= 0)>
module attributes {ttl.launch_grid = [3 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @affine_if_inequality_else
  // CHECK: affine.if
  // CHECK: %[[RECV7:.*]] = ttl.cb_reserve
  // CHECK: ttl.copy {{.*}}, %[[RECV7]]
  // CHECK: ttl.copy
  func.func @affine_if_inequality_else() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #dstHalf(%x) {
      %recv_dst = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_dst
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
    } else {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}
