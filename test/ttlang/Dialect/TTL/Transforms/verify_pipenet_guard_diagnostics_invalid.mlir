// RUN: not ttlang-opt %s --split-input-file -ttl-verify-pipenet-guards 2>&1 | FileCheck %s

// A composed helper inlined more than once repeats its operations with their
// source and guard locations. The unanalyzable-guard error is reported once per
// operation location, operation name, and guard location, which is what it
// prints; operations that differ only in PipeNet or direction share it.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK: {{^}}copies.py:{{[0-9]+:[0-9]+}}: error: 'ttl.copy' op could not statically analyze the PipeNet guard
  // CHECK: {{^}}guard.py:{{[0-9]+:[0-9]+}}: note: this expression is not statically analyzable
  // CHECK-NOT: error:
  // CHECK: {{^}}later_copy.py:{{[0-9]+:[0-9]+}}: error: 'ttl.copy' op could not statically analyze the PipeNet guard
  // CHECK: {{^}}guard.py:{{[0-9]+:[0-9]+}}: note: this expression is not statically analyzable
  // CHECK-NOT: error:
  func.func @inlined_twice(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 1
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>
    %pipe2 = ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    %condition = arith.cmpi eq, %scaled, %zero : index loc("guard.py":1:1)
    scf.if %condition {
      %send0 = ttl.copy %dfb, %pipe0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("copies.py":1:1)
      %send1 = ttl.copy %dfb, %pipe0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("copies.py":1:1)
      %send2 = ttl.copy %dfb, %pipe1
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>)
          -> !ttl.transfer_handle<write> loc("copies.py":1:1)
      %send3 = ttl.copy %dfb, %pipe2
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("copies.py":1:1)
      %other_line = ttl.copy %dfb, %pipe0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("later_copy.py":1:1)
    }
    func.return
  }

  // An operation at the same location under a different guard keeps its error.
  // CHECK: {{^}}copies.py:{{[0-9]+:[0-9]+}}: error: 'ttl.copy' op could not statically analyze the PipeNet guard
  // CHECK: {{^}}second_guard.py:{{[0-9]+:[0-9]+}}: note: this expression is not statically analyzable
  // CHECK-NOT: error:
  func.func @other_guard(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    %condition = arith.cmpi eq, %scaled, %zero : index loc("second_guard.py":1:1)
    scf.if %condition {
      %send = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("copies.py":1:1)
    }
    func.return
  }

  // Name-only locations can be shared by distinct operations, so each keeps
  // its error.
  // CHECK-COUNT-2: error: loc("send"): 'ttl.copy' op could not statically analyze the PipeNet guard
  func.func @name_locations(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    %condition = arith.cmpi eq, %scaled, %zero : index loc("guard.py":1:1)
    scf.if %condition {
      %send0 = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("send")
      %send1 = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("send")
    }
    func.return
  }
  // CHECK-NOT: error:
}

// -----

// Scopes of a helper inlined twice share one error.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK: {{^}}scopes.py:{{[0-9]+:[0-9]+}}: error: 'ttl.pipenet_scope' op could not statically analyze the PipeNet guard
  // CHECK: {{^}}guard.py:{{[0-9]+:[0-9]+}}: note: this expression is not statically analyzable
  func.func @scopes_inlined_twice(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    %condition = arith.cmpi eq, %scaled, %zero : index loc("guard.py":1:1)
    scf.if %condition {
      ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64], ttl.pipe_net_roles = [0 : i64]} {
        ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        }
      } loc("scopes.py":1:1)
      ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64], ttl.pipe_net_roles = [0 : i64]} {
        ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        }
      } loc("scopes.py":1:1)
    }
    func.return
  }
  // CHECK-NOT: error:
}

// -----

// A `ttl.wait` on a receive request is guard-checked. The receives and waits
// of a helper inlined twice share one error per operation.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK: {{^}}receives.py:{{[0-9]+:[0-9]+}}: error: 'ttl.copy' op could not statically analyze the PipeNet guard
  // CHECK: {{^}}guard.py:{{[0-9]+:[0-9]+}}: note: this expression is not statically analyzable
  // CHECK-NOT: error:
  // CHECK: {{^}}receive_waits.py:{{[0-9]+:[0-9]+}}: error: 'ttl.wait' op could not statically analyze the PipeNet guard
  // CHECK: {{^}}guard.py:{{[0-9]+:[0-9]+}}: note: this expression is not statically analyzable
  func.func @receives_inlined_twice(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %pipe1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 1
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    %condition = arith.cmpi eq, %scaled, %zero : index loc("guard.py":1:1)
    scf.if %condition {
      %block = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %receive0 = ttl.copy %pipe0, %block
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request loc("receives.py":1:1)
      ttl.wait %receive0 : !ttl.receive_request loc("receive_waits.py":1:1)
      %receive1 = ttl.copy %pipe1, %block
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request loc("receives.py":1:1)
      ttl.wait %receive1 : !ttl.receive_request loc("receive_waits.py":1:1)
    }
    func.return
  }
  // CHECK-NOT: error:
}
