// RUN: sed "s|TEST_FILE|%t.mlir|g" %s > %t.mlir
// RUN: ttlang-opt %t.mlir --split-input-file --verify-diagnostics -ttl-verify-pipenet-guards
// RUN: not ttlang-opt %t.mlir --split-input-file -ttl-verify-pipenet-guards 2>&1 | FileCheck %s

// Generated copies can share one source location. Distinct PipeNets, endpoint
// domains, guard locations, and operation locations retain their diagnostics.
// CHECK: error: 'ttl.copy' op could not statically analyze the PipeNet guard
// CHECK: error: 'ttl.copy' op could not statically analyze the PipeNet guard
// CHECK: error: 'ttl.copy' op could not statically analyze the PipeNet guard
// CHECK: error: 'ttl.copy' op could not statically analyze the PipeNet guard
// CHECK: error: 'ttl.copy' op could not statically analyze the PipeNet guard
// CHECK-NOT: error: 'ttl.copy' op could not statically analyze the PipeNet guard

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @same_guard(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
    // expected-note @below {{this expression is not statically analyzable}}
    %condition = arith.cmpi eq, %scaled, %zero : index
    scf.if %condition {
      // expected-error @below {{could not statically analyze the PipeNet guard}}
      %send0 = ttl.copy %dfb, %pipe0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("TEST_FILE":31:7)
      %send1 = ttl.copy %dfb, %pipe0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("TEST_FILE":31:7)
      %send2 = ttl.copy %dfb, %pipe1
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 1>)
          -> !ttl.transfer_handle<write> loc("TEST_FILE":31:7)
      %send3 = ttl.copy %dfb, %pipe2
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("TEST_FILE":31:7)
      // expected-error @below {{could not statically analyze the PipeNet guard}}
      %other_source = ttl.copy %dfb, %pipe0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @other_guard(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    // expected-note @below {{this expression is not statically analyzable}}
    %condition = arith.cmpi eq, %scaled, %zero : index
    scf.if %condition {
      %send = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write> loc("TEST_FILE":31:7)
    }
    func.return
  }
}
