// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,convert-ttl-to-ttkernel)' | FileCheck %s

#records = #ttl.pipenet_records<net 0 name "local_roles" pipes [
  <srcX = 0, srcY = 0, dstStartX = 3, dstStartY = 0,
   dstEndX = 3, dstEndY = 0>,
  <srcX = 0, srcY = 0, dstStartX = 3, dstStartY = 0,
   dstEndX = 3, dstEndY = 0>,
  <srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0,
   dstEndX = 2, dstEndY = 0>
]>

module attributes {ttl.launch_grid = array<i64: 4, 1>} {
  // CHECK-LABEL: func.func @local_is_src
  // CHECK-NOT: scf.for
  // CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [2, 1, 0, 0]
  // CHECK: arith.cmpi ne
  // CHECK-NOT: ttkernel.experimental.constant_table_lookup
  // CHECK-NOT: scf.for
  func.func @local_is_src() -> i1
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %predicate = ttl.is_src {
        pipe_net_id = 0 : i64, records = #records}
    func.return %predicate : i1
  }

  // CHECK-LABEL: func.func @local_is_dst
  // CHECK-NOT: scf.for
  // CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [0, 0, 1, 2]
  // CHECK: arith.cmpi ne
  // CHECK-NOT: ttkernel.experimental.constant_table_lookup
  // CHECK-NOT: scf.for
  func.func @local_is_dst() -> i1
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %predicate = ttl.is_dst {
        pipe_net_id = 0 : i64, records = #records}
    func.return %predicate : i1
  }

  // CHECK-LABEL: func.func @local_is_active
  // CHECK-NOT: scf.for
  // CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [2, 1, 1, 2]
  // CHECK: arith.cmpi ne
  // CHECK-NOT: ttkernel.experimental.constant_table_lookup
  // CHECK-NOT: scf.for
  func.func @local_is_active() -> i1
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %predicate = ttl.is_active {
        pipe_net_id = 0 : i64, records = #records}
    func.return %predicate : i1
  }
}
