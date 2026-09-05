// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,convert-ttl-to-ttkernel)' | FileCheck %s

// Verify that each local PipeNet role predicate lowers to one participant-table
// lookup without a per-record loop.

#records = #ttl.pipenet_records<net 0 name "local_roles" pipes [
  <srcX = 0, srcY = 0, dstStartX = 3, dstStartY = 0,
   dstEndX = 3, dstEndY = 0>,
  <srcX = 0, srcY = 0, dstStartX = 3, dstStartY = 0,
   dstEndX = 3, dstEndY = 0>,
  <srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0,
   dstEndX = 2, dstEndY = 0>
]>

module attributes {ttl.launch_grid = array<i64: 4, 1>} {
  // Duplicate records must not change boolean source membership.
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

  // Destination membership includes each node selected by a record range.
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

  // Active membership is the union of the source and destination sets.
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
