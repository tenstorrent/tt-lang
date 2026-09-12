// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,convert-ttl-to-ttkernel)' | FileCheck %s --implicit-check-not=scf.for

// Verify row-major role-query lookup and result data flow for duplicate records,
// rectangular receivers, a loopback endpoint, and an inactive launch node.

#records = #ttl.pipenet_records<net 0 name "local_roles" pipes [
  <srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 0,
   dstEndX = 2, dstEndY = 1, isCollective = true>,
  <srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 0,
   dstEndX = 2, dstEndY = 1, isCollective = true>,
  <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
   dstEndX = 2, dstEndY = 0, isCollective = true>
]>

module attributes {ttl.launch_grid = array<i64: 3, 2>} {
  // Duplicate records must not change boolean source membership.
  // CHECK-LABEL: func.func @local_is_src
  // CHECK-NOT: scf.for
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[WIDTH:.*]] = arith.constant 3 : index
  // CHECK: %[[X:.*]] = ttkernel.my_logical_x
  // CHECK-NEXT: %[[Y:.*]] = ttkernel.my_logical_y
  // CHECK-NEXT: %[[ROW:.*]] = arith.muli %[[Y]], %[[WIDTH]] : index
  // CHECK-NEXT: %[[NODE:.*]] = arith.addi %[[ROW]], %[[X]] : index
  // CHECK-NEXT: %[[COUNT:.*]] = ttkernel.experimental.constant_table_lookup %[[NODE]], [0, 0, 1, 2, 0, 0]
  // CHECK-NEXT: %[[MATCH:.*]] = arith.cmpi ne, %[[COUNT]], %[[ZERO]] : index
  // CHECK-NEXT: ttl.dprint "source={}"(%[[MATCH]])
  // CHECK-NOT: ttkernel.experimental.constant_table_lookup
  // CHECK-NOT: scf.for
  func.func @local_is_src()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %predicate = ttl.is_src {
        pipe_net_id = 0 : i64, records = #records}
    "ttl.dprint"(%predicate) {fmt = "source={}", mode = "scalar"} : (i1) -> ()
    func.return
  }

  // Destination membership includes each node selected by a record range.
  // CHECK-LABEL: func.func @local_is_dst
  // CHECK-NOT: scf.for
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[COUNT:.*]] = ttkernel.experimental.constant_table_lookup {{.*}}, [0, 2, 3, 0, 2, 2]
  // CHECK-NEXT: %[[MATCH:.*]] = arith.cmpi ne, %[[COUNT]], %[[ZERO]] : index
  // CHECK-NEXT: ttl.dprint "destination={}"(%[[MATCH]])
  // CHECK-NOT: ttkernel.experimental.constant_table_lookup
  // CHECK-NOT: scf.for
  func.func @local_is_dst()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %predicate = ttl.is_dst {
        pipe_net_id = 0 : i64, records = #records}
    "ttl.dprint"(%predicate) {fmt = "destination={}", mode = "scalar"} : (i1) -> ()
    func.return
  }

  // Active membership counts the loopback record once, despite both roles.
  // CHECK-LABEL: func.func @local_is_active
  // CHECK-NOT: scf.for
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[COUNT:.*]] = ttkernel.experimental.constant_table_lookup {{.*}}, [0, 2, 3, 2, 2, 2]
  // CHECK-NEXT: %[[MATCH:.*]] = arith.cmpi ne, %[[COUNT]], %[[ZERO]] : index
  // CHECK-NEXT: ttl.dprint "active={}"(%[[MATCH]])
  // CHECK-NOT: ttkernel.experimental.constant_table_lookup
  // CHECK-NOT: scf.for
  func.func @local_is_active()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %predicate = ttl.is_active {
        pipe_net_id = 0 : i64, records = #records}
    "ttl.dprint"(%predicate) {fmt = "active={}", mode = "scalar"} : (i1) -> ()
    func.return
  }
}
