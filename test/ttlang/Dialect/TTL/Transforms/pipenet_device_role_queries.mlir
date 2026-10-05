// Verify that graph role queries retain device matching even when every
// endpoint has the same launch-node coordinates.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-verify-pipenet-guards,convert-ttl-to-ttkernel)' | FileCheck %s

#domain = #ttl.device_domain<components = <name = "device", extent = [4]>>
#records = #ttl.pipenet_records<net 1 name "device_roles" pipes [
  <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
   dstEndX = 0, dstEndY = 0,
   deviceTransfer = <domain = #domain,
     edge = <source = <coordinates = [0]>,
             destination = <coordinates = [3]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  // Matching launch coordinates must not select device 3 as a source.
  // CHECK-LABEL: func.func @device_source
  // CHECK: %[[ARG:.*]] = ttkernel.get_common_arg_val
  // CHECK-NEXT: %[[DEVICE:.*]] = arith.index_cast %[[ARG]] : i32 to index
  // CHECK: %[[RESULT:.*]] = scf.for
  // CHECK-COUNT-4: ttkernel.experimental.constant_table_lookup {{.*}}, [0]
  // CHECK-NEXT: %[[ENDPOINT:.*]] = ttkernel.experimental.constant_table_lookup {{.*}}, [0]
  // CHECK: %[[MATCH:.*]] = arith.cmpi eq, %[[DEVICE]], %[[ENDPOINT]] : index
  // CHECK-NEXT: %[[SELECTED:.*]] = arith.andi %{{.*}}, %[[MATCH]] : i1
  // CHECK-NEXT: %[[ACCUMULATED:.*]] = arith.ori %{{.*}}, %[[SELECTED]] : i1
  // CHECK-NEXT: scf.yield %[[ACCUMULATED]] : i1
  // CHECK: ttl.dprint "source={}"(%[[RESULT]])
  func.func @device_source()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %is_source = ttl.is_src {pipe_net_id = 1 : i64, records = #records}
    "ttl.dprint"(%is_source) {fmt = "source={}", mode = "scalar"} : (i1) -> ()
    func.return
  }

  // Destination matching must use the destination device rather than source.
  // CHECK-LABEL: func.func @device_destination
  // CHECK: %[[ARG:.*]] = ttkernel.get_common_arg_val
  // CHECK-NEXT: %[[DEVICE:.*]] = arith.index_cast %[[ARG]] : i32 to index
  // CHECK: %[[RESULT:.*]] = scf.for
  // CHECK: %[[ENDPOINT:.*]] = ttkernel.experimental.constant_table_lookup {{.*}}, [3]
  // CHECK: %[[MATCH:.*]] = arith.cmpi eq, %[[DEVICE]], %[[ENDPOINT]] : index
  // CHECK-NEXT: %[[SELECTED:.*]] = arith.andi %{{.*}}, %[[MATCH]] : i1
  // CHECK-NEXT: %[[ACCUMULATED:.*]] = arith.ori %{{.*}}, %[[SELECTED]] : i1
  // CHECK-NEXT: scf.yield %[[ACCUMULATED]] : i1
  // CHECK: ttl.dprint "destination={}"(%[[RESULT]])
  func.func @device_destination()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %is_destination = ttl.is_dst {pipe_net_id = 1 : i64, records = #records}
    "ttl.dprint"(%is_destination) {fmt = "destination={}", mode = "scalar"} : (i1) -> ()
    func.return
  }

  // Active queries select either endpoint device, leaving devices 1 and 2 idle.
  // CHECK-LABEL: func.func @device_active
  // CHECK: %[[ARG:.*]] = ttkernel.get_common_arg_val
  // CHECK-NEXT: %[[DEVICE:.*]] = arith.index_cast %[[ARG]] : i32 to index
  // CHECK: %[[RESULT:.*]] = scf.for
  // CHECK: %[[ENDPOINT:.*]] = ttkernel.experimental.constant_table_lookup {{.*}}, [0, 3]
  // CHECK: %[[MATCH:.*]] = arith.cmpi eq, %[[DEVICE]], %[[ENDPOINT]] : index
  // CHECK-NEXT: %[[SELECTED:.*]] = arith.andi %{{.*}}, %[[MATCH]] : i1
  // CHECK-NEXT: %[[ACCUMULATED:.*]] = arith.ori %{{.*}}, %[[SELECTED]] : i1
  // CHECK-NEXT: scf.yield %[[ACCUMULATED]] : i1
  // CHECK: ttl.dprint "active={}"(%[[RESULT]])
  func.func @device_active()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %is_active = ttl.is_active {pipe_net_id = 1 : i64, records = #records}
    "ttl.dprint"(%is_active) {fmt = "active={}", mode = "scalar"} : (i1) -> ()
    func.return
  }
}
