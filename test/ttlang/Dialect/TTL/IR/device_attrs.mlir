// RUN: ttlang-opt %s | FileCheck %s

// Verify typed architecture-neutral device domains and transfers.

#device_transfer = #ttl.device_transfer<
  domain = <components = <name = "device", extent = [4]>>,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [3]>>>

module {
  // Verify that a node-level pipe can retain its logical device edge.
  // CHECK-LABEL: func.func @device_transfer_attr
  // CHECK-SAME: device_transfer = #ttl.device_transfer<
  // CHECK-SAME: domain = <components = <name = "device", extent = [4]>>
  // CHECK-SAME: edge = <source = <coordinates = {{\[0\]}}>, destination = <coordinates = {{\[3\]}}>>
  func.func @device_transfer_attr() attributes {device_transfer = #device_transfer} {
    return
  }

  // Verify logical-device indexing and predicates over the same domain.
  // CHECK-LABEL: func.func @device_predicates
  // CHECK-NEXT: %{{.*}} = ttl.current_device_index <components = <name = "device", extent = [4]>> : index
  // CHECK-NEXT: %{{.*}} = ttl.is_device <coordinates = [2]> in <components = <name = "device", extent = [4]>> : i1
  // CHECK-NEXT: %{{.*}} = ttl.is_device_in_range <lo = <coordinates = [1]>, hi = <coordinates = [3]>> in <components = <name = "device", extent = [4]>> : i1
  // CHECK-NEXT: return
  func.func @device_predicates() {
    %device_index = ttl.current_device_index
      <components = <name = "device", extent = [4]>> : index
    %is_device = ttl.is_device
      <coordinates = [2]> in
      <components = <name = "device", extent = [4]>> : i1
    %is_in_range = ttl.is_device_in_range
      <lo = <coordinates = [1]>, hi = <coordinates = [3]>> in
      <components = <name = "device", extent = [4]>> : i1
    return
  }
}
