// RUN: ttlang-opt %s | FileCheck %s

// Verify typed architecture-neutral device domains and transfer graphs.

#explicit = #ttl.transfer_graph<
  domain = <components = <name = "device", extent = [2, 4]>>,
  edges = [<source = <coordinates = [0, 0]>, destination = <coordinates = [1, 0]>>]>

#structured = #ttl.transfer_graph<
  domain = <components = <name = "board", extent = [2]>, <name = "device", extent = [4]>>,
  structured = #ttl.axis_neighbor_transfer<component = "device", axis = 0 : i64, offset = 1 : i64, wrap = false>>

#gather = #ttl.transfer_graph<
  domain = <components = <name = "device", extent = [4]>>,
  structured = #ttl.gather_transfer<component = "device", root = <coordinates = [0]>>>

#multicast = #ttl.transfer_graph<
  domain = <components = <name = "device", extent = [4]>>,
  structured = #ttl.multicast_transfer<component = "device", source = <coordinates = [0]>>>

#device_transfer = #ttl.device_transfer<
  domain = <components = <name = "device", extent = [4]>>,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [3]>>>

module {
  // Verify that the high-level graph accepts an edge across any logical axis.
  // CHECK-LABEL: func.func @explicit_graph_attr
  // CHECK-SAME: transfer_graph = #ttl.transfer_graph<
  // CHECK-SAME: domain = <components = <name = "device", extent = [2, 4]>>
  // CHECK-SAME: edges = [<source = <coordinates = {{\[0, 0\]}}>, destination = <coordinates = {{\[1, 0\]}}>>]>
  func.func @explicit_graph_attr() attributes {transfer_graph = #explicit} {
    return
  }

  // Verify that product domains retain named components without topology.
  // CHECK-LABEL: func.func @structured_graph_attr
  // CHECK-SAME: transfer_graph = #ttl.transfer_graph<
  // CHECK-SAME: domain = <components = <name = "board", extent = [2]>, <name = "device", extent = [4]>>
  // CHECK-SAME: structured = #ttl.axis_neighbor_transfer<component = "device", axis = 0 : i64, offset = 1 : i64, wrap = false>
  func.func @structured_graph_attr() attributes {transfer_graph = #structured} {
    return
  }

  // Verify each rooted structured transfer has a distinct concrete attribute.
  // CHECK-LABEL: func.func @rooted_graph_attrs
  // CHECK-SAME: gather = #ttl.transfer_graph<
  // CHECK-SAME: structured = #ttl.gather_transfer<component = "device", root = <coordinates = {{\[0\]}}>>
  // CHECK-SAME: multicast = #ttl.transfer_graph<
  // CHECK-SAME: structured = #ttl.multicast_transfer<component = "device", source = <coordinates = {{\[0\]}}>>
  func.func @rooted_graph_attrs() attributes {gather = #gather, multicast = #multicast} {
    return
  }

  // Verify that a node-level pipe can retain its logical device edge.
  // CHECK-LABEL: func.func @device_transfer_attr
  // CHECK-SAME: device_transfer = #ttl.device_transfer<
  // CHECK-SAME: domain = <components = <name = "device", extent = [4]>>
  // CHECK-SAME: edge = <source = <coordinates = {{\[0\]}}>, destination = <coordinates = {{\[3\]}}>>
  func.func @device_transfer_attr() attributes {device_transfer = #device_transfer} {
    return
  }

  // Verify exact-device and range predicates over the same logical domain.
  // CHECK-LABEL: func.func @device_predicates
  // CHECK-NEXT: %{{.*}} = ttl.is_device <coordinates = [2]> in <components = <name = "device", extent = [4]>> : i1
  // CHECK-NEXT: %{{.*}} = ttl.is_device_in_range <lo = <coordinates = [1]>, hi = <coordinates = [3]>> in <components = <name = "device", extent = [4]>> : i1
  // CHECK-NEXT: return
  func.func @device_predicates() {
    %is_device = ttl.is_device
      <coordinates = [2]> in
      <components = <name = "device", extent = [4]>> : i1
    %is_in_range = ttl.is_device_in_range
      <lo = <coordinates = [1]>, hi = <coordinates = [3]>> in
      <components = <name = "device", extent = [4]>> : i1
    return
  }
}
