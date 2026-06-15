// RUN: ttlang-opt %s | FileCheck %s

// Verifies that canonical device-domain and transfer-graph attributes parse,
// verify cross-field invariants, and print without exposing helper attrs.

#single = #ttl.device_domain<levels = [{cluster_axis = 1 : i64, extent = array<i64: 1, 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>
#explicit = #ttl.transfer_graph<domain = #single, edges = [{destination = [array<i64: 0, 1>], source = [array<i64: 0, 0>]}]>

#hier = #ttl.device_domain<levels = [{cluster_axis = 0 : i64, extent = array<i64: 1>, mesh_id = 0 : i64, name = "board", periodic = true, topology = "fabric_ring"}, {cluster_axis = 0 : i64, extent = array<i64: 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>
#structured = #ttl.transfer_graph<domain = #hier, edges = [], structured = {axis = 0 : i64, kind = "axis_neighbor", level = "device", offset = 1 : i64, wrap = false}>

module {
  // CHECK-LABEL: func.func @domain_attr
  // CHECK-SAME: domain = #ttl.device_domain<levels = [{cluster_axis = 1 : i64, extent = array<i64: 1, 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>
  func.func @domain_attr() attributes {domain = #single} {
    return
  }

  // CHECK-LABEL: func.func @explicit_graph_attr
  // CHECK-SAME: transfer_graph = #ttl.transfer_graph<
  // CHECK-SAME: domain = <levels = [{cluster_axis = 1 : i64, extent = array<i64: 1, 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>
  // CHECK-SAME: edges = [{destination = [array<i64: 0, 1>], source = [array<i64: 0, 0>]}]
  func.func @explicit_graph_attr() attributes {transfer_graph = #explicit} {
    return
  }

  // CHECK-LABEL: func.func @structured_graph_attr
  // CHECK-SAME: transfer_graph = #ttl.transfer_graph<
  // CHECK-SAME: domain = <levels = [{cluster_axis = 0 : i64, extent = array<i64: 1>, mesh_id = 0 : i64, name = "board", periodic = true, topology = "fabric_ring"}, {cluster_axis = 0 : i64, extent = array<i64: 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>
  // CHECK-SAME: edges = []
  // CHECK-SAME: structured = {axis = 0 : i64, kind = "axis_neighbor", level = "device", offset = 1 : i64, wrap = false}
  func.func @structured_graph_attr() attributes {transfer_graph = #structured} {
    return
  }
}
