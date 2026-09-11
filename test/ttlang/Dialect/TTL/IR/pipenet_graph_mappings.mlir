// RUN: ttlang-opt %s | FileCheck %s

// Summary: Verifies graph PipeNet attributes store graph and node relations
// separately.

// The graph and node-pipe relation print separately instead of as concrete
// device-edge by node-pipe records.
// CHECK-LABEL: func.func @all_to_all_mapping
// CHECK-SAME: test.records = #ttl.pipenet_records<net 7 name "exchange" mappings
// CHECK-SAME: kind = all_to_all
// CHECK-SAME: pipes[<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0
func.func @all_to_all_mapping() attributes {
    test.records = #ttl.pipenet_records<net 7 name "exchange" mappings
      <graph = <domain = <components = <name = "device", extent = [4]>>,
        kind = all_to_all, componentName = "device", properties = {}>,
       pipes[<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
              dstEndX = 0, dstEndY = 0>]>>} {
  return
}

// The six graph kinds retain their declarative properties.
// CHECK-LABEL: func.func @graph_kinds
// CHECK-SAME: test.axis_neighbor = #ttl.transfer_graph<
// CHECK-SAME: kind = axis_neighbor
// CHECK-SAME: properties = {axis = 0 : i64, offset = 1 : i64, wrap = false}
// CHECK-SAME: test.explicit = #ttl.transfer_graph<
// CHECK-SAME: kind = explicit
// CHECK-SAME: test.gather = #ttl.transfer_graph<
// CHECK-SAME: kind = gather
// CHECK-SAME: test.scatter = #ttl.transfer_graph<
// CHECK-SAME: kind = scatter
// CHECK-SAME: test.stencil = #ttl.transfer_graph<
// CHECK-SAME: kind = stencil
func.func @graph_kinds() attributes {
    test.axis_neighbor = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [4]>>,
      kind = axis_neighbor, componentName = "device",
      properties = {axis = 0 : i64, offset = 1 : i64, wrap = false}>,
    test.explicit = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [4]>>,
      kind = explicit, properties = {
        edges = [#ttl.transfer_edge<source = <coordinates = [0]>,
                                    destination = <coordinates = [1]>>]}>,
    test.gather = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [4]>>,
      kind = gather, componentName = "device",
      properties = {root = #ttl.device_ref<coordinates = [0]>}>,
    test.scatter = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [4]>>,
      kind = scatter, componentName = "device",
      properties = {source = #ttl.device_ref<coordinates = [0]>}>,
    test.stencil = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [4]>>,
      kind = stencil, componentName = "device",
      properties = {offsets = [array<i64: -1>, array<i64: 1>], wrap = false}>} {
  return
}

// Mapping order determines callback order when relations use different pipes.
// CHECK-LABEL: func.func @mapping_order
// CHECK-SAME: kind = gather
// CHECK-SAME: pipes[<srcX = 1, srcY = 0
// CHECK-SAME: kind = scatter
// CHECK-SAME: pipes[<srcX = 2, srcY = 0
func.func @mapping_order() attributes {
    test.records = #ttl.pipenet_records<net 9 mappings
      <graph = <domain = <components = <name = "device", extent = [4]>>,
        kind = gather, componentName = "device",
        properties = {root = #ttl.device_ref<coordinates = [0]>}>,
       pipes[<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
              dstEndX = 0, dstEndY = 0>]>,
      <graph = <domain = <components = <name = "device", extent = [4]>>,
        kind = scatter, componentName = "device",
        properties = {source = #ttl.device_ref<coordinates = [0]>}>,
       pipes[<srcX = 2, srcY = 0, dstStartX = 0, dstStartY = 0,
              dstEndX = 0, dstEndY = 0>]>>} {
  return
}
