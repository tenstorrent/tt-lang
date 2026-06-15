// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Invalid tests for canonical device-domain and transfer-graph attributes.
// Each split section checks one verifier error.

// expected-error @below {{duplicate device_domain level name `device`}}
#domain = #ttl.device_domain<levels = [{cluster_axis = 0 : i64, extent = array<i64: 1>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}, {cluster_axis = 0 : i64, extent = array<i64: 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>

// -----

#domain = #ttl.device_domain<levels = [{cluster_axis = 1 : i64, extent = array<i64: 1, 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>

// expected-error @below {{transfer_graph edge 0.destination level `device` axis 1 is out of bounds for extent 4, got 4}}
#graph = #ttl.transfer_graph<domain = #domain, edges = [{destination = [array<i64: 0, 4>], source = [array<i64: 0, 0>]}]>

// -----

#domain = #ttl.device_domain<levels = [{cluster_axis = 0 : i64, extent = array<i64: 2>, mesh_id = 0 : i64, name = "board", periodic = true, topology = "fabric_ring"}, {cluster_axis = 0 : i64, extent = array<i64: 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>

// expected-error @below {{transfer_graph edge 0 requires multiple topology levels; multi-level route lowering is deferred to MD-14}}
#graph = #ttl.transfer_graph<domain = #domain, edges = [{destination = [array<i64: 1>, array<i64: 1>], source = [array<i64: 0>, array<i64: 0>]}]>

// -----

#domain = #ttl.device_domain<levels = [{cluster_axis = 1 : i64, extent = array<i64: 2, 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>

// expected-error @below {{transfer_graph edge 0 requires non-cluster axis 0 on level `device` to stay fixed}}
#graph = #ttl.transfer_graph<domain = #domain, edges = [{destination = [array<i64: 1, 1>], source = [array<i64: 0, 0>]}]>

// -----

#domain = #ttl.device_domain<levels = [{cluster_axis = 0 : i64, extent = array<i64: 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>

// expected-error @below {{transfer_graph must be explicit or structured, not both}}
#graph = #ttl.transfer_graph<domain = #domain, edges = [{destination = [array<i64: 1>], source = [array<i64: 0>]}], structured = {axis = 0 : i64, kind = "axis_neighbor", level = "device", offset = 1 : i64, wrap = false}>

// -----

#domain = #ttl.device_domain<levels = [{cluster_axis = 1 : i64, extent = array<i64: 1, 4>, mesh_id = 0 : i64, name = "device", periodic = false, topology = "fabric_1d"}]>

// expected-error @below {{axis_neighbor on level `device` must use cluster_axis 1, got axis 0}}
#graph = #ttl.transfer_graph<domain = #domain, edges = [], structured = {axis = 0 : i64, kind = "axis_neighbor", level = "device", offset = 1 : i64, wrap = false}>
