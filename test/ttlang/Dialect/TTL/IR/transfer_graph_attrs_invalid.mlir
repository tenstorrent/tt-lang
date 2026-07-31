// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Verify diagnostics for typed device-domain and transfer-graph invariants.

// A product domain requires unique component names.
// expected-error @below {{duplicate device domain component name 'device'}}
#domain = #ttl.device_domain<components = <name = "device", extent = [1]>, <name = "device", extent = [4]>>

// -----

// Explicit device references must be within every component extent.
// expected-error @below {{transfer graph edge 0.destination component 'device' axis 1 is out of bounds for extent 4, got 4}}
#graph = #ttl.transfer_graph<
  domain = <components = <name = "device", extent = [1, 4]>>,
  edges = [<source = <coordinates = [0, 0]>, destination = <coordinates = [0, 4]>>]>

// -----

// A graph cannot contain explicit and structured relation forms together.
// expected-error @below {{transfer graph must be explicit or structured, not both}}
#graph = #ttl.transfer_graph<
  domain = <components = <name = "device", extent = [4]>>,
  edges = [<source = <coordinates = [0]>, destination = <coordinates = [1]>>],
  structured = #ttl.axis_neighbor_transfer<component = "device", axis = 0 : i64, offset = 1 : i64, wrap = false>>

// -----

// Structured relations must name a component in the associated domain.
// expected-error @below {{structured transfer references unknown component 'board'}}
#graph = #ttl.transfer_graph<
  domain = <components = <name = "device", extent = [4]>>,
  structured = #ttl.axis_neighbor_transfer<component = "board", axis = 0 : i64, offset = 1 : i64, wrap = false>>

// -----

// Axis-neighbor dimensions are logical-domain dimensions.
// expected-error @below {{axis_neighbor axis 1 is out of bounds for component 'device' rank 1}}
#graph = #ttl.transfer_graph<
  domain = <components = <name = "device", extent = [4]>>,
  structured = #ttl.axis_neighbor_transfer<component = "device", axis = 1 : i64, offset = 1 : i64, wrap = false>>

// -----

// Component extents contain only positive dimensions.
// expected-error @below {{device domain component 'device' extent axis 0 must be positive, got 0}}
#component = #ttl.device_domain_component<name = "device", extent = [0]>

// -----

// Device references provide one coordinate per domain component.
// expected-error @below {{transfer graph edge 0.source has 1 component coordinates, expected 2}}
#graph = #ttl.transfer_graph<
  domain = <components = <name = "board", extent = [2]>, <name = "device", extent = [4]>>,
  edges = [<source = <coordinates = [0]>, destination = <coordinates = [1], [0]>>]>

// -----

// Device ranges use half-open bounds.
// expected-error @below {{device range component 0 axis 0 requires lo < hi, got lo=2, hi=2}}
#range = #ttl.device_range<lo = <coordinates = [2]>, hi = <coordinates = [2]>>

// -----

// Axis-neighbor relations require a positive logical offset.
// expected-error @below {{axis_neighbor offset must be positive, got 0}}
#structured = #ttl.axis_neighbor_transfer<component = "device", axis = 0 : i64, offset = 0 : i64, wrap = false>

// -----

// Each stencil translation must include at least one component.
// expected-error @below {{stencil offset 0 must not be empty}}
#structured = #ttl.stencil_transfer<component = "device", offsets = [[]], wrap = false>

// -----

// Stencil relations exclude self-transfers.
// expected-error @below {{stencil offsets must not contain the zero offset}}
#structured = #ttl.stencil_transfer<component = "device", offsets = [[0, 0]], wrap = false>

// -----

// Every stencil offset must match its selected component rank.
// expected-error @below {{stencil offset 0 has rank 1, expected 2 for component 'device'}}
#graph = #ttl.transfer_graph<
  domain = <components = <name = "device", extent = [2, 2]>>,
  structured = #ttl.stencil_transfer<component = "device", offsets = [[1]], wrap = false>>

// -----

// Duplicate offsets do not define distinct transfer edges.
// expected-error @below {{stencil offsets must be unique}}
#structured = #ttl.stencil_transfer<component = "device", offsets = [[1, 0], [1, 0]], wrap = false>

// -----

// A bound device transfer must remain within its associated domain.
// expected-error @below {{device transfer edge.destination component 'device' axis 0 is out of bounds for extent 4, got 4}}
#transfer = #ttl.device_transfer<
  domain = <components = <name = "device", extent = [4]>>,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [4]>>>

// -----

// An exact-device predicate must reference a member of its domain.
module {
  func.func @invalid_is_device() {
    // expected-error @below {{device component 'device' axis 0 is out of bounds for extent 4, got 4}}
    %invalid = ttl.is_device
      <coordinates = [4]> in
      <components = <name = "device", extent = [4]>> : i1
    return
  }
}

// -----

// A range predicate permits the domain extent only as its exclusive bound.
module {
  func.func @invalid_is_device_range() {
    // expected-error @below {{range upper bound component 'device' axis 0 is out of bounds for extent 4, got 5}}
    %invalid = ttl.is_device_in_range
      <lo = <coordinates = [1]>, hi = <coordinates = [5]>> in
      <components = <name = "device", extent = [4]>> : i1
    return
  }
}
