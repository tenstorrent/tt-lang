// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Verify diagnostics for typed device-domain and transfer invariants.

// A product domain requires unique component names.
// expected-error @below {{duplicate device domain component name 'device'}}
#domain = #ttl.device_domain<components = <name = "device", extent = [1]>, <name = "device", extent = [4]>>

// -----

// Device transfer references must be within every component extent.
// expected-error @below {{device transfer edge.destination component 'device' axis 1 is out of bounds for extent 4, got 4}}
#transfer = #ttl.device_transfer<
  domain = <components = <name = "device", extent = [1, 4]>>,
  edge = <source = <coordinates = [0, 0]>, destination = <coordinates = [0, 4]>>>

// -----

// Device references provide one coordinate per domain component.
// expected-error @below {{device transfer edge.source has 1 component coordinates, expected 2}}
#transfer = #ttl.device_transfer<
  domain = <components = <name = "board", extent = [2]>, <name = "device", extent = [4]>>,
  edge = <source = <coordinates = [0]>, destination = <coordinates = [1], [0]>>>

// -----

// Exact transfers cannot target the source device.
// expected-error @below {{transfer edge source must differ from destination}}
#edge = #ttl.transfer_edge<
  source = <coordinates = [0]>, destination = <coordinates = [0]>>

// -----

// Multicast transfers cannot include the source device.
// expected-error @below {{device transfer edge source-in-destination multicast is not supported}}
#transfer = #ttl.device_transfer<
  domain = <components = <name = "device", extent = [4]>>,
  edge = <source = <coordinates = [1]>, destinationRange = <lo = <coordinates = [0]>, hi = <coordinates = [2]>>>>

// -----

// Component extents contain only positive dimensions.
// expected-error @below {{device domain component 'device' extent axis 0 must be positive, got 0}}
#component = #ttl.device_domain_component<name = "device", extent = [0]>

// -----

// Device ranges use half-open bounds.
// expected-error @below {{device range component 0 axis 0 requires lo < hi, got lo=2, hi=2}}
#range = #ttl.device_range<lo = <coordinates = [2]>, hi = <coordinates = [2]>>

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
