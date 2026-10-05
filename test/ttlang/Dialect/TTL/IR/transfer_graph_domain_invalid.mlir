// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: Reject sparse explicit graphs whose domains cannot be indexed.

// One edge does not make a domain of 2^63 devices index-representable.
func.func @signed_device_count_overflow() attributes {
  // expected-error @below {{transfer graph device count exceeds the supported index range}}
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "device", extent = [2147483648, 4294967296]>>,
    kind = explicit, properties = {
      edges = [#ttl.transfer_edge<source = <coordinates = [0, 0]>,
                                 destination = <coordinates = [0, 1]>>]}>
} { return }

// -----

// Detect multiplication overflow inside one component before lowering uses the
// device count in signed index arithmetic.
func.func @component_device_count_overflow() attributes {
  // expected-error @below {{transfer graph device count exceeds the supported index range}}
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "device", extent = [4294967296, 4294967296]>>,
    kind = explicit, properties = {
      edges = [#ttl.transfer_edge<source = <coordinates = [0, 0]>,
                                 destination = <coordinates = [0, 1]>>]}>
} { return }

// -----

// Individually representable components can overflow when multiplied together.
func.func @product_device_count_overflow() attributes {
  // expected-error @below {{transfer graph device count exceeds the supported index range}}
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "group", extent = [4294967296]>,
                           <name = "device", extent = [4294967296]>>,
    kind = explicit, properties = {
      edges = [#ttl.transfer_edge<source = <coordinates = [0], [0]>,
                                 destination = <coordinates = [0], [1]>>]}>
} { return }
