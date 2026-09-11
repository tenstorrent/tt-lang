// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: Verifies invalid graph and node-pipe mappings are rejected.

// A mapping's node pipe cannot bind another logical-device transfer.
func.func @device_bound_node_pipe() attributes {
    // expected-error @below {{node pipes must not contain a bound device transfer}}
    test.mapping = #ttl.pipe_mapping<graph = <
      domain = <components = <name = "device", extent = [2]>>,
      kind = all_to_all, componentName = "device", properties = {}>, pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0,
         deviceTransfer = <
           domain = <components = <name = "device", extent = [2]>>,
           edge = <source = <coordinates = [0]>,
                   destination = <coordinates = [1]>>>>]>} {
  return
}

// -----

// Repeating a node pipe repeats every complete transfer in the mapping.
func.func @duplicate_node_pipe() attributes {
    // expected-error @below {{contains a duplicate node pipe}}
    test.mapping = #ttl.pipe_mapping<graph = <
      domain = <components = <name = "device", extent = [2]>>,
      kind = all_to_all, componentName = "device", properties = {}>, pipes[
        <srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0>,
        <srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0>]>} {
  return
}

// -----

// Every mapping in one PipeNet uses the same logical-device domain.
func.func @mismatched_mapping_domains() attributes {
    // expected-error @below {{all graph mappings must use the same logical device domain}}
    test.records = #ttl.pipenet_records<net 0 mappings
      <graph = <domain = <components = <name = "device", extent = [2]>>,
        kind = all_to_all, componentName = "device", properties = {}>,
       pipes[<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
              dstEndX = 0, dstEndY = 0>]>,
      <graph = <domain = <components = <name = "device", extent = [3]>>,
        kind = all_to_all, componentName = "device", properties = {}>,
       pipes[<srcX = 2, srcY = 0, dstStartX = 0, dstStartY = 0,
              dstEndX = 0, dstEndY = 0>]>>} {
  return
}

// -----

// Separate mappings cannot repeat the same device edge and node pipe.
func.func @duplicate_edge_and_node_pipe() attributes {
    // expected-error @below {{graph mappings repeat the same device edge and node pipe}}
    test.records = #ttl.pipenet_records<net 0 mappings
      <graph = <domain = <components = <name = "device", extent = [2]>>,
        kind = all_to_all, componentName = "device", properties = {}>,
       pipes[<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
              dstEndX = 0, dstEndY = 0>]>,
      <graph = <domain = <components = <name = "device", extent = [2]>>,
        kind = all_to_all, componentName = "device", properties = {}>,
       pipes[<srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
              dstEndX = 0, dstEndY = 0>]>>} {
  return
}

// -----

// Structured graphs must contain at least one edge.
func.func @empty_structured_relation() attributes {
    // expected-error @below {{transfer graph relation contains no edges}}
    test.graph = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [1]>>,
      kind = all_to_all, componentName = "device", properties = {}>} {
  return
}

// -----

// Axis-neighbor properties require a positive offset.
func.func @invalid_axis_neighbor_offset() attributes {
    // expected-error @below {{axis-neighbor offset must be positive}}
    test.graph = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [2]>>,
      kind = axis_neighbor, componentName = "device",
      properties = {axis = 0 : i64, offset = 0 : i64, wrap = false}>} {
  return
}

// -----

// The product domain must fit the compiler's signed index representation.
func.func @unrepresentable_graph_domain() attributes {
    // expected-error @below {{structured transfer graph device count exceeds the supported index range}}
    test.graph = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [3037000500, 3037000500]>>,
      kind = all_to_all, componentName = "device", properties = {}>} {
  return
}

// -----

// A PipeNet attribute stores either local records or graph mappings.
func.func @mixed_record_representations() attributes {
    // expected-error @below {{requires exactly one of pipe records or graph mappings}}
    test.records = #ttl.pipenet_records<net 0
      pipes[<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
             dstEndX = 1, dstEndY = 0>]
      mappings <graph = <
        domain = <components = <name = "device", extent = [2]>>,
        kind = all_to_all, componentName = "device", properties = {}>,
        pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
               dstEndX = 0, dstEndY = 0>]>>} {
  return
}

// -----

// Graph mappings currently support point-to-point node pipes only.
func.func @collective_node_pipe() attributes {
    // expected-error @below {{graph node collective destinations require multicast lowering}}
    test.mapping = #ttl.pipe_mapping<graph = <
      domain = <components = <name = "device", extent = [2]>>,
      kind = all_to_all, componentName = "device", properties = {}>, pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 1, dstEndY = 0, isCollective = true>]>} {
  return
}

// -----

// Explicit graph edges are unique because edge order defines callback order.
func.func @duplicate_explicit_edge() attributes {
    // expected-error @below {{explicit transfer graph edge 1 duplicates an earlier edge}}
    test.graph = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [2]>>,
      kind = explicit, properties = {
        edges = [#ttl.transfer_edge<source = <coordinates = [0]>,
                                    destination = <coordinates = [1]>>,
                 #ttl.transfer_edge<source = <coordinates = [0]>,
                                    destination = <coordinates = [1]>>]}>} {
  return
}

// -----

// A structured graph names one component of its logical-device domain.
func.func @unknown_structured_component() attributes {
    // expected-error @below {{structured transfer graph references unknown domain component 'worker'}}
    test.graph = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [2]>>,
      kind = all_to_all, componentName = "worker", properties = {}>} {
  return
}

// -----

// A wrapped axis offset must not map every source back to itself.
func.func @self_wrapping_axis_neighbor() attributes {
    // expected-error @below {{axis-neighbor transfer relation must contain a non-self edge}}
    test.graph = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [4]>>,
      kind = axis_neighbor, componentName = "device",
      properties = {axis = 0 : i64, offset = 4 : i64, wrap = true}>} {
  return
}

// -----

// Wrapped stencil offsets must identify distinct effective translations.
func.func @duplicate_wrapped_stencil_offset() attributes {
    // expected-error @below {{stencil offset 1 duplicates an earlier effective offset}}
    test.graph = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [4]>>,
      kind = stencil, componentName = "device",
      properties = {offsets = [array<i64: 1>, array<i64: 5>], wrap = true}>} {
  return
}

// -----

// Gather roots must belong to the selected domain component.
func.func @gather_root_out_of_bounds() attributes {
    // expected-error @below {{root component 'device' axis 0 is out of bounds for extent 4, got 4}}
    test.graph = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [4]>>,
      kind = gather, componentName = "device",
      properties = {root = #ttl.device_ref<coordinates = [4]>}>} {
  return
}

// -----

// All-to-all relations have no relation-specific properties.
func.func @all_to_all_properties() attributes {
    // expected-error @below {{all-to-all transfer graph does not accept properties}}
    test.graph = #ttl.transfer_graph<
      domain = <components = <name = "device", extent = [2]>>,
      kind = all_to_all, componentName = "device",
      properties = {unexpected = true}>} {
  return
}
