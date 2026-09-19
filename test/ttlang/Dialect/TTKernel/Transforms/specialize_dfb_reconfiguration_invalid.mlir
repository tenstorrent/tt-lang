// RUN: ttlang-opt --split-input-file --verify-diagnostics --ttkernel-specialize-dfb-reconfiguration %s

module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = []
  }
} {
  func.func @unknown_ordinal(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    // expected-error @below {{references an unknown DFB reconfiguration ordinal}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"
        template_args [0 : ui32] (%configuration_address) {
          header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 1 : i64
        } : (ui32) -> ()
    return
  }
}

// -----

module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = []
  }
} {
  func.func @malformed_core_coordinate(%configuration_address: ui32) attributes {
    ttl.core_coord = [0, 0]
  } {
    // expected-error @below {{contains malformed DFB reconfiguration metadata}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"
        template_args [0 : ui32] (%configuration_address) {
          header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
        } : (ui32) -> ()
    return
  }
}

// -----

module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = []
  }
} {
  func.func @malformed_template_arguments(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    // expected-error @below {{contains malformed DFB reconfiguration metadata}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"
        template_args [1 : ui32] (%configuration_address) {
          header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
        } : (ui32) -> ()
    return
  }
}
