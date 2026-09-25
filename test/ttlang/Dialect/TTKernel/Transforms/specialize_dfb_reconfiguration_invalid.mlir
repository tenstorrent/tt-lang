// RUN: ttlang-opt --split-input-file --verify-diagnostics --ttkernel-specialize-dfb-reconfiguration %s

module {
  func.func @missing_plan(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    // expected-error @below {{requires finalized DFB reconfiguration metadata}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
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
  func.func @unknown_ordinal(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    // expected-error @below {{references an unknown DFB reconfiguration ordinal}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
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
    // expected-error @below {{is in a kernel with a malformed}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
      header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
    } : (ui32) -> ()
    return
  }
}

// -----

module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = [{dfb_index = 0 : i32}]
  }
} {
  func.func @missing_configurations(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    // expected-error @below {{contains malformed DFB reconfiguration metadata}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
      header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
    } : (ui32) -> ()
    return
  }
}

// -----

module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = [{dfb_index = 64 : i32, configurations = [
      {entry_reconfiguration = 0 : i64, block_count = 1 : i32,
       num_tiles = 1 : i32, page_size = 2048 : i32}]}]
  }
} {
  func.func @dfb_index_beyond_record_capacity(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    // expected-error @below {{contains malformed DFB reconfiguration metadata}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
      header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
    } : (ui32) -> ()
    return
  }
}

// -----

module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = [{dfb_index = 0 : i32, configurations = [
      {entry_reconfiguration = 0 : i64, block_count = 1 : i32,
       num_tiles = 1 : i32, page_size = 2048 : i32},
      {entry_reconfiguration = 0 : i64, block_count = 2 : i32,
       num_tiles = 1 : i32, page_size = 2048 : i32}]}]
  }
} {
  func.func @ambiguous_configuration(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    // expected-error @below {{contains malformed DFB reconfiguration metadata}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
      header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
    } : (ui32) -> ()
    return
  }
}

// -----

module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = [{dfb_index = 0 : i32, configurations = [
      {entry_reconfiguration = 0 : i64, block_count = 1 : i32,
       num_tiles = 0 : i32, page_size = 2048 : i32}]}]
  }
} {
  func.func @empty_configuration(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    // expected-error @below {{contains malformed DFB reconfiguration metadata}}
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
      header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
    } : (ui32) -> ()
    return
  }
}
