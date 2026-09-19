// RUN: ttlang-opt --split-input-file --ttkernel-specialize-dfb-reconfiguration %s | FileCheck %s

// Records are selected by core while retaining their offsets in the compact
// runtime address array. Descriptor participation is selected per core.
// CHECK-LABEL: func.func @core_0_0
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces_specialized" template_args [2 : ui32, 1 : ui32, 0 : ui32, 2 : ui32, 1024 : ui32, 4 : ui32, 256 : ui32, 0 : ui32, 5 : ui32, 32 : ui32, 32 : ui32, 16 : ui32, 4 : ui32, 5 : ui32, 5 : ui32]
// CHECK-SAME: dfb_resource_indices = array<i32: 2>
// CHECK-NOT: ttl.dfb_reconfiguration_ordinal
// CHECK-LABEL: func.func @core_1_0
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_descriptors_specialized" template_args [2 : ui32, 1 : ui32, 1 : ui32, 5 : ui32, 4096 : ui32, 2 : ui32, 2048 : ui32, 1 : ui32, 6 : ui32, 32 : ui32, 32 : ui32, 16 : ui32, 4 : ui32, 6 : ui32, 6 : ui32]
// CHECK-SAME: dfb_resource_indices = array<i32: 5>
// CHECK-NOT: ttl.dfb_reconfiguration_ordinal
// CHECK-LABEL: func.func @core_2_0
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces_specialized" template_args [2 : ui32, 0 : ui32]
// CHECK-NOT: dfb_resource_indices
// CHECK-NOT: ttl.dfb_reconfiguration_ordinal
module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = [
      {dfb_index = 2 : i32, configurations = [
        {entry_reconfiguration = 0 : i64,
         storage_segments = [{nodes = [[0, 0]]}]}]},
      {dfb_index = 5 : i32, configurations = [
        {entry_reconfiguration = 0 : i64,
         storage_segments = [{nodes = [[1, 0]]}]}]}
    ]
  }
} {
  func.func @core_0_0(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    ttkernel.opaque_call "experimental::reconfigure_dfb_descriptors"
        template_args [2 : ui32,
          2 : ui32, 1024 : ui32, 4 : ui32, 256 : ui32, 0 : ui32,
          5 : ui32, 32 : ui32, 32 : ui32, 16 : ui32, 4 : ui32, 5 : ui32, 5 : ui32,
          5 : ui32, 4096 : ui32, 2 : ui32, 2048 : ui32, 1 : ui32,
          6 : ui32, 32 : ui32, 32 : ui32, 16 : ui32, 4 : ui32, 6 : ui32, 6 : ui32]
        (%configuration_address) {
          header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
        } : (ui32) -> ()
    return
  }

  func.func @core_1_0(%configuration_address: ui32) attributes {
    ttl.core_coord = [[1, 0]]
  } {
    ttkernel.opaque_call "experimental::reconfigure_dfb_descriptors"
        template_args [2 : ui32,
          2 : ui32, 1024 : ui32, 4 : ui32, 256 : ui32, 0 : ui32,
          5 : ui32, 32 : ui32, 32 : ui32, 16 : ui32, 4 : ui32, 5 : ui32, 5 : ui32,
          5 : ui32, 4096 : ui32, 2 : ui32, 2048 : ui32, 1 : ui32,
          6 : ui32, 32 : ui32, 32 : ui32, 16 : ui32, 4 : ui32, 6 : ui32, 6 : ui32]
        (%configuration_address) {
          header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
        } : (ui32) -> ()
    return
  }

  func.func @core_2_0(%configuration_address: ui32) attributes {
    ttl.core_coord = [[2, 0]]
  } {
    ttkernel.opaque_call "experimental::reconfigure_dfb_descriptors"
        template_args [2 : ui32,
          2 : ui32, 1024 : ui32, 4 : ui32, 256 : ui32, 0 : ui32,
          5 : ui32, 32 : ui32, 32 : ui32, 16 : ui32, 4 : ui32, 5 : ui32, 5 : ui32,
          5 : ui32, 4096 : ui32, 2 : ui32, 2048 : ui32, 1 : ui32,
          6 : ui32, 32 : ui32, 32 : ui32, 16 : ui32, 4 : ui32, 6 : ui32, 6 : ui32]
        (%configuration_address) {
          header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
        } : (ui32) -> ()
    return
  }
}

// -----

// A function without a specialized coordinate retains generic mask handling.
// CHECK-LABEL: func.func @generic
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces" template_args [0 : ui32]
// CHECK-NOT: ttl.dfb_reconfiguration_ordinal
module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = []
  }
} {
  func.func @generic(%configuration_address: ui32) {
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"
        template_args [0 : ui32] (%configuration_address) {
          header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
        } : (ui32) -> ()
    return
  }
}
