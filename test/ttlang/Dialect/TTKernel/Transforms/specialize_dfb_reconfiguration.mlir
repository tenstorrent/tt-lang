// RUN: ttlang-opt --split-input-file --ttkernel-specialize-dfb-reconfiguration %s | FileCheck %s

// Invariant compiler-managed and tensor-backed storage is specialized after
// each kernel has one core coordinate.
// CHECK-LABEL: func.func @core_0_0
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces" template_args [2 : ui32, 2 : ui32, 1024 : ui32, 4 : ui32, 256 : ui32, 5 : ui32, 4096 : ui32, 2 : ui32, 2048 : ui32]
// CHECK-SAME: dfb_resource_indices = array<i32: 2, 5>
// CHECK-NOT: ttl.dfb_reconfiguration_ordinal
// CHECK-LABEL: func.func @core_1_0
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces" template_args [2 : ui32, 2 : ui32, 1024 : ui32, 4 : ui32, 256 : ui32, 5 : ui32, 4096 : ui32, 2 : ui32, 2048 : ui32]
// CHECK-SAME: dfb_resource_indices = array<i32: 2, 5>
// CHECK-NOT: ttl.dfb_reconfiguration_ordinal
module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = [
      {dfb_index = 2 : i32, configurations = [
        {block_count = 1 : i32, num_tiles = 2 : i32, page_size = 256 : i32},
        {entry_reconfiguration = 0 : i64, block_count = 2 : i32,
         num_tiles = 2 : i32, page_size = 256 : i32}]},
      {dfb_index = 5 : i32, configurations = [
        {block_count = 1 : i32, num_tiles = 1 : i32, page_size = 2048 : i32,
         storage_segments = [
           {nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 3, byte_offset = 64, byte_size = 2048>},
           {nodes = [[1, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 4, byte_offset = 128, byte_size = 2048>}]},
        {entry_reconfiguration = 0 : i64, block_count = 1 : i32,
         num_tiles = 2 : i32, page_size = 2048 : i32,
         storage_segments = [
           {nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 3, byte_offset = 64, byte_size = 4096>},
           {nodes = [[1, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 4, byte_offset = 128, byte_size = 4096>}]}]}
    ]
  }
} {
  func.func @core_0_0(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
      header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
    } : (ui32) -> ()
    return
  }

  func.func @core_1_0(%configuration_address: ui32) attributes {
    ttl.core_coord = [[1, 0]]
  } {
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
      header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
    } : (ui32) -> ()
    return
  }
}

// -----

// Storage-source changes still specialize fixed descriptor fields because the
// selected DFB address remains loaded from its runtime record.
// CHECK-LABEL: func.func @variable_storage
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces" template_args [1 : ui32, 3 : ui32, 2048 : ui32, 1 : ui32, 2048 : ui32](%arg0)
// CHECK-SAME: dfb_resource_indices = array<i32: 3>
// CHECK-NOT: ttl.dfb_reconfiguration_ordinal
module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = [{dfb_index = 3 : i32, configurations = [
      {block_count = 1 : i32, num_tiles = 1 : i32, page_size = 2048 : i32,
       storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 2048>}]},
      {entry_reconfiguration = 0 : i64, block_count = 1 : i32,
       num_tiles = 1 : i32, page_size = 2048 : i32,
       storage_segments = [{nodes = [[0, 0]], tensor_backing = #ttl.tensor_backing<tensor_index = 2, byte_offset = 0, byte_size = 2048>}]}]}]
  }
} {
  func.func @variable_storage(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
      header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
    } : (ui32) -> ()
    return
  }
}

// -----

// A synchronization point with no active DFBs avoids runtime mask scans.
// CHECK-LABEL: func.func @no_active_dfbs
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces" template_args [0 : ui32]
// CHECK-NOT: dfb_resource_indices
// CHECK-NOT: ttl.dfb_reconfiguration_ordinal
module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = []
  }
} {
  func.func @no_active_dfbs(%configuration_address: ui32) attributes {
    ttl.core_coord = [[0, 0]]
  } {
    ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%configuration_address) {
      header = "<cstdint>", ttl.dfb_reconfiguration_ordinal = 0 : i64
    } : (ui32) -> ()
    return
  }
}
