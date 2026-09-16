// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// RUN: ttlang-transfer-graph-test %s | FileCheck %s
//
// Check every device's incident count, global edge ordinal, and both endpoints
// against static enumeration. Later product components contribute to the
// device-index stride even when they are not part of the selected component.

// CHECK: two_components_group_axis0_offset1_bounded: 6 devices, 3 edges, both endpoint roles verified
func.func @two_components_group_axis0_offset1_bounded() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "group", extent = [2]>, <name = "device", extent = [3]>>,
    kind = axis_neighbor, componentName = "group",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = false}>
} {
  return
}

// CHECK: two_components_group_axis0_offset1_wrap: 6 devices, 6 edges, both endpoint roles verified
func.func @two_components_group_axis0_offset1_wrap() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "group", extent = [2]>, <name = "device", extent = [3]>>,
    kind = axis_neighbor, componentName = "group",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = true}>
} {
  return
}

// CHECK: two_components_device_axis0_offset1_bounded: 6 devices, 4 edges, both endpoint roles verified
func.func @two_components_device_axis0_offset1_bounded() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "group", extent = [2]>, <name = "device", extent = [3]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = false}>
} {
  return
}

// CHECK: two_components_device_axis0_offset2_bounded: 6 devices, 2 edges, both endpoint roles verified
func.func @two_components_device_axis0_offset2_bounded() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "group", extent = [2]>, <name = "device", extent = [3]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 0 : i64, offset = 2 : i64, wrap = false}>
} {
  return
}

// CHECK: two_components_device_axis0_offset1_wrap: 6 devices, 6 edges, both endpoint roles verified
func.func @two_components_device_axis0_offset1_wrap() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "group", extent = [2]>, <name = "device", extent = [3]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = true}>
} {
  return
}

// CHECK: two_components_device_axis0_offset2_wrap: 6 devices, 6 edges, both endpoint roles verified
func.func @two_components_device_axis0_offset2_wrap() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "group", extent = [2]>, <name = "device", extent = [3]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 0 : i64, offset = 2 : i64, wrap = true}>
} {
  return
}

// CHECK: three_components_outer_axis0_offset1_bounded: 48 devices, 24 edges, both endpoint roles verified
func.func @three_components_outer_axis0_offset1_bounded() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "outer",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = false}>
} {
  return
}

// CHECK: three_components_outer_axis0_offset1_wrap: 48 devices, 48 edges, both endpoint roles verified
func.func @three_components_outer_axis0_offset1_wrap() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "outer",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = true}>
} {
  return
}

// CHECK: three_components_group_axis0_offset1_bounded: 48 devices, 24 edges, both endpoint roles verified
func.func @three_components_group_axis0_offset1_bounded() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "group",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = false}>
} {
  return
}

// CHECK: three_components_group_axis0_offset1_wrap: 48 devices, 48 edges, both endpoint roles verified
func.func @three_components_group_axis0_offset1_wrap() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "group",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = true}>
} {
  return
}

// CHECK: three_components_group_axis1_offset1_bounded: 48 devices, 32 edges, both endpoint roles verified
func.func @three_components_group_axis1_offset1_bounded() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "group",
    properties = {axis = 1 : i64, offset = 1 : i64, wrap = false}>
} {
  return
}

// CHECK: three_components_group_axis1_offset2_bounded: 48 devices, 16 edges, both endpoint roles verified
func.func @three_components_group_axis1_offset2_bounded() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "group",
    properties = {axis = 1 : i64, offset = 2 : i64, wrap = false}>
} {
  return
}

// CHECK: three_components_group_axis1_offset1_wrap: 48 devices, 48 edges, both endpoint roles verified
func.func @three_components_group_axis1_offset1_wrap() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "group",
    properties = {axis = 1 : i64, offset = 1 : i64, wrap = true}>
} {
  return
}

// CHECK: three_components_group_axis1_offset2_wrap: 48 devices, 48 edges, both endpoint roles verified
func.func @three_components_group_axis1_offset2_wrap() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "group",
    properties = {axis = 1 : i64, offset = 2 : i64, wrap = true}>
} {
  return
}

// CHECK: three_components_device_axis0_offset1_bounded: 48 devices, 24 edges, both endpoint roles verified
func.func @three_components_device_axis0_offset1_bounded() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = false}>
} {
  return
}

// CHECK: three_components_device_axis0_offset1_wrap: 48 devices, 48 edges, both endpoint roles verified
func.func @three_components_device_axis0_offset1_wrap() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = true}>
} {
  return
}

// CHECK: three_components_device_axis1_offset1_bounded: 48 devices, 24 edges, both endpoint roles verified
func.func @three_components_device_axis1_offset1_bounded() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 1 : i64, offset = 1 : i64, wrap = false}>
} {
  return
}

// CHECK: three_components_device_axis1_offset1_wrap: 48 devices, 48 edges, both endpoint roles verified
func.func @three_components_device_axis1_offset1_wrap() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "outer", extent = [2]>, <name = "group", extent = [2, 3]>, <name = "device", extent = [2, 2]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 1 : i64, offset = 1 : i64, wrap = true}>
} {
  return
}
