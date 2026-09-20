// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// RUN: ttlang-transfer-graph-test %s | FileCheck %s
//
// Shared component-index helpers must preserve edge order and endpoints for
// every graph kind, including nonzero roots and negative stencil offsets.

#domain = #ttl.device_domain<components = <name = "group", extent = [2, 3]>, <name = "device", extent = [2]>>

// CHECK: gather: 12 devices, 10 edges, both endpoint roles verified
func.func @gather() attributes {
  test.graph = #ttl.transfer_graph<domain = #domain,
    kind = gather, componentName = "group",
    properties = {root = #ttl.device_ref<coordinates = [1, 1]>}>
} { return }

// CHECK: scatter: 12 devices, 10 edges, both endpoint roles verified
func.func @scatter() attributes {
  test.graph = #ttl.transfer_graph<domain = #domain,
    kind = scatter, componentName = "group",
    properties = {source = #ttl.device_ref<coordinates = [1, 1]>}>
} { return }

// CHECK: all_to_all: 12 devices, 60 edges, both endpoint roles verified
func.func @all_to_all() attributes {
  test.graph = #ttl.transfer_graph<domain = #domain,
    kind = all_to_all, componentName = "group", properties = {}>
} { return }

// CHECK: stencil_bounded: 12 devices, 22 edges, both endpoint roles verified
func.func @stencil_bounded() attributes {
  test.graph = #ttl.transfer_graph<domain = #domain,
    kind = stencil, componentName = "group",
    properties = {offsets = [array<i64: 0, -1>, array<i64: 1, 0>, array<i64: 0, 1>], wrap = false}>
} { return }

// CHECK: stencil_wrapped: 12 devices, 36 edges, both endpoint roles verified
func.func @stencil_wrapped() attributes {
  test.graph = #ttl.transfer_graph<domain = #domain,
    kind = stencil, componentName = "group",
    properties = {offsets = [array<i64: 0, -1>, array<i64: 1, 0>, array<i64: 0, 1>], wrap = true}>
} { return }

#product_domain = #ttl.device_domain<
  components = <name = "host", extent = [2]>,
               <name = "group", extent = [2, 3]>,
               <name = "device", extent = [2]>>

// CHECK: stencil_middle_component: 24 devices, 44 edges, both endpoint roles verified
func.func @stencil_middle_component() attributes {
  test.graph = #ttl.transfer_graph<domain = #product_domain,
    kind = stencil, componentName = "group",
    properties = {offsets = [array<i64: 0, -1>, array<i64: 1, 0>, array<i64: 0, 1>], wrap = false}>
} { return }

// CHECK: explicit: 12 devices, 4 edges, both endpoint roles verified
func.func @explicit() attributes {
  test.graph = #ttl.transfer_graph<domain = #domain,
    kind = explicit, properties = {
      edges = [#ttl.transfer_edge<source = <coordinates = [1, 2], [1]>, destination = <coordinates = [0, 0], [0]>>,
               #ttl.transfer_edge<source = <coordinates = [0, 0], [0]>, destination = <coordinates = [1, 2], [1]>>,
               #ttl.transfer_edge<source = <coordinates = [1, 2], [1]>, destination = <coordinates = [0, 1], [1]>>,
               #ttl.transfer_edge<source = <coordinates = [0, 1], [1]>, destination = <coordinates = [0, 0], [0]>>]}>
} { return }

// CHECK: large_wrapped_offset: 12 devices, 12 edges, both endpoint roles verified
func.func @large_wrapped_offset() attributes {
  test.graph = #ttl.transfer_graph<domain = #domain,
    kind = axis_neighbor, componentName = "group",
    properties = {axis = 1 : i64, offset = 5 : i64, wrap = true}>
} { return }

// CHECK: single_component: 6 devices, 3 edges, both endpoint roles verified
func.func @single_component() attributes {
  test.graph = #ttl.transfer_graph<
    domain = <components = <name = "device", extent = [2, 3]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = false}>
} { return }
