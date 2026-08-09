// Verify that routing-plane operations translate to the tt-fabric linear API.
// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK: #include "tt_metal/fabric/hw/inc/linear/api.h"
// CHECK: static __attribute__((noinline)) void
// CHECK: routing_plane_atomic_inc(
// CHECK: packet_header->to_noc_unicast_atomic_inc(
// CHECK: sender.send_payload_flush_blocking_from_address(
// CHECK-LABEL: void kernel_main() {
// CHECK: tt::tt_fabric::RoutingPlaneConnectionManager [[MANAGER:.*]];
// CHECK: size_t [[ARG_INDEX:.*]] = 4;
// CHECK: uint32_t [[ROUTE_ID:.*]] = 0;
// CHECK: if ([[COUNT:.*]] != 0) {
// CHECK-NEXT: open_connections([[MANAGER]], [[COUNT]], [[ARG_INDEX]]);
// CHECK-NEXT: PacketHeaderPool::reset();
// CHECK-NEXT: [[ROUTE_ID]] = PacketHeaderPool::allocate_header_n([[COUNT]]);
// CHECK: experimental::routing_plane_atomic_inc([[MANAGER]], [[ROUTE_ID]], [[INDEX:[^,]+]], [[DEST_DEVICE:[^,]+]], [[DEST_MESH:[^,]+]],
// CHECK: auto *packet_header = PacketHeaderPool::header_table[[[ROUTE_ID]]].first + [[INDEX]];
// CHECK-NEXT: #if defined(FABRIC_2D)
// CHECK-NEXT: tt::tt_fabric::fabric_set_unicast_route(
// CHECK-NEXT: packet_header, static_cast<uint16_t>([[DEST_DEVICE]]), static_cast<uint16_t>([[DEST_MESH]]));
// CHECK-NEXT: #else
// CHECK-NEXT: tt::tt_fabric::fabric_set_unicast_route(
// CHECK-NEXT: packet_header, static_cast<uint16_t>([[DEST_DEVICE]]));
// CHECK-NEXT: #endif
// CHECK-NEXT: auto &sender = [[MANAGER]].get(static_cast<uint8_t>([[INDEX]])).sender;
// CHECK: packet_header->to_noc_fused_unicast_write_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
// CHECK: sender.send_payload_without_header_non_blocking_from_address(
// CHECK: sender.send_payload_flush_blocking_from_address(
// CHECK: if ([[COUNT]] != 0) {
// CHECK-NEXT: close_connections([[MANAGER]]);
// CHECK-NOT: }});
// CHECK-NOT:   }}

module {
  func.func @routing_plane() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %count = arith.constant 1 : i32
    %connection_index = arith.constant 0 : i32
    %destination_device_id = arith.constant 2 : i32
    %destination_mesh_id = arith.constant 3 : i32
    %node_x = arith.constant 2 : index
    %node_y = arith.constant 3 : index
    %semaphore = arith.constant 4096 : i32
    %noc = arith.constant 0 : i8
    %source = arith.constant 8192 : i32
    %size = arith.constant 1024 : i32
    %destination = arith.constant 12288 : i32
    %increment = arith.constant 1 : i32
    %semaphore_address = ttkernel.get_noc_addr(
      %node_x, %node_y, %semaphore, %noc)
      : (index, index, i32, i8) -> !ttkernel.noc_addr
    %destination_address = ttkernel.get_noc_addr(
      %node_x, %node_y, %destination, %noc)
      : (index, index, i32, i8) -> !ttkernel.noc_addr
    %manager = ttkernel.routing_plane.create_connection_manager
      : !ttkernel.routing_plane_connection_manager
    %route_id = ttkernel.routing_plane.open_connections
      %manager, %count runtime_arg_base = 4
      : (!ttkernel.routing_plane_connection_manager, i32) -> i32
    ttkernel.routing_plane.atomic_inc(
      %manager, %route_id, %connection_index, %destination_device_id,
      %destination_mesh_id, %semaphore_address, %increment)
      : (!ttkernel.routing_plane_connection_manager, i32, i32, i32, i32,
         !ttkernel.noc_addr, i32) -> ()
    ttkernel.routing_plane.fused_write_atomic_inc(
      %manager, %route_id, %connection_index, %destination_device_id,
      %destination_mesh_id, %source, %size,
      %destination_address, %semaphore_address, %increment)
      : (!ttkernel.routing_plane_connection_manager, i32, i32, i32, i32,
         i32, i32,
         !ttkernel.noc_addr, !ttkernel.noc_addr, i32) -> ()
    ttkernel.routing_plane.close_connections(%manager, %count)
      : (!ttkernel.routing_plane_connection_manager, i32) -> ()
    func.return
  }
}
