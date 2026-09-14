// Verify that routing-plane operations translate to the tt-fabric linear API.
// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK: #include "tt_metal/fabric/hw/inc/linear/api.h"
// CHECK: #include "tt_metal/fabric/hw/inc/fabric_config.h"
// CHECK: static __attribute__((noinline)) void
// CHECK: routing_plane_atomic_inc(
// CHECK: packet_header->to_noc_unicast_atomic_inc(
// CHECK: manager.sendPayloadFlushBlockingFromAddress(
// CHECK-LABEL: static __attribute__((noinline)) void routing_plane_fused_write_atomic_inc(
// CHECK: const uint32_t [[MAX_PACKET_SIZE:.*]] = tt::tt_fabric::get_fabric_max_packet_size();
// CHECK: while (sizeBytes > [[MAX_PACKET_SIZE]]) {
// CHECK: packetHeader->to_noc_unicast_write(
// CHECK: manager.sendPayloadWithoutHeaderNonBlockingFromAddress(
// CHECK-NEXT: [[MAX_PACKET_SIZE]]);
// CHECK: sourceAddress += [[MAX_PACKET_SIZE]];
// CHECK-NEXT: destinationAddress += [[MAX_PACKET_SIZE]];
// CHECK-NEXT: sizeBytes -= [[MAX_PACKET_SIZE]];
// CHECK: packetHeader->to_noc_fused_unicast_write_atomic_inc(
// CHECK: manager.sendPayloadWithoutHeaderNonBlockingFromAddress(
// CHECK-NEXT: sizeBytes);
// CHECK: static __attribute__((noinline)) void
// CHECK-NEXT: routing_plane_write(
// CHECK: packet_header->to_noc_unicast_write(
// CHECK: manager.sendPayloadWithoutHeaderNonBlockingFromAddress(
// CHECK-LABEL: FORCE_INLINE void routing_plane_scatter_write(
// CHECK: packetHeader->to_noc_unicast_scatter_write(
// CHECK: manager.sendScatterWrite(connectionIndex, packetHeader, sourceAddress);
// CHECK: noc_async_writes_flushed();
// CHECK-LABEL: void kernel_main() {
// CHECK: size_t [[RUNTIME_ARG_BASE:.*]] = 5;
// CHECK: experimental::RoutingPlaneConnectionManager [[MANAGER:.*]];
// CHECK: uint32_t [[ROUTE_ID:.*]] = 0;
// CHECK: if ([[COUNT:.*]] != 0) {
// CHECK-NEXT: [[ROUTE_ID]] = [[MANAGER]].open([[COUNT]], [[RUNTIME_ARG_BASE]]);
// CHECK: experimental::routing_plane_atomic_inc([[MANAGER]], [[ROUTE_ID]], [[INDEX:[^,]+]], [[DEST_DEVICE:[^,]+]], [[DEST_MESH:[^,]+]], [[HOPS:[^,]+]],
// CHECK: experimental::routing_plane_write([[MANAGER]], [[ROUTE_ID]], [[INDEX]], [[DEST_DEVICE]], [[DEST_MESH]], [[HOPS]],
// CHECK: experimental::routing_plane_scatter_write([[MANAGER]], [[ROUTE_ID]], [[INDEX]], [[DEST_DEVICE]], [[DEST_MESH]], [[HOPS]],
// CHECK: experimental::routing_plane_fused_write_atomic_inc([[MANAGER]], [[ROUTE_ID]], [[INDEX]], [[DEST_DEVICE]], [[DEST_MESH]], [[HOPS]],
// CHECK: if ([[COUNT]] != 0) {
// CHECK-NEXT: [[MANAGER]].close([[COUNT]]);

module {
  func.func @routing_plane() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %count = arith.constant 1 : i32
    %runtime_arg_base = arith.constant 5 : index
    %connection_index = arith.constant 0 : i32
    %destination_device_id = arith.constant 2 : i32
    %destination_mesh_id = arith.constant 3 : i32
    %destination_hop_count = arith.constant 4 : i32
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
      %manager, %count runtime_arg_base = %runtime_arg_base
      : (!ttkernel.routing_plane_connection_manager, i32, index) -> i32
    ttkernel.routing_plane.atomic_inc(
      %manager, %route_id, %connection_index, %destination_device_id,
      %destination_mesh_id, %destination_hop_count, %semaphore_address,
      %increment)
      : (!ttkernel.routing_plane_connection_manager, i32, i32, i32, i32, i32,
         !ttkernel.noc_addr, i32) -> ()
    ttkernel.routing_plane.write(
      %manager, %route_id, %connection_index, %destination_device_id,
      %destination_mesh_id, %destination_hop_count, %source, %size,
      %destination_address)
      : (!ttkernel.routing_plane_connection_manager, i32, i32, i32, i32, i32,
         i32, i32, !ttkernel.noc_addr) -> ()
    %chunk_count = arith.constant 4 : i32
    ttkernel.routing_plane.scatter_write(
      %manager, %route_id, %connection_index, %destination_device_id,
      %destination_mesh_id, %destination_hop_count, %source, %size,
      %chunk_count, %destination_address, %destination_address,
      %destination_address, %destination_address)
      : (!ttkernel.routing_plane_connection_manager, i32, i32, i32, i32, i32,
         i32, i32, i32, !ttkernel.noc_addr, !ttkernel.noc_addr,
         !ttkernel.noc_addr, !ttkernel.noc_addr) -> ()
    ttkernel.routing_plane.fused_write_atomic_inc(
      %manager, %route_id, %connection_index, %destination_device_id,
      %destination_mesh_id, %destination_hop_count, %source, %size,
      %destination_address, %semaphore_address, %increment)
      : (!ttkernel.routing_plane_connection_manager, i32, i32, i32, i32, i32,
         i32, i32,
         !ttkernel.noc_addr, !ttkernel.noc_addr, i32) -> ()
    ttkernel.routing_plane.close_connections(%manager, %count)
      : (!ttkernel.routing_plane_connection_manager, i32) -> ()
    func.return
  }
}
