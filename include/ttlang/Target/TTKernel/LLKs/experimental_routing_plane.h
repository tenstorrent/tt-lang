// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_H

namespace experimental {

// Keep the fabric command sequence out of device-specific transfer branches.
// A call is smaller than repeating the sequence for every logical transfer.
static __attribute__((noinline)) void routing_plane_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager &manager, uint32_t route_id,
    uint32_t connection_index, uint32_t destination_device_id,
    uint32_t destination_mesh_id, uint32_t destination_hop_count,
    uint64_t semaphore_address, uint32_t increment) {
  auto *packet_header =
      PacketHeaderPool::header_table[route_id].first + connection_index;
#if defined(FABRIC_2D)
  tt::tt_fabric::fabric_set_unicast_route(
      packet_header, static_cast<uint16_t>(destination_device_id),
      static_cast<uint16_t>(destination_mesh_id));
#else
  tt::tt_fabric::fabric_set_unicast_route<false>(
      packet_header, static_cast<uint16_t>(destination_hop_count));
#endif
  auto &sender = manager.get(static_cast<uint8_t>(connection_index)).sender;
  packet_header->to_noc_unicast_atomic_inc(
      tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_address,
                                                      increment});
  sender.wait_for_empty_write_slot();
  sender.send_payload_flush_blocking_from_address(
      reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));
}

static __attribute__((noinline)) void
routing_plane_write(tt::tt_fabric::RoutingPlaneConnectionManager &manager,
                    uint32_t route_id, uint32_t connection_index,
                    uint32_t destination_device_id,
                    uint32_t destination_mesh_id,
                    uint32_t destination_hop_count, uint32_t source_address,
                    uint32_t size_bytes, uint64_t destination_address) {
  auto *packet_header =
      PacketHeaderPool::header_table[route_id].first + connection_index;
#if defined(FABRIC_2D)
  tt::tt_fabric::fabric_set_unicast_route(
      packet_header, static_cast<uint16_t>(destination_device_id),
      static_cast<uint16_t>(destination_mesh_id));
#else
  tt::tt_fabric::fabric_set_unicast_route<false>(
      packet_header, static_cast<uint16_t>(destination_hop_count));
#endif
  auto &sender = manager.get(static_cast<uint8_t>(connection_index)).sender;
  const uint32_t max_packet_size = tt::tt_fabric::get_fabric_max_packet_size();
  while (size_bytes > 0) {
    const uint32_t packet_size =
        size_bytes > max_packet_size ? max_packet_size : size_bytes;
    packet_header->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{destination_address},
        packet_size);
    sender.wait_for_empty_write_slot();
    sender.send_payload_without_header_non_blocking_from_address(source_address,
                                                                 packet_size);
    sender.send_payload_flush_blocking_from_address(
        reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));
    source_address += packet_size;
    destination_address += packet_size;
    size_bytes -= packet_size;
  }
}

static __attribute__((noinline)) void routing_plane_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager &manager, uint32_t routeId,
    uint32_t connectionIndex, uint32_t destinationDeviceId,
    uint32_t destinationMeshId, uint32_t destinationHopCount,
    uint32_t sourceAddress, uint32_t chunkSizeBytes, uint32_t chunkCount,
    uint64_t destinationAddress0, uint64_t destinationAddress1,
    uint64_t destinationAddress2, uint64_t destinationAddress3) {
  ASSERT(chunkCount > 0 && chunkCount <= NOC_SCATTER_WRITE_MAX_CHUNKS);
  auto *packetHeader =
      PacketHeaderPool::header_table[routeId].first + connectionIndex;
#if defined(FABRIC_2D)
  tt::tt_fabric::fabric_set_unicast_route(
      packetHeader, static_cast<uint16_t>(destinationDeviceId),
      static_cast<uint16_t>(destinationMeshId));
#else
  tt::tt_fabric::fabric_set_unicast_route<false>(
      packetHeader, static_cast<uint16_t>(destinationHopCount));
#endif
  auto &sender = manager.get(static_cast<uint8_t>(connectionIndex)).sender;
  const uint32_t maxPacketSize = tt::tt_fabric::get_fabric_max_packet_size();
  ASSERT(chunkSizeBytes > 0 && chunkSizeBytes <= maxPacketSize);
  const uint32_t availableChunksPerPacket = maxPacketSize / chunkSizeBytes;
  const uint32_t maxChunksPerPacket =
      availableChunksPerPacket < NOC_SCATTER_WRITE_MAX_CHUNKS
          ? availableChunksPerPacket
          : NOC_SCATTER_WRITE_MAX_CHUNKS;
  uint64_t destinationAddresses[NOC_SCATTER_WRITE_MAX_CHUNKS] = {
      destinationAddress0, destinationAddress1, destinationAddress2,
      destinationAddress3};
  uint16_t chunkSizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {
      static_cast<uint16_t>(chunkSizeBytes),
      static_cast<uint16_t>(chunkSizeBytes),
      static_cast<uint16_t>(chunkSizeBytes)};

  uint32_t firstChunk = 0;
  while (firstChunk < chunkCount) {
    const uint32_t remainingChunkCount = chunkCount - firstChunk;
    const uint32_t packetChunkCount = maxChunksPerPacket < remainingChunkCount
                                          ? maxChunksPerPacket
                                          : remainingChunkCount;
    if (packetChunkCount == 1) {
      packetHeader->to_noc_unicast_write(
          tt::tt_fabric::NocUnicastCommandHeader{
              destinationAddresses[firstChunk]},
          chunkSizeBytes);
      sender.wait_for_empty_write_slot();
      sender.send_payload_without_header_non_blocking_from_address(
          sourceAddress, chunkSizeBytes);
      sender.send_payload_flush_non_blocking_from_address(
          reinterpret_cast<uint32_t>(packetHeader), sizeof(PACKET_HEADER_TYPE));
    } else {
      const uint16_t packetSizeBytes =
          static_cast<uint16_t>(chunkSizeBytes * packetChunkCount);
      packetHeader->to_noc_unicast_scatter_write(
          tt::tt_fabric::NocUnicastScatterCommandHeader(
              destinationAddresses + firstChunk, chunkSizes,
              static_cast<uint8_t>(packetChunkCount)),
          packetSizeBytes);
      // The stateful send preserves the NoC command selected above.
      tt::tt_fabric::linear::experimental::
          fabric_unicast_noc_scatter_write_with_state<
              UnicastScatterWriteUpdateMask::None>(&sender, packetHeader,
                                                   sourceAddress);
    }
    // The next packet or completion command reuses this packet header.
    noc_async_writes_flushed();
    sourceAddress += packetChunkCount * chunkSizeBytes;
    firstChunk += packetChunkCount;
  }
}

static __attribute__((noinline)) void routing_plane_fused_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager &manager, uint32_t routeId,
    uint32_t connectionIndex, uint32_t destinationDeviceId,
    uint32_t destinationMeshId, uint32_t destinationHopCount,
    uint32_t sourceAddress, uint32_t sizeBytes, uint64_t destinationAddress,
    uint64_t semaphoreAddress, uint32_t increment) {
  auto *packetHeader =
      PacketHeaderPool::header_table[routeId].first + connectionIndex;
#if defined(FABRIC_2D)
  tt::tt_fabric::fabric_set_unicast_route(
      packetHeader, static_cast<uint16_t>(destinationDeviceId),
      static_cast<uint16_t>(destinationMeshId));
#else
  tt::tt_fabric::fabric_set_unicast_route<false>(
      packetHeader, static_cast<uint16_t>(destinationHopCount));
#endif
  auto &sender = manager.get(static_cast<uint8_t>(connectionIndex)).sender;
  const uint32_t maxPacketSize = tt::tt_fabric::get_fabric_max_packet_size();

  while (sizeBytes > maxPacketSize) {
    packetHeader->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{destinationAddress},
        maxPacketSize);
    sender.wait_for_empty_write_slot();
    sender.send_payload_without_header_non_blocking_from_address(sourceAddress,
                                                                 maxPacketSize);
    sender.send_payload_flush_blocking_from_address(
        reinterpret_cast<uint32_t>(packetHeader), sizeof(PACKET_HEADER_TYPE));
    sourceAddress += maxPacketSize;
    destinationAddress += maxPacketSize;
    sizeBytes -= maxPacketSize;
  }

  packetHeader->to_noc_fused_unicast_write_atomic_inc(
      tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
          destinationAddress, semaphoreAddress, increment, true},
      sizeBytes);
  sender.wait_for_empty_write_slot();
  sender.send_payload_without_header_non_blocking_from_address(sourceAddress,
                                                               sizeBytes);
  sender.send_payload_flush_blocking_from_address(
      reinterpret_cast<uint32_t>(packetHeader), sizeof(PACKET_HEADER_TYPE));
}

} // namespace experimental

#endif // TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_H
