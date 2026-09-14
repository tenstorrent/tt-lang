// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_SCATTER_WRITE_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_SCATTER_WRITE_H

namespace experimental {

// Preserve contiguous source data while splitting destination chunks across
// packets at the active packet limit.
FORCE_INLINE void routing_plane_scatter_write(
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

} // namespace experimental

#endif // TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_SCATTER_WRITE_H
