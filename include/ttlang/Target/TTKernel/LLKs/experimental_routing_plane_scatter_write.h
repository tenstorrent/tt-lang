// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_SCATTER_WRITE_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_SCATTER_WRITE_H

namespace experimental {

// Group destination chunks into the fewest packets permitted by the fabric
// packet and scatter limits.
FORCE_INLINE void routing_plane_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager &manager, uint32_t route_id,
    uint32_t connection_index, uint32_t destination_device_id,
    uint32_t destination_mesh_id, uint32_t destination_hop_count,
    uint32_t source_address, uint32_t chunk_size_bytes, uint32_t chunk_count,
    uint64_t destination_address_0, uint64_t destination_address_1,
    uint64_t destination_address_2, uint64_t destination_address_3) {
  ASSERT(chunk_count > 0 && chunk_count <= NOC_SCATTER_WRITE_MAX_CHUNKS);
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
  ASSERT(chunk_size_bytes > 0 && chunk_size_bytes <= max_packet_size);
  const uint32_t available_chunks_per_packet =
      max_packet_size / chunk_size_bytes;
  const uint32_t max_chunks_per_packet =
      available_chunks_per_packet < NOC_SCATTER_WRITE_MAX_CHUNKS
          ? available_chunks_per_packet
          : NOC_SCATTER_WRITE_MAX_CHUNKS;
  uint64_t destination_addresses[NOC_SCATTER_WRITE_MAX_CHUNKS] = {
      destination_address_0, destination_address_1, destination_address_2,
      destination_address_3};
  uint16_t chunk_sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {
      static_cast<uint16_t>(chunk_size_bytes),
      static_cast<uint16_t>(chunk_size_bytes),
      static_cast<uint16_t>(chunk_size_bytes)};

  uint32_t first_chunk = 0;
  while (first_chunk < chunk_count) {
    const uint32_t remaining_chunk_count = chunk_count - first_chunk;
    const uint32_t packet_chunk_count =
        max_chunks_per_packet < remaining_chunk_count ? max_chunks_per_packet
                                                      : remaining_chunk_count;
    if (packet_chunk_count == 1) {
      packet_header->to_noc_unicast_write(
          tt::tt_fabric::NocUnicastCommandHeader{
              destination_addresses[first_chunk]},
          chunk_size_bytes);
      sender.wait_for_empty_write_slot();
      sender.send_payload_without_header_non_blocking_from_address(
          source_address, chunk_size_bytes);
      sender.send_payload_flush_non_blocking_from_address(
          reinterpret_cast<uint32_t>(packet_header),
          sizeof(PACKET_HEADER_TYPE));
    } else {
      const uint16_t packet_size_bytes =
          static_cast<uint16_t>(chunk_size_bytes * packet_chunk_count);
      packet_header->to_noc_unicast_scatter_write(
          tt::tt_fabric::NocUnicastScatterCommandHeader(
              destination_addresses + first_chunk, chunk_sizes,
              static_cast<uint8_t>(packet_chunk_count)),
          packet_size_bytes);
      // The stateful send preserves the NoC command selected above.
      tt::tt_fabric::linear::experimental::
          fabric_unicast_noc_scatter_write_with_state<
              UnicastScatterWriteUpdateMask::None>(&sender, packet_header,
                                                   source_address);
    }
    // The next packet or completion command reuses this packet header.
    noc_async_writes_flushed();
    source_address += packet_chunk_count * chunk_size_bytes;
    first_chunk += packet_chunk_count;
  }
}

} // namespace experimental

#endif // TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_SCATTER_WRITE_H
