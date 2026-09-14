// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_WRITE_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_WRITE_H

namespace experimental {

// Split transfers at the active packet limit required by the fabric interface.
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

} // namespace experimental

#endif // TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_WRITE_H
