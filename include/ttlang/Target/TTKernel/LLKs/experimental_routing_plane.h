// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_H

// fabric_config.h consumes PACKET_HEADER_TYPE from linear/api.h.
// clang-format off
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/fabric_config.h"
// clang-format on

#if defined(TTLANG_FABRIC_MUX_CLIENT)
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#endif

namespace experimental {

/// Selects direct or mux transport from host-planned runtime arguments.
class RoutingPlaneConnectionManager {
private:
  auto &directSender(uint32_t connectionIndex) {
    return directManager.get(static_cast<uint8_t>(connectionIndex)).sender;
  }

public:
  uint32_t open(uint32_t connectionCount, size_t runtimeArgBase) {
#if defined(TTLANG_FABRIC_MUX_CLIENT)
    muxActive = get_arg_val<uint32_t>(runtimeArgBase++) != 0;
    if (muxActive) {
      ASSERT(connectionCount == 1);
      openMux(runtimeArgBase);
      PacketHeaderPool::reset();
      muxPacketHeader = PacketHeaderPool::allocate_header();
      return 0;
    }
#endif
    open_connections(directManager, connectionCount, runtimeArgBase);
    PacketHeaderPool::reset();
    return PacketHeaderPool::allocate_header_n(connectionCount);
  }

  void close(uint32_t) {
#if defined(TTLANG_FABRIC_MUX_CLIENT)
    if (muxActive) {
      closeMux();
      return;
    }
#endif
    close_connections(directManager);
  }

  volatile tt_l1_ptr PACKET_HEADER_TYPE *
  packetHeader(uint32_t routeId, uint32_t connectionIndex) {
#if defined(TTLANG_FABRIC_MUX_CLIENT)
    if (muxActive) {
      ASSERT(connectionIndex == 0);
      return muxPacketHeader;
    }
#endif
    return PacketHeaderPool::header_table[routeId].first + connectionIndex;
  }

  void waitForEmptyWriteSlot(uint32_t connectionIndex) {
#if defined(TTLANG_FABRIC_MUX_CLIENT)
    if (muxActive) {
      muxSender.wait_for_empty_write_slot();
      return;
    }
#endif
    directSender(connectionIndex).wait_for_empty_write_slot();
  }

  void sendPayloadWithoutHeaderNonBlockingFromAddress(uint32_t connectionIndex,
                                                      uint32_t sourceAddress,
                                                      uint32_t sizeBytes) {
#if defined(TTLANG_FABRIC_MUX_CLIENT)
    if (muxActive) {
      muxSender.send_payload_without_header_non_blocking_from_address(
          sourceAddress, sizeBytes);
      return;
    }
#endif
    directSender(connectionIndex)
        .send_payload_without_header_non_blocking_from_address(sourceAddress,
                                                               sizeBytes);
  }

  void sendPayloadFlushBlockingFromAddress(uint32_t connectionIndex,
                                           uint32_t sourceAddress,
                                           uint32_t sizeBytes) {
#if defined(TTLANG_FABRIC_MUX_CLIENT)
    if (muxActive) {
      muxSender.send_payload_flush_blocking_from_address(sourceAddress,
                                                         sizeBytes);
      return;
    }
#endif
    directSender(connectionIndex)
        .send_payload_flush_blocking_from_address(sourceAddress, sizeBytes);
  }

  void sendPayloadFlushNonBlockingFromAddress(uint32_t connectionIndex,
                                              uint32_t sourceAddress,
                                              uint32_t sizeBytes) {
#if defined(TTLANG_FABRIC_MUX_CLIENT)
    if (muxActive) {
      muxSender.send_payload_flush_non_blocking_from_address(sourceAddress,
                                                             sizeBytes);
      return;
    }
#endif
    directSender(connectionIndex)
        .send_payload_flush_non_blocking_from_address(sourceAddress, sizeBytes);
  }

  void sendScatterWrite(uint32_t connectionIndex,
                        volatile tt_l1_ptr PACKET_HEADER_TYPE *packetHeader,
                        uint32_t sourceAddress) {
#if defined(TTLANG_FABRIC_MUX_CLIENT)
    if (muxActive) {
      tt::tt_fabric::linear::experimental::
          fabric_unicast_noc_scatter_write_with_state<
              UnicastScatterWriteUpdateMask::None>(&muxSender, packetHeader,
                                                   sourceAddress);
      return;
    }
#endif
    auto &sender = directSender(connectionIndex);
    tt::tt_fabric::linear::experimental::
        fabric_unicast_noc_scatter_write_with_state<
            UnicastScatterWriteUpdateMask::None>(&sender, packetHeader,
                                                 sourceAddress);
  }

private:
#if defined(TTLANG_FABRIC_MUX_CLIENT)
  static constexpr uint8_t muxNumBuffers = TTLANG_FABRIC_MUX_NUM_BUFFERS;

  void openMux(size_t runtimeArgBase) {
    const bool connectionValid = get_arg_val<uint32_t>(runtimeArgBase++) != 0;
    terminationMaster = get_arg_val<uint32_t>(runtimeArgBase++) != 0;
    ASSERT(connectionValid);
    const uint8_t muxX = get_arg_val<uint32_t>(runtimeArgBase++);
    const uint8_t muxY = get_arg_val<uint32_t>(runtimeArgBase++);
    const uint32_t muxChannelBase = get_arg_val<uint32_t>(runtimeArgBase++);
    const uint32_t muxConnectionInfo = get_arg_val<uint32_t>(runtimeArgBase++);
    const uint32_t muxConnectionHandshake =
        get_arg_val<uint32_t>(runtimeArgBase++);
    const uint32_t muxFlowControl = get_arg_val<uint32_t>(runtimeArgBase++);
    const uint32_t muxBufferIndex = get_arg_val<uint32_t>(runtimeArgBase++);
    const uint8_t muxChannelId = get_arg_val<uint32_t>(runtimeArgBase++);
    terminationSyncAddress =
        get_semaphore(get_arg_val<uint32_t>(runtimeArgBase++));
    const uint32_t localMuxStatusAddress =
        get_semaphore(get_arg_val<uint32_t>(runtimeArgBase++));
    const uint32_t localFlowControlAddress =
        get_semaphore(get_arg_val<uint32_t>(runtimeArgBase++));
    const uint32_t localTeardownAddress =
        get_semaphore(get_arg_val<uint32_t>(runtimeArgBase++));
    const uint32_t localBufferIndexAddress =
        get_semaphore(get_arg_val<uint32_t>(runtimeArgBase++));
    terminationMasterX = get_arg_val<uint32_t>(runtimeArgBase++);
    terminationMasterY = get_arg_val<uint32_t>(runtimeArgBase++);
    muxClientCount = get_arg_val<uint32_t>(runtimeArgBase++);

    muxSender =
        tt::tt_fabric::build_connection_to_fabric_endpoint<muxNumBuffers>(
            muxX, muxY, muxChannelId, muxNumBuffers,
            TTLANG_FABRIC_MUX_CHANNEL_BUFFER_SIZE_BYTES, muxChannelBase,
            muxConnectionInfo, muxConnectionHandshake, muxFlowControl,
            muxBufferIndex, localFlowControlAddress, localTeardownAddress,
            localBufferIndexAddress);
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        muxX, muxY, TTLANG_FABRIC_MUX_STATUS_ADDRESS, localMuxStatusAddress);
    tt::tt_fabric::fabric_client_connect(muxSender);
    fabricMuxX = muxX;
    fabricMuxY = muxY;
  }

  void closeMux() {
    // Disconnect only after the mux has consumed every submitted packet.
    while (muxSender.get_num_free_write_slots() != muxNumBuffers) {
    }
    tt::tt_fabric::fabric_client_disconnect(muxSender);
    if (terminationMaster) {
      auto *terminationSync = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
          terminationSyncAddress);
      noc_semaphore_wait(terminationSync, muxClientCount - 1);
      tt::tt_fabric::fabric_endpoint_terminate(
          fabricMuxX, fabricMuxY, TTLANG_FABRIC_MUX_TERMINATION_SIGNAL_ADDRESS);
      return;
    }
    noc_semaphore_inc(get_noc_addr(terminationMasterX, terminationMasterY,
                                   terminationSyncAddress),
                      1);
    noc_async_atomic_barrier();
  }

  bool muxActive = false;
  bool terminationMaster = false;
  uint8_t fabricMuxX = 0;
  uint8_t fabricMuxY = 0;
  uint32_t terminationSyncAddress = 0;
  uint32_t terminationMasterX = 0;
  uint32_t terminationMasterY = 0;
  uint32_t muxClientCount = 0;
  volatile tt_l1_ptr PACKET_HEADER_TYPE *muxPacketHeader = nullptr;
  tt::tt_fabric::WorkerToFabricMuxSender<muxNumBuffers> muxSender;
#endif
  tt::tt_fabric::RoutingPlaneConnectionManager directManager;
};

// Keep the fabric command sequence out of device-specific transfer branches.
// A call is smaller than repeating the sequence for every logical transfer.
static __attribute__((noinline)) void routing_plane_atomic_inc(
    RoutingPlaneConnectionManager &manager, uint32_t route_id,
    uint32_t connection_index, uint32_t destination_device_id,
    uint32_t destination_mesh_id, uint32_t destination_hop_count,
    uint64_t semaphore_address, uint32_t increment) {
  auto *packet_header = manager.packetHeader(route_id, connection_index);
#if defined(FABRIC_2D)
  tt::tt_fabric::fabric_set_unicast_route(
      packet_header, static_cast<uint16_t>(destination_device_id),
      static_cast<uint16_t>(destination_mesh_id));
#else
  tt::tt_fabric::fabric_set_unicast_route<false>(
      packet_header, static_cast<uint16_t>(destination_hop_count));
#endif
  packet_header->to_noc_unicast_atomic_inc(
      tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_address,
                                                      increment});
  manager.waitForEmptyWriteSlot(connection_index);
  manager.sendPayloadFlushBlockingFromAddress(
      connection_index, reinterpret_cast<uint32_t>(packet_header),
      sizeof(PACKET_HEADER_TYPE));
}

static __attribute__((noinline)) void routing_plane_fused_write_atomic_inc(
    RoutingPlaneConnectionManager &manager, uint32_t routeId,
    uint32_t connectionIndex, uint32_t destinationDeviceId,
    uint32_t destinationMeshId, uint32_t destinationHopCount,
    uint32_t sourceAddress, uint32_t sizeBytes, uint64_t destinationAddress,
    uint64_t semaphoreAddress, uint32_t increment) {
  auto *packetHeader = manager.packetHeader(routeId, connectionIndex);
#if defined(FABRIC_2D)
  tt::tt_fabric::fabric_set_unicast_route(
      packetHeader, static_cast<uint16_t>(destinationDeviceId),
      static_cast<uint16_t>(destinationMeshId));
#else
  tt::tt_fabric::fabric_set_unicast_route<false>(
      packetHeader, static_cast<uint16_t>(destinationHopCount));
#endif
  const uint32_t maxPacketSize = tt::tt_fabric::get_fabric_max_packet_size();

  while (sizeBytes > maxPacketSize) {
    packetHeader->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{destinationAddress},
        maxPacketSize);
    manager.waitForEmptyWriteSlot(connectionIndex);
    manager.sendPayloadWithoutHeaderNonBlockingFromAddress(
        connectionIndex, sourceAddress, maxPacketSize);
    manager.sendPayloadFlushNonBlockingFromAddress(
        connectionIndex, reinterpret_cast<uint32_t>(packetHeader),
        sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();
    sourceAddress += maxPacketSize;
    destinationAddress += maxPacketSize;
    sizeBytes -= maxPacketSize;
  }

  packetHeader->to_noc_fused_unicast_write_atomic_inc(
      tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
          destinationAddress, semaphoreAddress, increment, true},
      sizeBytes);
  manager.waitForEmptyWriteSlot(connectionIndex);
  manager.sendPayloadWithoutHeaderNonBlockingFromAddress(
      connectionIndex, sourceAddress, sizeBytes);
  manager.sendPayloadFlushBlockingFromAddress(
      connectionIndex, reinterpret_cast<uint32_t>(packetHeader),
      sizeof(PACKET_HEADER_TYPE));
}

} // namespace experimental

#endif // TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROUTING_PLANE_H
