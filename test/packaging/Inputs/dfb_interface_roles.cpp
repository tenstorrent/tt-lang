// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercises interface and geometry updates; synchronization is not executed.
#include <array>
#include <cassert>
#include <cstdint>

#define FORCE_INLINE inline
#define tt_l1_ptr
#define TTI_STALLWAIT(...) ((void)0)
#define TTI_SETDMAREG(...) ((void)0)
#define LO_16(value) (value)

namespace p_stall {
constexpr uint32_t UNPACK = 1, PACK = 2, STALL_TDMA = 3;
}
namespace p_gpr_unpack {
constexpr uint32_t TMP0 = 4;
}
namespace p_gpr_pack {
constexpr uint32_t TMP0 = 8;
}

constexpr uint32_t cb_addr_shift = 4;

struct LocalCBInterface {
  uint32_t fifo_rd_ptr;
  uint32_t fifo_wr_ptr;
  uint32_t fifo_wr_tile_ptr;
  uint32_t tiles_acked_received_init;
  uint32_t fifo_limit;
  uint32_t fifo_size;
  uint32_t fifo_num_pages;
  uint32_t fifo_page_size;
};

std::array<LocalCBInterface, 64> interfaces;
std::array<uint32_t, 64> received;
std::array<uint32_t, 64> acked;

inline LocalCBInterface &get_local_cb_interface(uint32_t index) {
  return interfaces.at(index);
}
inline uint32_t *get_cb_tiles_received_ptr(uint32_t index) {
  return &received.at(index);
}
inline uint32_t *get_cb_tiles_acked_ptr(uint32_t index) {
  return &acked.at(index);
}
inline void noc_async_full_barrier() {}
inline void tensix_sync() {}
inline void sync_regfile_write(uint32_t) {}

#if TEST_SHARED_GEOMETRY
struct GeometryUpdate {
  uint32_t index;
  uint32_t baseBytes;
  uint32_t pageBytes;
  uint32_t pages;
};

std::array<GeometryUpdate, 64> geometryUpdates;
uint32_t geometryUpdateCount = 0;

inline void __emule_cb_rebind_geometry(uint32_t index, uint32_t baseBytes,
                                       uint32_t pageBytes, uint32_t pages) {
  geometryUpdates.at(geometryUpdateCount++) = {index, baseBytes, pageBytes,
                                               pages};
}
#endif

#include "ttlang/Target/TTKernel/LLKs/experimental_dfb_reconfiguration.h"
#include "ttlang/Target/TTKernel/LLKs/experimental_dfb_reset.h"

void checkInterface(const LocalCBInterface &actual,
                    const LocalCBInterface &expected) {
  assert(actual.fifo_rd_ptr == expected.fifo_rd_ptr);
  assert(actual.fifo_wr_ptr == expected.fifo_wr_ptr);
  assert(actual.fifo_wr_tile_ptr == expected.fifo_wr_tile_ptr);
  assert(actual.tiles_acked_received_init ==
         expected.tiles_acked_received_init);
  assert(actual.fifo_limit == expected.fifo_limit);
  assert(actual.fifo_size == expected.fifo_size);
  assert(actual.fifo_num_pages == expected.fifo_num_pages);
  assert(actual.fifo_page_size == expected.fifo_page_size);
}

void initializeInterfaces() {
#if TEST_SHARED_GEOMETRY
  geometryUpdateCount = 0;
#endif
  for (uint32_t index = 0; index < interfaces.size(); ++index) {
    uint32_t base = 0x100 + index * 0x40;
    interfaces[index] = {base + 7,    base + 9, 11,         13,
                         base + 0x20, 0x20,     15 + index, 4};
    received[index] = 101 + index;
    acked[index] = 201 + index;
  }
}

using Configuration = std::array<uint32_t, 264>;

template <typename Configurations>
void checkReconfiguration(Configuration &configuration,
                          const Configuration &expectedConfiguration,
                          const std::array<uint32_t, 2> &mask) {
  initializeInterfaces();
  auto before = interfaces;
#if TEST_SHARED_GEOMETRY
  Configurations::rebindSharedGeometry(configuration.data());
  // Rebinding shared geometry must not update any RISC-local interface.
  for (uint32_t index = 0; index < interfaces.size(); ++index) {
    checkInterface(interfaces[index], before[index]);
  }
#endif
#if TEST_ACTIVE
  Configurations::template run<TEST_RECONFIG_READ, TEST_RECONFIG_WRITE,
                               TEST_RECONFIG_WRITE_TILE,
                               TEST_RECONFIG_RESET_COUNTERS>(
      configuration.data());
#else
  ::experimental::reconfigure_dfb_interfaces(0);
  ::experimental::reconfigure_dfb_interfaces<0>(0);
#endif
  uint32_t expectedUpdateCount = 0;
  for (uint32_t index = 0; index < interfaces.size(); ++index) {
    bool selected = ((mask[index / 32] >> (index % 32)) & 1U) != 0;
    auto expected = before[index];
    if (selected && TEST_ACTIVE) {
      uint32_t configuredAddress = expectedConfiguration[index * 4];
      uint32_t base = configuredAddress == 0
                          ? before[index].fifo_limit - before[index].fifo_size
                          : configuredAddress >> cb_addr_shift;
      uint32_t size = expectedConfiguration[index * 4 + 1] >> cb_addr_shift;
      uint32_t pages = expectedConfiguration[index * 4 + 2];
      uint32_t pageSize = expectedConfiguration[index * 4 + 3] >> cb_addr_shift;
      if (TEST_READ) {
        expected.fifo_rd_ptr = base;
      }
      if (TEST_WRITE) {
        expected.fifo_wr_ptr = base;
        expected.fifo_num_pages = pages;
      }
      if (TEST_WRITE_TILE) {
        expected.fifo_wr_tile_ptr = 0;
      }
      expected.tiles_acked_received_init = 0;
      expected.fifo_limit = base + size;
      expected.fifo_size = size;
      expected.fifo_page_size = pageSize;
#if TEST_SHARED_GEOMETRY
      assert(expectedUpdateCount < geometryUpdateCount);
      const auto &update = geometryUpdates.at(expectedUpdateCount++);
      assert(update.index == index);
      assert(update.baseBytes == (base << cb_addr_shift));
      assert(update.pageBytes == (pageSize << cb_addr_shift));
      assert(update.pages == pages);
#endif
    }
    checkInterface(interfaces[index], expected);
    bool resetCounters = selected && TEST_RESET_COUNTERS;
    assert(received[index] == (resetCounters ? 0 : 101 + index));
    assert(acked[index] == (resetCounters ? 0 : 201 + index));
  }
#if TEST_SHARED_GEOMETRY
  assert(geometryUpdateCount == expectedUpdateCount);
#endif
}

int main() {
  // Include an empty mask, each endpoint of both words, and a combined mask.
  constexpr std::array<std::array<uint32_t, 2>, 6> masks = {
      {{0, 0},
       {1, 0},
       {0x80000000U, 0},
       {0, 1},
       {0, 0x80000000U},
       {0x80000001U, 0x80000001U}}};
  Configuration configuration = {};
  for (uint32_t index = 0; index < interfaces.size(); ++index) {
    uint32_t base = 0x1000 + index * 0x100;
    uint32_t pages = 5 + index % 3;
    configuration[index * 4] = base << cb_addr_shift;
    configuration[index * 4 + 1] = (pages * 8) << cb_addr_shift;
    configuration[index * 4 + 2] = pages;
    configuration[index * 4 + 3] = 8 << cb_addr_shift;
  }

  for (const auto &mask : masks) {
    initializeInterfaces();
    auto before = interfaces;
    ::experimental::dfb_reset_detail::applyMask(mask[0], 0);
    ::experimental::dfb_reset_detail::applyMask(mask[1], 32);
    for (uint32_t index = 0; index < interfaces.size(); ++index) {
      bool selected = ((mask[index / 32] >> (index % 32)) & 1U) != 0;
      auto expected = before[index];
      if (selected && TEST_ACTIVE) {
        uint32_t base = 0x100 + index * 0x40;
        if (TEST_READ) {
          expected.fifo_rd_ptr = base;
        }
        if (TEST_WRITE) {
          expected.fifo_wr_ptr = base;
        }
        if (TEST_WRITE_TILE) {
          expected.fifo_wr_tile_ptr = 0;
        }
        expected.tiles_acked_received_init = 0;
      }
      checkInterface(interfaces[index], expected);
      bool resetCounters = selected && TEST_RESET_COUNTERS;
      assert(received[index] == (resetCounters ? 0 : 101 + index));
      assert(acked[index] == (resetCounters ? 0 : 201 + index));
    }

    for (bool preserveAddress : {false, true}) {
      auto runtimeConfiguration = configuration;
      runtimeConfiguration[256] = mask[0];
      runtimeConfiguration[257] = mask[1];
      if (preserveAddress) {
        for (uint32_t index = 0; index < interfaces.size(); ++index) {
          runtimeConfiguration[index * 4] = 0;
        }
      }
      checkReconfiguration<
          ::experimental::dfb_reconfiguration_detail::RuntimeConfigurations>(
          runtimeConfiguration, runtimeConfiguration, mask);
    }
  }

  for (bool preserveAddress : {false, true}) {
    auto runtimeConfiguration = configuration;
    if (preserveAddress) {
      for (uint32_t index = 0; index < interfaces.size(); ++index) {
        runtimeConfiguration[index * 4] = 0;
      }
    }
    // Static records, not the empty runtime masks or runtime geometry, select
    // the updates. Runtime records still supply the FIFO addresses.
    auto expectedConfiguration = runtimeConfiguration;
    constexpr std::array<std::array<uint32_t, 4>, 4> staticGeometry = {
        {{0, 96, 4, 24}, {31, 30, 3, 10}, {32, 70, 5, 14}, {63, 72, 9, 8}}};
    for (const auto &record : staticGeometry) {
      uint32_t index = record[0];
      expectedConfiguration[index * 4 + 1] = record[1] << cb_addr_shift;
      expectedConfiguration[index * 4 + 2] = record[2];
      expectedConfiguration[index * 4 + 3] = record[3] << cb_addr_shift;
    }
    checkReconfiguration<
        ::experimental::dfb_reconfiguration_detail::StaticConfigurations<>>(
        runtimeConfiguration, expectedConfiguration, {0, 0});
    checkReconfiguration<
        ::experimental::dfb_reconfiguration_detail::StaticConfigurations<
            0, 96 << cb_addr_shift, 4, 24 << cb_addr_shift>>(
        runtimeConfiguration, expectedConfiguration, {1, 0});
    checkReconfiguration<
        ::experimental::dfb_reconfiguration_detail::StaticConfigurations<
            0, 96 << cb_addr_shift, 4, 24 << cb_addr_shift, 31,
            30 << cb_addr_shift, 3, 10 << cb_addr_shift, 32,
            70 << cb_addr_shift, 5, 14 << cb_addr_shift, 63,
            72 << cb_addr_shift, 9, 8 << cb_addr_shift>>(
        runtimeConfiguration, expectedConfiguration,
        {0x80000001U, 0x80000001U});
  }
}
