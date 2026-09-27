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
  for (uint32_t index = 0; index < interfaces.size(); ++index) {
    uint32_t base = 0x100 + index * 0x40;
    interfaces[index] = {base + 7,    base + 9, 11,         13,
                         base + 0x20, 0x20,     15 + index, 4};
    received[index] = 101 + index;
    acked[index] = 201 + index;
  }
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
  std::array<uint32_t, 264> configuration = {};
  for (uint32_t index = 0; index < interfaces.size(); ++index) {
    uint32_t base = 0x1000 + index * 0x100;
    uint32_t pages = 5 + index % 3;
    configuration[index * 4] = base << cb_addr_shift;
    configuration[index * 4 + 1] = (pages * 8) << cb_addr_shift;
    configuration[index * 4 + 2] = pages;
    configuration[index * 4 + 3] = 8 << cb_addr_shift;
  }

  for (const auto &mask : masks) {
#if TEST_SHARED_GEOMETRY
    geometryUpdateCount = 0;
    ::experimental::dfb_reconfiguration_detail::rebindSharedGeometry(
        configuration.data(), mask[0], 0);
    ::experimental::dfb_reconfiguration_detail::rebindSharedGeometry(
        configuration.data(), mask[1], 32);
    uint32_t expectedUpdateCount = 0;
    for (uint32_t index = 0; index < interfaces.size(); ++index) {
      bool selected = ((mask[index / 32] >> (index % 32)) & 1U) != 0;
      if (!selected) {
        continue;
      }
      assert(expectedUpdateCount < geometryUpdateCount);
      const auto &update = geometryUpdates.at(expectedUpdateCount++);
      assert(update.index == index);
      assert(update.baseBytes == ((0x1000 + index * 0x100) << cb_addr_shift));
      assert(update.pageBytes == (8 << cb_addr_shift));
      assert(update.pages == 5 + index % 3);
    }
    assert(geometryUpdateCount == expectedUpdateCount);
#endif

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

    initializeInterfaces();
    before = interfaces;
#if TEST_ACTIVE
    ::experimental::dfb_reconfiguration_detail::applyMask<
        TEST_RECONFIG_READ, TEST_RECONFIG_WRITE, TEST_RECONFIG_WRITE_TILE,
        TEST_RECONFIG_RESET_COUNTERS>(configuration.data(), mask[0], 0);
    ::experimental::dfb_reconfiguration_detail::applyMask<
        TEST_RECONFIG_READ, TEST_RECONFIG_WRITE, TEST_RECONFIG_WRITE_TILE,
        TEST_RECONFIG_RESET_COUNTERS>(configuration.data(), mask[1], 32);
#else
    ::experimental::reconfigure_dfb_interfaces(0);
#endif
    for (uint32_t index = 0; index < interfaces.size(); ++index) {
      bool selected = ((mask[index / 32] >> (index % 32)) & 1U) != 0;
      auto expected = before[index];
      if (selected && TEST_ACTIVE) {
        uint32_t base = 0x1000 + index * 0x100;
        uint32_t pages = 5 + index % 3;
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
        expected.fifo_limit = base + pages * 8;
        expected.fifo_size = pages * 8;
        expected.fifo_page_size = 8;
      }
      checkInterface(interfaces[index], expected);
      bool resetCounters = selected && TEST_RESET_COUNTERS;
      assert(received[index] == (resetCounters ? 0 : 101 + index));
      assert(acked[index] == (resetCounters ? 0 : 201 + index));
    }
  }
}
