// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifndef TTLANG_COMPILER_L1_H
#define TTLANG_COMPILER_L1_H
#include <cstdint>
namespace ttlang::l1 {
/// Single-producer/single-consumer storage with two block sequence counters.
template <uint32_t PageBytes, uint32_t PagesPerBlock, uint32_t BlockCount,
          uint32_t PayloadOffset>
class Buffer {
  static_assert(PageBytes > 0 && PagesPerBlock > 0 && BlockCount > 0 &&
                uint64_t{PagesPerBlock} * BlockCount < (uint64_t{1} << 31));
  static constexpr uint32_t sequenceModulus = 2 * BlockCount;
  static constexpr uint32_t published = 0;
  static constexpr uint32_t consumed = 4;
  uint32_t state;

  uint32_t occupancy() const {
    uint32_t producer = target::load(state + published);
    uint32_t consumer = target::load(state + consumed);
    return producer >= consumer ? producer - consumer
                                : sequenceModulus - (consumer - producer);
  }

  void advance(uint32_t counter) const {
    uint32_t next = target::load(state + counter) + 1;
    target::store(state + counter, next == sequenceModulus ? 0 : next);
  }

  uint32_t address(uint32_t counter) const {
    return state + PayloadOffset +
           (target::load(state + counter) % BlockCount) * PagesPerBlock *
               PageBytes;
  }

public:
  explicit Buffer(uint32_t address) : state(address) {}
  void reserve_back(uint32_t pages) const {
    if constexpr (!target::ownsProducer) {
      return;
    }
    ASSERT(pages == PagesPerBlock);
    while (occupancy() == BlockCount) {
    }
  }
  void wait_front(uint32_t pages) const {
    if constexpr (!target::ownsConsumer) {
      return;
    }
    ASSERT(pages == PagesPerBlock);
    while (occupancy() == 0) {
    }
  }
  void push_back(uint32_t pages) const {
    if constexpr (!target::ownsProducer) {
      return;
    }
    ASSERT(pages == PagesPerBlock);
    target::complete();
    advance(published);
  }
  void pop_front(uint32_t pages) const {
    if constexpr (!target::ownsConsumer) {
      return;
    }
    ASSERT(pages == PagesPerBlock);
    target::complete();
    advance(consumed);
  }
  uint32_t get_write_ptr() const { return address(published); }
  uint32_t get_read_ptr() const { return address(consumed); }
};
} // namespace ttlang::l1
#endif
