// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifndef TTLANG_COMPILER_L1_H
#define TTLANG_COMPILER_L1_H
#include <cstdint>
#define TTLANG_DFB_STORAGE_COMPILER_L1 1
namespace ttlang::l1 {
/// Clears the producer and consumer sequences for one compiler-managed DFB.
inline void resetState(uint32_t state) {
  if constexpr (!target::ownsDFBInterface) {
    return;
  }
  target::store(state, 0);
  target::store(state + sizeof(uint32_t), 0);
}

/// Single-producer/single-consumer storage with two page sequence counters.
template <uint32_t PageBytes, uint32_t PagesPerBlock, uint32_t BlockCount,
          uint32_t StorageCapacityPages, uint32_t PayloadOffset,
          int32_t PayloadCommonArgIndex>
class Buffer {
  static_assert(PageBytes > 0 && PagesPerBlock > 0 && BlockCount > 0 &&
                uint64_t{PagesPerBlock} * BlockCount <= StorageCapacityPages &&
                StorageCapacityPages < (uint64_t{1} << 31));
  static constexpr uint32_t sequenceModulus = 2 * StorageCapacityPages;
  static constexpr uint32_t published = 0;
  static constexpr uint32_t consumed = 4;
  uint32_t state;
  uint32_t payload;

  static uint32_t getPayloadAddress(uint32_t stateAddress) {
    if constexpr (PayloadCommonArgIndex < 0) {
      return stateAddress + PayloadOffset;
    }
    return target::commonArg(PayloadCommonArgIndex) + PayloadOffset;
  }

  uint32_t occupancy() const {
    uint32_t producer = target::load(state + published);
    uint32_t consumer = target::load(state + consumed);
    return producer >= consumer ? producer - consumer
                                : sequenceModulus - (consumer - producer);
  }

  static void validatePages(uint32_t pages) {
    ASSERT(pages > 0 && pages % PagesPerBlock == 0);
    ASSERT(pages <= StorageCapacityPages);
  }

  void assertContiguous(uint32_t counter, uint32_t pages) const {
    ASSERT(target::load(state + counter) % StorageCapacityPages + pages <=
           StorageCapacityPages);
  }

  void advance(uint32_t counter, uint32_t pages) const {
    uint32_t current = target::load(state + counter);
    uint32_t wrapThreshold = sequenceModulus - pages;
    uint32_t next =
        current >= wrapThreshold ? current - wrapThreshold : current + pages;
    target::store(state + counter, next);
  }

  uint32_t address(uint32_t counter) const {
    return payload +
           (target::load(state + counter) % StorageCapacityPages) * PageBytes;
  }

public:
  static constexpr uint32_t page_size_bytes = PageBytes;
  static constexpr uint32_t pages_per_block = PagesPerBlock;
  static constexpr uint32_t block_count = BlockCount;
  static constexpr uint32_t storage_capacity_pages = StorageCapacityPages;
  static constexpr uint32_t payload_offset = PayloadOffset;
  explicit Buffer(uint32_t address)
      : state(address), payload(getPayloadAddress(address)) {}
  void reserve_back(uint32_t pages) const {
    if constexpr (!target::ownsProducer) {
      return;
    }
    validatePages(pages);
    while (StorageCapacityPages - occupancy() < pages) {
    }
    assertContiguous(published, pages);
  }
  void wait_front(uint32_t pages) const {
    if constexpr (!target::ownsConsumer) {
      return;
    }
    validatePages(pages);
    while (occupancy() < pages) {
    }
    assertContiguous(consumed, pages);
  }
  void push_back(uint32_t pages) const {
    if constexpr (!target::ownsProducer) {
      return;
    }
    validatePages(pages);
    target::complete();
    advance(published, pages);
  }
  void pop_front(uint32_t pages) const {
    if constexpr (!target::ownsConsumer) {
      return;
    }
    validatePages(pages);
    target::complete();
    advance(consumed, pages);
  }
  uint32_t get_write_ptr() const { return address(published); }
  uint32_t get_read_ptr() const { return address(consumed); }
};

template <uint32_t PageBytes, uint32_t PagesPerBlock, uint32_t BlockCount,
          uint32_t StorageCapacityPages, uint32_t StateOffset,
          uint32_t PayloadOffset, int32_t PayloadCommonArgIndex>
class DFBDescriptor
    : public Buffer<PageBytes, PagesPerBlock, BlockCount, StorageCapacityPages,
                    PayloadOffset, PayloadCommonArgIndex> {
public:
  using Buffer<PageBytes, PagesPerBlock, BlockCount, StorageCapacityPages,
               PayloadOffset, PayloadCommonArgIndex>::Buffer;
  /// Binds this descriptor to its compile-time allocation in the core arena.
  static DFBDescriptor bind() {
    return DFBDescriptor(target::arenaBase() + StateOffset);
  }
};
} // namespace ttlang::l1
#endif
