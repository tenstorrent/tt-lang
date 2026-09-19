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
  static constexpr bool use16BitSequence =
      sequenceModulus <= (uint64_t{1} << 16);
  static constexpr uint32_t published = 0;
  static constexpr uint32_t consumed = 4;
  uint32_t state;
  uint32_t payload;
  mutable uint32_t acquiredProducerSequence;
  mutable uint32_t acquiredConsumerSequence;

  static uint32_t getPayloadAddress(uint32_t stateAddress) {
    if constexpr (PayloadCommonArgIndex < 0) {
      return stateAddress + PayloadOffset;
    }
    return target::commonArg(PayloadCommonArgIndex) + PayloadOffset;
  }

  static uint32_t occupancy(uint32_t producer, uint32_t consumer) {
    return producer >= consumer ? producer - consumer
                                : sequenceModulus - (consumer - producer);
  }

  static void validatePages(uint32_t pages) {
    ASSERT(pages > 0 && pages % PagesPerBlock == 0);
    ASSERT(pages <= StorageCapacityPages);
  }

  static void assertContiguous(uint32_t sequence, uint32_t pages) {
    ASSERT(sequence % StorageCapacityPages + pages <= StorageCapacityPages);
  }

  static uint32_t advance(uint32_t current, uint32_t pages) {
    uint32_t wrapThreshold = sequenceModulus - pages;
    return current >= wrapThreshold ? current - wrapThreshold : current + pages;
  }

  static uint32_t loadSequence(uint32_t address) {
    if constexpr (use16BitSequence) {
      return target::loadSequence16(address);
    }
    return target::load(address);
  }

  static void publishSequence(uint32_t address, uint32_t sequence) {
    if constexpr (use16BitSequence) {
      target::publishSequence16(address, sequence);
      return;
    }
    target::complete();
    target::store(address, sequence);
  }

  uint32_t address(uint32_t sequence) const {
    return payload + (sequence % StorageCapacityPages) * PageBytes;
  }

public:
  static constexpr uint32_t page_size_bytes = PageBytes;
  static constexpr uint32_t pages_per_block = PagesPerBlock;
  static constexpr uint32_t block_count = BlockCount;
  static constexpr uint32_t storage_capacity_pages = StorageCapacityPages;
  static constexpr uint32_t payload_offset = PayloadOffset;
  explicit Buffer(uint32_t address)
      : state(address), payload(getPayloadAddress(address)),
        acquiredProducerSequence(0), acquiredConsumerSequence(0) {}
  void reserve_back(uint32_t pages) const {
    if constexpr (!target::ownsProducer) {
      return;
    }
    validatePages(pages);
    acquiredProducerSequence = loadSequence(state + published);
    while (StorageCapacityPages - occupancy(acquiredProducerSequence,
                                            loadSequence(state + consumed)) <
           pages) {
    }
    assertContiguous(acquiredProducerSequence, pages);
  }
  void wait_front(uint32_t pages) const {
    if constexpr (!target::ownsConsumer) {
      return;
    }
    validatePages(pages);
    acquiredConsumerSequence = loadSequence(state + consumed);
    while (occupancy(loadSequence(state + published),
                     acquiredConsumerSequence) < pages) {
    }
    assertContiguous(acquiredConsumerSequence, pages);
  }
  void push_back(uint32_t pages) const {
    if constexpr (!target::ownsProducer) {
      return;
    }
    validatePages(pages);
    acquiredProducerSequence = advance(acquiredProducerSequence, pages);
    publishSequence(state + published, acquiredProducerSequence);
  }
  void pop_front(uint32_t pages) const {
    if constexpr (!target::ownsConsumer) {
      return;
    }
    validatePages(pages);
    acquiredConsumerSequence = advance(acquiredConsumerSequence, pages);
    publishSequence(state + consumed, acquiredConsumerSequence);
  }
  uint32_t get_write_ptr() const { return address(acquiredProducerSequence); }
  uint32_t get_read_ptr() const { return address(acquiredConsumerSequence); }
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
