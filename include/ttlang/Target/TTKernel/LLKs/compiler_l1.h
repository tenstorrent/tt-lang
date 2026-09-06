// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifndef TTLANG_COMPILER_L1_H
#define TTLANG_COMPILER_L1_H
#include <cstdint>
namespace ttlang::l1 {
/// Single-producer/single-consumer storage with ordinary L1 control words.
/// The runtime owns and zero-initializes the complete allocation per
/// invocation.
template <uint32_t PageBytes, uint32_t Capacity, uint32_t PayloadOffset>
class Buffer {
  static_assert(PageBytes > 0 && Capacity > 0 && Capacity < (1u << 31));
  uint32_t state;
  static constexpr uint32_t published = 0;
  static constexpr uint32_t consumed = 4;
  static constexpr uint32_t writePosition = 8;
  static constexpr uint32_t readPosition = 12;

public:
  explicit Buffer(uint32_t address) : state(address) {}
  void reserve_back(uint32_t pages) const {
    ASSERT(pages <= Capacity);
    while (Capacity - (target::load(state + published) -
                       target::load(state + consumed)) <
           pages) {
    }
    ASSERT(target::load(state + writePosition) + pages <= Capacity);
  }
  void wait_front(uint32_t pages) const {
    ASSERT(pages <= Capacity);
    while (target::load(state + published) - target::load(state + consumed) <
           pages) {
    }
    ASSERT(target::load(state + readPosition) + pages <= Capacity);
  }
  void push_back(uint32_t pages) const {
    target::complete();
    target::store(state + writePosition,
                  (target::load(state + writePosition) + pages) % Capacity);
    target::store(state + published, target::load(state + published) + pages);
  }
  void pop_front(uint32_t pages) const {
    target::complete();
    target::store(state + readPosition,
                  (target::load(state + readPosition) + pages) % Capacity);
    target::store(state + consumed, target::load(state + consumed) + pages);
  }
  uint32_t get_write_ptr() const {
    return state + PayloadOffset +
           target::load(state + writePosition) * PageBytes;
  }
  uint32_t get_read_ptr() const {
    return state + PayloadOffset +
           target::load(state + readPosition) * PageBytes;
  }
};
} // namespace ttlang::l1
#endif
