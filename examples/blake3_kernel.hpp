// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/add_int_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/bitwise.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/shift.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif

template <uint32_t Cb>
inline int32_t ttlang_blake3_peek_write_u32(int dfb) {
  (void)dfb;
#if defined(COMPILE_FOR_TRISC)
  return 0;
#else
  return static_cast<int32_t>(
      *reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(Cb)));
#endif
}

template <uint32_t Cb>
inline int32_t ttlang_blake3_peek_read_u32(int dfb) {
  (void)dfb;
#if defined(COMPILE_FOR_TRISC)
  return static_cast<int32_t>(ckernel::read_tile_value(Cb, 0, 0));
#else
  return static_cast<int32_t>(
      *reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_read_ptr(Cb)));
#endif
}

namespace ttlang_blake3 {

constexpr uint32_t WordCount = 16;
constexpr uint32_t CvWords = 8;
constexpr uint32_t CounterTile = 0;
constexpr uint32_t BlockLenTile = 1;
constexpr uint32_t FlagsTile = 2;
constexpr uint32_t ActiveTile = 3;

constexpr uint32_t Iv[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

constexpr uint32_t MessagePermutation[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8,
};

#if defined(COMPILE_FOR_TRISC)
using namespace ckernel;

constexpr DataFormat Fmt = DataFormat::Int32;

// G-state a,b,c,d occupy DST 0-3 for the whole mix. Message words and rotate
// scratch use DST 4-7 so Int32 dest-acc (8 tiles) is fully used.
constexpr uint32_t DstA = 0;
constexpr uint32_t DstB = 1;
constexpr uint32_t DstC = 2;
constexpr uint32_t DstD = 3;
constexpr uint32_t DstMx = 4;
constexpr uint32_t DstMy = 5;
constexpr uint32_t DstRorLeft = 6;
constexpr uint32_t DstRorAmt = 7;

inline void loadTile(uint32_t cb, uint32_t tile, uint32_t dst) {
  copy_tile_init(cb);
  copy_tile(cb, tile, dst);
}

inline void addDst(uint32_t a, uint32_t b, uint32_t o) {
  add_int_tile_init();
  add_int_tile<Fmt>(a, b, o);
}

inline void xorDst(uint32_t a, uint32_t b, uint32_t o) {
  binary_bitwise_tile_init();
  bitwise_xor_binary_tile<Fmt>(a, b, o);
}

inline void andDst(uint32_t a, uint32_t b, uint32_t o) {
  binary_bitwise_tile_init();
  bitwise_and_binary_tile<Fmt>(a, b, o);
}

inline void orDst(uint32_t a, uint32_t b, uint32_t o) {
  binary_bitwise_tile_init();
  bitwise_or_binary_tile<Fmt>(a, b, o);
}

inline void rorDst(uint32_t valueDst, uint32_t amount) {
  fill_tile_init();
  fill_tile_int<Fmt>(DstRorLeft, 0);
  addDst(valueDst, DstRorLeft, DstRorLeft);
  left_shift_tile_init();
  left_shift_tile<Fmt>(DstRorLeft, 32 - amount);
  fill_tile_int<Fmt>(DstRorAmt, amount);
  binary_shift_tile_init();
  binary_logical_right_shift_tile<Fmt>(valueDst, DstRorAmt, valueDst);
  orDst(valueDst, DstRorLeft, valueDst);
}

template <typename Scratch>
inline void storeDestState(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  cb_reserve_back(Scratch::index, WordCount);
  pack_tile<true>(DstA, Scratch::index, a);
  pack_tile<true>(DstB, Scratch::index, b);
  pack_tile<true>(DstC, Scratch::index, c);
  pack_tile<true>(DstD, Scratch::index, d);
  tile_regs_release();
  for (uint32_t i = 0; i < WordCount; ++i) {
    if (i == a || i == b || i == c || i == d) {
      continue;
    }
    tile_regs_acquire();
    loadTile(Scratch::index, i, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, Scratch::index, i);
    tile_regs_release();
  }
  cb_pop_front(Scratch::index, WordCount);
  cb_push_back(Scratch::index, WordCount);
  cb_wait_front(Scratch::index, WordCount);
}

template <typename Scratch>
inline void mixMsg(uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                   uint32_t msgCb, uint32_t x, uint32_t y) {
  tile_regs_acquire();
  loadTile(Scratch::index, a, DstA);
  loadTile(Scratch::index, b, DstB);
  loadTile(Scratch::index, c, DstC);
  loadTile(Scratch::index, d, DstD);
  loadTile(msgCb, x, DstMx);
  loadTile(msgCb, y, DstMy);

  addDst(DstA, DstB, DstA);
  addDst(DstA, DstMx, DstA);
  xorDst(DstD, DstA, DstD);
  rorDst(DstD, 16);
  addDst(DstC, DstD, DstC);
  xorDst(DstB, DstC, DstB);
  rorDst(DstB, 12);
  addDst(DstA, DstB, DstA);
  addDst(DstA, DstMy, DstA);
  xorDst(DstD, DstA, DstD);
  rorDst(DstD, 8);
  addDst(DstC, DstD, DstC);
  xorDst(DstB, DstC, DstB);
  rorDst(DstB, 7);

  tile_regs_commit();
  tile_regs_wait();
  storeDestState<Scratch>(a, b, c, d);
}

inline void permute(uint32_t message[16]) {
  uint32_t permuted[16];
  for (uint32_t i = 0; i < 16; ++i) {
    permuted[i] = message[MessagePermutation[i]];
  }
  for (uint32_t i = 0; i < 16; ++i) {
    message[i] = permuted[i];
  }
}

template <uint32_t MsgCb, uint32_t CvCb, uint32_t MetaCb, uint32_t OutCb,
          typename Scratch>
inline void runCompress() {
  unary_op_init_common(MsgCb, OutCb);
  add_int_tile_init();
  binary_bitwise_tile_init();
  left_shift_tile_init();
  fill_tile_init();
  binary_shift_tile_init();

  cb_reserve_back(Scratch::index, WordCount);
  for (uint32_t i = 0; i < WordCount; ++i) {
    tile_regs_acquire();
    if (i < CvWords) {
      loadTile(CvCb, i, 0);
    } else if (i < 12) {
      fill_tile_init();
      fill_tile_int<Fmt>(0, Iv[i - CvWords]);
    } else if (i == 12) {
      loadTile(MetaCb, CounterTile, 0);
    } else if (i == 13) {
      fill_tile_init();
      fill_tile_int<Fmt>(0, 0);
    } else if (i == 14) {
      loadTile(MetaCb, BlockLenTile, 0);
    } else {
      loadTile(MetaCb, FlagsTile, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, Scratch::index, i);
    tile_regs_release();
  }
  cb_push_back(Scratch::index, WordCount);
  cb_wait_front(Scratch::index, WordCount);

  uint32_t message[16];
  for (uint32_t i = 0; i < 16; ++i) {
    message[i] = i;
  }

  for (uint32_t roundIndex = 0; roundIndex < 7; ++roundIndex) {
    mixMsg<Scratch>(0, 4, 8, 12, MsgCb, message[0], message[1]);
    mixMsg<Scratch>(1, 5, 9, 13, MsgCb, message[2], message[3]);
    mixMsg<Scratch>(2, 6, 10, 14, MsgCb, message[4], message[5]);
    mixMsg<Scratch>(3, 7, 11, 15, MsgCb, message[6], message[7]);
    mixMsg<Scratch>(0, 5, 10, 15, MsgCb, message[8], message[9]);
    mixMsg<Scratch>(1, 6, 11, 12, MsgCb, message[10], message[11]);
    mixMsg<Scratch>(2, 7, 8, 13, MsgCb, message[12], message[13]);
    mixMsg<Scratch>(3, 4, 9, 14, MsgCb, message[14], message[15]);
    if (roundIndex != 6) {
      permute(message);
    }
  }

  for (uint32_t i = 0; i < CvWords; ++i) {
    tile_regs_acquire();
    loadTile(Scratch::index, i, 0);
    loadTile(Scratch::index, i + CvWords, 1);
    xorDst(0, 1, 0);
    loadTile(MetaCb, ActiveTile, 1);
    andDst(0, 1, 0);
    loadTile(CvCb, i, 2);
    loadTile(MetaCb, ActiveTile, 3);
    bitwise_xor_tile_init();
    bitwise_xor_tile<Fmt>(3, 0xFFFFFFFFu);
    andDst(2, 3, 2);
    orDst(0, 2, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, OutCb, i);
    tile_regs_release();
  }
  cb_pop_front(Scratch::index, WordCount);
}

#endif // COMPILE_FOR_TRISC

} // namespace ttlang_blake3

#if defined(COMPILE_FOR_TRISC)
template <uint32_t MsgCb, uint32_t CvCb, uint32_t MetaCb, uint32_t OutCb,
          typename Scratch>
inline void ttlang_blake3_compress_tiles(int msgDfb, int cvDfb, int metaDfb,
                                         int outDfb, int scratchDfb) {
  (void)msgDfb;
  (void)cvDfb;
  (void)metaDfb;
  (void)outDfb;
  (void)scratchDfb;
  ttlang_blake3::runCompress<MsgCb, CvCb, MetaCb, OutCb, Scratch>();
}
#endif
