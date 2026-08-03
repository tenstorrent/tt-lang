// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// External compute shim that exercises int/bool/float template args and
// int/bool/float/DFB (CB index) function args via ttl.call_extern_func.
//
// Template args (compile-time):
//   OutCB       -- output CB index
//   IntScale    -- integer scale factor
//   NegateTpl   -- bool: negate when true
//   ScaleTplBits-- IEEE-754 f32 bit pattern for a float scale
//
// Func args (runtime):
//   in_cb       -- input CB index (from a DFB passed as func_args)
//   out_cb      -- output CB index, declaring the output DFB dependency
//   scale_f     -- float scale
//   int_factor   -- integer factor multiplied into the scale
//   also_negate -- bool: also request negate
//
// Result: out = (+/-) in * IntScale * scale_tpl * scale_f * int_factor
// where the sign is negative when NegateTpl || also_negate.

#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace {

inline float float_from_bits(uint32_t bits) {
  float value;
  static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
  __builtin_memcpy(&value, &bits, sizeof(value));
  return value;
}

inline uint32_t bits_from_float(float value) {
  uint32_t bits;
  static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
  __builtin_memcpy(&bits, &value, sizeof(bits));
  return bits;
}

} // namespace

template <uint32_t OutCB, int IntScale, bool NegateTpl, uint32_t ScaleTplBits>
void typed_args_shim(uint32_t in_cb, uint32_t out_cb, float scale_f,
                     int32_t int_factor, bool also_negate) {
  (void)out_cb;
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;

  const float scale_tpl = float_from_bits(ScaleTplBits);
  const float combined_scale = static_cast<float>(IntScale) * scale_tpl *
                               scale_f * static_cast<float>(int_factor);
  const bool do_negate = NegateTpl || also_negate;

  unary_op_init_common(in_cb, OutCB);
  copy_tile_init(in_cb);
  binop_with_scalar_tile_init();

  tile_regs_acquire();
  copy_tile(in_cb, 0, 0);

  if (do_negate) {
    negative_tile_init();
    negative_tile(0);
  }

  mul_unary_tile(0, bits_from_float(combined_scale));

  tile_regs_commit();
  tile_regs_wait();
  pack_tile(0, OutCB);
  tile_regs_release();
#else
  (void)in_cb;
  (void)scale_f;
  (void)int_factor;
  (void)also_negate;
#endif
}
