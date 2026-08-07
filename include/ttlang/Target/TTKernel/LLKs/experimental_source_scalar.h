// SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SOURCE_SCALAR_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SOURCE_SCALAR_H

#include <cstdint>

#include "api/compute/eltwise_binary.h"

#ifdef TRISC_MATH
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_rsqrt.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

#ifdef TRISC_UNPACK
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common_api.h"
#endif

namespace experimental {
namespace source_scalar_detail {

#ifdef TRISC_MATH

using ckernel::ADDR_MOD_0;
using ckernel::ADDR_MOD_1;
using ckernel::ADDR_MOD_2;
using ckernel::ADDR_MOD_3;
using ckernel::ADDR_MOD_4;
using ckernel::addr_mod_t;
using ckernel::ckernel_template;
using ckernel::dest_offset_id;
using ckernel::DstSync;
using ckernel::DstTileShape;
using ckernel::MathFidelity;
using ckernel::to_underlying;
using ckernel::UnpackDestination;
using ckernel::VectorMode;
using ckernel::math::is_high_fidelity;

template <bool approximationMode, int iterations, bool fp32DestAccEnabled,
          bool fastApproximation>
inline void calculateAddRsqrt(std::uint32_t addendBits) {
#pragma GCC unroll 8
  for (int iteration = 0; iteration < iterations; ++iteration) {
    sfpi::vFloat input = sfpi::dst_reg[0];
    sfpi::vFloat biased =
        input + ckernel::sfpu::Converter::as_float(addendBits);
    sfpi::vFloat inverseSquareRoot =
        ckernel::sfpu::_calculate_sqrt_body_<approximationMode, true,
                                             fastApproximation>(biased);
    if constexpr (fp32DestAccEnabled) {
      sfpi::dst_reg[0] = inverseSquareRoot;
    } else {
      sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(
          inverseSquareRoot, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg++;
  }
}

template <bool approximationMode>
inline void initializeAddRsqrt() {
  ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt>(
      ckernel::sfpu::sqrt_init<approximationMode>);
}

template <bool approximationMode, bool fp32DestAccEnabled,
          bool fastApproximation, int iterations>
inline void applyAddRsqrt(std::uint32_t dstIndex, std::uint32_t addendBits) {
  _llk_math_eltwise_unary_sfpu_params_(
      calculateAddRsqrt<approximationMode, iterations, fp32DestAccEnabled,
                        fastApproximation>,
      dstIndex, VectorMode::RC_custom, addendBits);
}

template <std::uint32_t numTiles, MathFidelity mathFidelity>
inline void configureMathMop(std::uint32_t numFaces) {
  static_assert(numTiles >= 1 && numTiles <= 8,
                "source-scalar multiplication requires 1 to 8 tiles");
  LLK_ASSERT(numFaces == 1 || numFaces == 2 || numFaces == 4,
             "numFaces must be 1, 2, or 4");
  constexpr bool highFidelity = is_high_fidelity(mathFidelity);
  constexpr auto broadcastType = ckernel::p_elwise::SRCB_BCAST_ALL;

  if constexpr (highFidelity) {
    ckernel_template operationTemplate(
        numFaces, to_underlying(mathFidelity),
        TT_OP_ELWMUL(0, 0, broadcastType, ADDR_MOD_0, 0),
        TT_OP_ELWMUL(0, 0, broadcastType, ADDR_MOD_2, 0));
    operationTemplate.set_last_inner_loop_instr(TT_OP_ELWMUL(
        ckernel::p_setrwc::CLR_A, 0, broadcastType, ADDR_MOD_3, 0));
    operationTemplate.set_last_outer_loop_instr(TT_OP_ELWMUL(
        ckernel::p_setrwc::CLR_A, 0, broadcastType, ADDR_MOD_4, 0));
    operationTemplate.program();
  } else {
    ckernel_template operationTemplate(
        numTiles, numFaces, TT_OP_ELWMUL(0, 0, broadcastType, ADDR_MOD_0, 0),
        TT_OP_ELWMUL(ckernel::p_setrwc::CLR_A, 0, broadcastType, ADDR_MOD_2,
                     0));
    operationTemplate.set_last_inner_loop_instr(TT_OP_ELWMUL(
        ckernel::p_setrwc::CLR_A, 0, broadcastType, ADDR_MOD_3, 0));
    operationTemplate.set_last_outer_loop_instr(TT_OP_ELWMUL(
        ckernel::p_setrwc::CLR_A, 0, broadcastType, ADDR_MOD_3, 0));
    operationTemplate.program();
  }
}

inline void reuseScalarAsSource() {
  TTI_STALLWAIT(ckernel::p_stall::STALL_MATH,
                ckernel::p_stall::WAIT_SFPU | ckernel::p_stall::SRCB_VLD);
  TTI_MOVD2B(0, ckernel::p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_1,
             ckernel::p_movd2b::MOV_1_ROW, 0);
}

template <DstSync dstSync, bool fp32DestAccEnabled>
inline void acquireScalar(std::uint32_t scalarDstIndex,
                          std::uint32_t outputDstIndex) {
  ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32,
                                    UnpackDestination::SrcRegs>(scalarDstIndex);
  reuseScalarAsSource();
  if constexpr (dstSync == DstSync::SyncFull) {
    TT_ZEROACC(ckernel::p_zeroacc::CLR_ALL, fp32DestAccEnabled, 0, ADDR_MOD_1,
               0);
  } else {
    static_assert(dstSync == DstSync::SyncHalf);
    TT_ZEROACC(ckernel::p_zeroacc::CLR_HALF, fp32DestAccEnabled, 0, ADDR_MOD_1,
               dest_offset_id);
  }
  ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32,
                                    UnpackDestination::SrcRegs>(outputDstIndex);
}

template <std::uint32_t numTiles, MathFidelity mathFidelity>
inline void applyScalar() {
  constexpr bool highFidelity = is_high_fidelity(mathFidelity);

  if constexpr (highFidelity) {
    for (std::uint32_t tileIndex = 0; tileIndex < numTiles; ++tileIndex) {
      ckernel_template::run();
    }
  } else {
    ckernel_template::run();
  }
}

inline void releaseScalar() {
  TTI_SETRWC(ckernel::p_setrwc::CLR_B, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
}

template <MathFidelity mathFidelity>
inline void configureMathAddressModifiers(std::uint32_t numFaces) {
  constexpr bool highFidelity = is_high_fidelity(mathFidelity);
  constexpr std::uint32_t fidelityIncrement = highFidelity ? 1 : 0;

  addr_mod_t{
      .srca = {.incr = 8},
      .srcb = {.incr = 0},
      .dest = {.incr = 8},
  }
      .set(ADDR_MOD_0);
  addr_mod_t{
      .srca = {.incr = 0},
      .srcb = {.incr = 0},
      .dest = {.incr = 0},
  }
      .set(ADDR_MOD_1);

  if constexpr (highFidelity) {
    addr_mod_t{.srca = {.incr = 0, .clr = 1},
               .srcb = {.incr = 0, .clr = 1},
               .dest = {.incr = 0, .clr = 0, .cr = 1},
               .fidelity = {.incr = fidelityIncrement}}
        .set(ADDR_MOD_2);
    addr_mod_t{.srca = {.incr = 0, .clr = 1},
               .srcb = {.incr = 0, .clr = 1},
               .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1},
               .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_3);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1},
        .srcb = {.incr = 0, .clr = 1},
        .dest = {.incr = static_cast<std::int16_t>(8 + (4 - numFaces) * 16),
                 .clr = 0,
                 .cr = 0,
                 .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_4);
  } else {
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1},
        .srcb = {.incr = 0, .clr = 1},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1},
        .srcb = {.incr = 0, .clr = 1},
        .dest = {.incr = static_cast<std::int16_t>(8 + (4 - numFaces) * 16)},
    }
        .set(ADDR_MOD_3);
  }
}

template <std::uint32_t numTiles, MathFidelity mathFidelity>
inline void initializeMath(std::uint32_t numFaces) {
  LLK_ASSERT(numFaces == 1 || numFaces == 2 || numFaces == 4,
             "numFaces must be 1, 2, or 4");
  configureMathAddressModifiers<mathFidelity>(numFaces);
  configureMathMop<numTiles, mathFidelity>(numFaces);
  TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
  ckernel::math::reset_counters(ckernel::p_setrwc::SET_ABD_F);
}

template <std::uint32_t numTiles, MathFidelity mathFidelity>
inline void initializeMathForOperand(std::uint32_t operand) {
  const std::uint32_t operandId = get_operand_id(operand);
  initializeMath<numTiles, mathFidelity>(get_operand_num_faces(operandId));
}

#endif // TRISC_MATH

#ifdef TRISC_UNPACK

using ckernel::cfg_reg_rmw_tensix;
using ckernel::ckernel_template;
using ckernel::SrcA;
using ckernel::SrcB;
using ckernel::unpacker::config_unpacker_x_end;

template <std::uint32_t numTiles>
inline void configureUnpackMop(std::uint32_t numFaces) {
  LLK_ASSERT(numFaces == 1 || numFaces == 2 || numFaces == 4,
             "numFaces must be 1, 2, or 4");

  static constexpr std::uint32_t unpackSourceA =
      TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, ckernel::p_unpacr::RAREFYB_DISABLE,
                   0, 0, 0, 0, 1);
  static constexpr std::uint32_t setSourceBValid =
      TT_OP_UNPACR_NOP(SrcB, 0, 0, ckernel::p_unpacr_nop::SET_DVALID, 0, 0, 0,
                       0, ckernel::p_unpacr_nop::UNP_ZEROSRC);

  if (numFaces == 1) {
    ckernel_template unpackTemplate(1, numTiles, unpackSourceA);
    unpackTemplate.set_start_op(setSourceBValid);
    unpackTemplate.program();
    return;
  }

  const std::uint32_t iterations = numTiles * numFaces / 2;
  ckernel_template unpackTemplate(1, iterations, unpackSourceA, unpackSourceA);
  unpackTemplate.set_start_op(setSourceBValid);
  unpackTemplate.program();
}

template <std::uint32_t numTiles>
inline void initializeUnpack(std::uint32_t faceRowDimension,
                             std::uint32_t numFaces) {
  cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
  config_unpacker_x_end<ckernel::p_setadc::UNP_B>(faceRowDimension);
  configureUnpackMop<numTiles>(numFaces);
}

template <std::uint32_t numTiles>
inline void initializeUnpackForOperand(std::uint32_t operand) {
  const std::uint32_t operandId = get_operand_id(operand);
  initializeUnpack<numTiles>(get_operand_face_r_dim(operandId),
                             get_operand_num_faces(operandId));
}

#endif // TRISC_UNPACK

template <std::uint32_t numTiles>
ALWI void initializeScalarMultiply(std::uint32_t inputDfb) {
  static_assert(numTiles >= 1 && numTiles <= 8,
                "source-scalar multiplication requires 1 to 8 tiles");
  UNPACK((initializeUnpackForOperand<numTiles>(inputDfb)));
  MATH((initializeMathForOperand<numTiles, MATH_FIDELITY>(inputDfb)));
}

inline void acquireSourceScalar(std::uint32_t scalarDstIndex,
                                std::uint32_t outputDstIndex) {
  MATH((acquireScalar<DST_SYNC_MODE, DST_ACCUM_MODE>(scalarDstIndex,
                                                     outputDstIndex)));
}

template <std::uint32_t numTiles>
ALWI void multiplyBySourceScalar(std::uint32_t inputDfb) {
  UNPACK((llk_unpack_A<ckernel::BroadcastType::SCALAR, true,
                       ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
      inputDfb, 0)));
  MATH((applyScalar<numTiles, MATH_FIDELITY>()));
}

inline void releaseSourceScalar() { MATH((releaseScalar())); }

} // namespace source_scalar_detail

ALWI void add_rsqrt_init() {
  MATH((source_scalar_detail::initializeAddRsqrt<APPROX>()));
}

ALWI void add_rsqrt(std::uint32_t dstIndex, std::uint32_t addendBits) {
  MATH((source_scalar_detail::applyAddRsqrt<APPROX, DST_ACCUM_MODE, false, 1>(
      dstIndex, addendBits)));
}

template <std::uint32_t numTiles>
ALWI void source_scalar_mul_init(std::uint32_t inputDfb) {
  static_assert(numTiles >= 1 && numTiles <= 8,
                "source-scalar multiplication requires 1 to 8 tiles");
  source_scalar_detail::initializeScalarMultiply<numTiles>(inputDfb);
}

ALWI void source_scalar_acquire(std::uint32_t scalarDstIndex,
                                std::uint32_t outputDstIndex) {
  source_scalar_detail::acquireSourceScalar(scalarDstIndex, outputDstIndex);
}

template <std::uint32_t numTiles>
ALWI void source_scalar_mul(std::uint32_t inputDfb) {
  source_scalar_detail::multiplyBySourceScalar<numTiles>(inputDfb);
}

ALWI void source_scalar_release() {
  source_scalar_detail::releaseSourceScalar();
}

template <std::uint32_t numTiles>
ALWI void source_scalar_mul(std::uint32_t inputDfb,
                            std::uint32_t scalarDstIndex,
                            std::uint32_t outputDstIndex) {
  source_scalar_acquire(scalarDstIndex, outputDstIndex);
  source_scalar_mul<numTiles>(inputDfb);
  source_scalar_release();
}

} // namespace experimental

#endif // TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SOURCE_SCALAR_H
