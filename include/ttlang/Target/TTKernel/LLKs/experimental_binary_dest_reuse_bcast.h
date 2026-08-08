// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_BINARY_DEST_REUSE_BCAST_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_BINARY_DEST_REUSE_BCAST_H

#include <cstdint>

#include "api/compute/eltwise_binary.h"

namespace experimental {
namespace binary_dest_reuse_bcast_detail {

#ifdef TRISC_UNPACK

inline void initializeColumnBroadcastUnpack(std::uint32_t inputDfb) {
  const std::uint32_t operandId = get_operand_id(inputDfb);
  const ckernel::TensorShape tensorShape = get_operand_tensor_shape(operandId);
  const std::uint32_t numFaceRows = tensorShape.num_faces_r_dim;
  const std::uint32_t numFaceColumns = tensorShape.num_faces_c_dim;
  LLK_ASSERT(ckernel::validate_tensor_shape_tile_dependent_ops_(tensorShape),
             "invalid column-broadcast tensor shape");
  LLK_ASSERT(numFaceColumns >= numFaceRows,
             "column broadcast requires a square or wide tile");

  const std::uint32_t inputFormat = unpack_src_format[operandId];
  const std::uint32_t unpackedFormat = unpack_dst_format[operandId];
  llk::san::unpack_operand_check(
      llk::san::IGNORE, llk::san::IGNORE, inputFormat, llk::san::IGNORE,
      unpackedFormat, llk::san::IGNORE, tensorShape.face_r_dim,
      llk::san::IGNORE, tensorShape.total_num_faces());
  llk::san::operation_init<llk::san::Operation::UnpackA>(
      ckernel::BroadcastType::COL, true,
      ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA, false, inputFormat,
      unpackedFormat);

  ckernel::cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
  ckernel::unpacker::config_unpacker_x_end<ckernel::p_setadc::UNP_B>(
      tensorShape.face_r_dim);

  static constexpr std::uint32_t unpackSourceB =
      TT_OP_UNPACR(ckernel::SrcB, 0b1, 0, 0, 0, 1, 1,
                   ckernel::p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
  static constexpr std::uint32_t setSourceAValid =
      TT_OP_UNPACR_NOP(ckernel::SrcA, 0, 0, ckernel::p_unpacr_nop::SET_DVALID,
                       0, 0, 0, 0, ckernel::p_unpacr_nop::UNP_ZEROSRC);

  ckernel::ckernel_template unpackTemplate(numFaceRows, numFaceColumns,
                                           setSourceAValid);
  unpackTemplate.set_start_op(unpackSourceB);
  if (numFaceColumns == 1) {
    unpackTemplate.set_end_op(
        TT_OP_SETADCZW(ckernel::p_setadc::UNP_B, 0, 0, 0, 1, 0b0001));
  } else {
    unpackTemplate.set_end_op(
        TT_OP_SETADCZW(ckernel::p_setadc::UNP_B, 0, 0, 0, 2, 0b0001));
  }
  unpackTemplate.program();
}

#endif // TRISC_UNPACK

} // namespace binary_dest_reuse_bcast_detail

template <ckernel::EltwiseBinaryType eltwiseType,
          ckernel::BroadcastType broadcastType,
          ckernel::EltwiseBinaryReuseDestType reuseType>
ALWI void
binary_dest_reuse_bcast_tiles_init(std::uint32_t inputDfb,
                                   std::uint32_t callLine = __builtin_LINE()) {
  static_assert(eltwiseType == ckernel::EltwiseBinaryType::ELWMUL,
                "only multiplication is supported");
  static_assert(broadcastType == ckernel::BroadcastType::COL,
                "only column broadcast is supported");
  static_assert(reuseType == ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                "broadcast dataflow-buffer operand must occupy source B");
  state_configure<ckernel::Operand::SRCB>(inputDfb, callLine);
  UNPACK((binary_dest_reuse_bcast_detail::initializeColumnBroadcastUnpack(
      inputDfb)));
  MATH((llk_math_eltwise_binary_init<eltwiseType, broadcastType, MATH_FIDELITY,
                                     reuseType>(inputDfb, inputDfb, false)));
}

template <ckernel::EltwiseBinaryType eltwiseType,
          ckernel::BroadcastType broadcastType,
          ckernel::EltwiseBinaryReuseDestType reuseType>
ALWI void binary_dest_reuse_bcast_tiles(std::uint32_t inputDfb,
                                        std::uint32_t inputTileIndex,
                                        std::uint32_t dstTileIndex) {
  static_assert(eltwiseType == ckernel::EltwiseBinaryType::ELWMUL,
                "only multiplication is supported");
  static_assert(broadcastType == ckernel::BroadcastType::COL,
                "only column broadcast is supported");
  static_assert(reuseType == ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                "broadcast dataflow-buffer operand must occupy source B");
#ifndef ARCH_QUASAR
  UNPACK(constexpr bool accToDest = true);
#else
  UNPACK(constexpr bool accToDest = false);
#endif
  UNPACK((llk_unpack_A<broadcastType, accToDest, reuseType>(inputDfb,
                                                            inputTileIndex)));
  MATH((llk_math_eltwise_binary<eltwiseType, broadcastType, DST_ACCUM_MODE,
                                MATH_FIDELITY, reuseType>(inputDfb, inputDfb,
                                                          dstTileIndex, true)));
}

} // namespace experimental

#endif // TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_BINARY_DEST_REUSE_BCAST_H
