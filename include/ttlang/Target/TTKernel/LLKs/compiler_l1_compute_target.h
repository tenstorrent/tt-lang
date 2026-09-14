// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifndef TTLANG_COMPILER_L1_COMPUTE_TARGET_H
#define TTLANG_COMPILER_L1_COMPUTE_TARGET_H
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "tensor_shape.h"
#include <cstdint>
namespace ttlang::l1::target {
inline constexpr uint32_t llkAddressWordBytes = MEMORY_WORD_SIZE_IN_BYTES;

template <uint32_t TileHeight, uint32_t TileWidth>
inline constexpr ckernel::TensorShape makeTensorShape() {
  return ckernel::make_tensor_shape(
      TileHeight < ckernel::MAX_FACE_R_DIM ? static_cast<uint8_t>(TileHeight)
                                           : ckernel::MAX_FACE_R_DIM,
      ckernel::MAX_FACE_C_DIM,
      TileHeight > ckernel::MAX_FACE_R_DIM ? ckernel::MAX_NUM_FACES_R_DIM : 1,
      TileWidth > ckernel::MAX_FACE_C_DIM ? ckernel::MAX_NUM_FACES_C_DIM : 1);
}

inline constexpr bool hasPartialFace(ckernel::TensorShape tensorShape) {
  return tensorShape.face_r_dim < ckernel::MAX_FACE_R_DIM;
}

inline constexpr bool isNarrowTile(ckernel::TensorShape tensorShape) {
  return tensorShape.total_col_dim() == ckernel::MAX_FACE_C_DIM;
}
/// Encodes a byte address in the preincremented address-word form used by LLKs.
inline uint32_t toLlkTileAddress(uint32_t byteAddress, uint32_t tile,
                                 uint32_t pageWords) {
  return byteAddress / llkAddressWordBytes + tile * pageWords - 1;
}

inline void resetMatmulThrottleState() {
#if defined(ARCH_BLACKHOLE)
  MATH((ckernel::throttled_mop_status = 0));
#endif
}

template <ckernel::DataCopyType CopyType, ckernel::BroadcastType Broadcast,
          typename Source>
inline void initializeUnaryDataCopy() {
#if defined(ARCH_BLACKHOLE)
  MATH((_llk_math_eltwise_unary_datacopy_init_<CopyType, DST_ACCUM_MODE,
                                               Broadcast>(
      Source::tensorShape.total_num_faces(), Source::unpackFormat, false)));
#else
  MATH((_llk_math_eltwise_unary_datacopy_init_<CopyType, DST_ACCUM_MODE,
                                               Broadcast>(
      Source::tensorShape.total_num_faces(), Source::unpackFormat)));
#endif
}

template <typename Output>
inline void initializePack() {
#if defined(ARCH_BLACKHOLE)
  PACK((_llk_pack_hw_configure_<DST_ACCUM_MODE, ckernel::PackMode::Default>(
      Output::format, Output::format, Output::pageWords,
      Output::tensorShape.face_r_dim, Output::tensorShape.total_col_dim(),
      Output::tensorShape.total_num_faces(),
      hasPartialFace(Output::tensorShape), 0)));
  PACK((_llk_pack_init_<ckernel::PackMode::Default>(
      Output::format, Output::tensorShape.face_r_dim,
      Output::tensorShape.total_col_dim(),
      Output::tensorShape.total_num_faces(), 1, false)));
  PACK((_llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE>()));
#else
  PACK((_llk_pack_hw_configure_<DST_ACCUM_MODE, ckernel::PackMode::Default>(
      Output::format, Output::format, Output::pageWords,
      Output::tensorShape.face_r_dim, Output::tensorShape.total_num_faces(),
      hasPartialFace(Output::tensorShape), isNarrowTile(Output::tensorShape),
      0)));
  PACK((_llk_pack_init_<ckernel::PackMode::Default>(
      Output::format, Output::tensorShape.face_r_dim,
      Output::tensorShape.total_num_faces(),
      hasPartialFace(Output::tensorShape), isNarrowTile(Output::tensorShape),
      1)));
  PACK((_llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE,
                             ckernel::PackMode::Default>(
      Output::tensorShape.face_r_dim, isNarrowTile(Output::tensorShape))));
#endif
}

template <typename Output, bool ReconfigureTileDimensions = false>
inline void reconfigurePack() {
#if defined(ARCH_BLACKHOLE)
  PACK((_llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
      Output::format, Output::format, Output::pageWords,
      Output::tensorShape.total_col_dim(),
      Output::tensorShape.total_num_faces(),
      hasPartialFace(Output::tensorShape))));
  if constexpr (ReconfigureTileDimensions) {
    PACK((_llk_pack_init_<ckernel::PackMode::Default, false, true>(
        Output::format, Output::tensorShape.face_r_dim,
        Output::tensorShape.total_col_dim(),
        Output::tensorShape.total_num_faces(), 1, false)));
  }
#else
  PACK((_llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
      Output::format, Output::format, Output::pageWords,
      Output::tensorShape.face_r_dim, Output::tensorShape.total_num_faces(),
      hasPartialFace(Output::tensorShape), isNarrowTile(Output::tensorShape))));
  if constexpr (ReconfigureTileDimensions) {
    PACK((_llk_pack_init_<ckernel::PackMode::Default, false, true>(
        Output::format, Output::tensorShape.face_r_dim,
        Output::tensorShape.total_num_faces(),
        hasPartialFace(Output::tensorShape), isNarrowTile(Output::tensorShape),
        1)));
  }
#endif
}

template <typename Lhs, typename Rhs>
inline void executeMatmul(uint32_t destination, uint32_t transpose,
                          uint32_t columns, uint32_t rows) {
#if defined(ARCH_BLACKHOLE) && defined(TRISC_MATH)
  bool throttled = (*ckernel::throttle_ptr % 2) != 0;
  if (throttled) {
    if (ckernel::throttled_mop_status != 1) {
      _llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE_MAX>(
          Lhs::tensorShape.total_row_dim(), Lhs::tensorShape.total_col_dim(),
          Rhs::tensorShape.total_row_dim(), Rhs::tensorShape.total_col_dim(),
          hasPartialFace(Lhs::tensorShape), transpose, columns, rows);
      ckernel::throttled_mop_status = 1;
    }
    llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(destination, columns, rows);
  } else {
    if (ckernel::throttled_mop_status != 0) {
      _llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE>(
          Lhs::tensorShape.total_row_dim(), Lhs::tensorShape.total_col_dim(),
          Rhs::tensorShape.total_row_dim(), Rhs::tensorShape.total_col_dim(),
          hasPartialFace(Lhs::tensorShape), transpose, columns, rows);
      ckernel::throttled_mop_status = 0;
    }
    llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(destination, columns, rows);
  }
#elif defined(TRISC_MATH)
  llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(destination, columns, rows);
#endif
}
} // namespace ttlang::l1::target
#endif
