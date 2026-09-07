// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifndef TTLANG_COMPILER_L1_COMPUTE_H
#define TTLANG_COMPILER_L1_COMPUTE_H
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
namespace ttlang::l1 {
template <uint32_t Format, uint32_t PageBytes, uint32_t TileHeight,
          uint32_t TileWidth, uint32_t PagesPerBlock, uint32_t BlockCount,
          uint32_t StorageCapacityPages, uint32_t PayloadOffset,
          int32_t PayloadCommonArgIndex, bool DirectToDestination>
class Operand
    : public Buffer<PageBytes, PagesPerBlock, BlockCount, StorageCapacityPages,
                    PayloadOffset, PayloadCommonArgIndex> {
public:
  using Buffer<PageBytes, PagesPerBlock, BlockCount, StorageCapacityPages,
               PayloadOffset, PayloadCommonArgIndex>::Buffer;
  static constexpr uint32_t format = Format;
  static constexpr bool directToDestination = DirectToDestination;
  static constexpr uint32_t unpackFormat =
      Format == static_cast<uint32_t>(DataFormat::Float32) &&
              !DirectToDestination
          ? static_cast<uint32_t>(DataFormat::Tf32)
          : Format;
  static constexpr uint32_t pageWords = PageBytes / 16;
  static constexpr uint32_t height = TileHeight;
  static constexpr uint32_t width = TileWidth;
  static constexpr uint32_t faceRowHeight = TileHeight < 16 ? TileHeight : 16;
  static constexpr uint32_t faceRowCount = TileHeight > 16 ? 2 : 1;
  static constexpr uint32_t faceColumnCount = TileWidth > 16 ? 2 : 1;
  static constexpr uint32_t faceCount = faceRowCount * faceColumnCount;
  static constexpr bool partialFace = TileHeight < 16;
  static constexpr bool narrowTile = TileWidth == 16;
  static constexpr ckernel::TensorShape tensorShape =
      ckernel::make_tensor_shape(faceRowHeight, 16, faceRowCount,
                                 faceColumnCount);
  static_assert(PageBytes % 16 == 0,
                "compute page size must be 16-byte aligned");
  uint32_t readTile(uint32_t tile) const {
    return target::toLlkTileAddress(this->get_read_ptr(), tile, pageWords);
  }
  uint32_t writeTile(uint32_t tile) const {
    return target::toLlkTileAddress(this->get_write_ptr(), tile, pageWords);
  }
};

template <uint32_t Format, uint32_t PageBytes, uint32_t TileHeight,
          uint32_t TileWidth, uint32_t PagesPerBlock, uint32_t BlockCount,
          uint32_t StorageCapacityPages, uint32_t StateOffset,
          uint32_t PayloadOffset, int32_t PayloadCommonArgIndex,
          bool DirectToDestination>
class ComputeDFBDescriptor
    : public Operand<Format, PageBytes, TileHeight, TileWidth, PagesPerBlock,
                     BlockCount, StorageCapacityPages, PayloadOffset,
                     PayloadCommonArgIndex, DirectToDestination> {
public:
  using Operand<Format, PageBytes, TileHeight, TileWidth, PagesPerBlock,
                BlockCount, StorageCapacityPages, PayloadOffset,
                PayloadCommonArgIndex, DirectToDestination>::Operand;
  /// Binds this descriptor to its compile-time allocation in the core arena.
  static ComputeDFBDescriptor bind() {
    return ComputeDFBDescriptor(target::arenaBase() + StateOffset);
  }
};

namespace target {
template <typename Source>
inline void copy_tile_init(Source source);
template <typename Lhs, typename Rhs>
__attribute__((noinline)) inline void
matmulInitShape(uint32_t transpose, uint32_t columns, uint32_t rows,
                uint32_t inner) {
  UNPACK((_llk_unpack_AB_matmul_init_(
      transpose, columns, rows, inner, Rhs::faceRowHeight, Lhs::faceRowHeight,
      Rhs::faceCount, Lhs::faceCount, Rhs::partialFace, Lhs::partialFace)));
  MATH((_llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE>(
      Lhs::height, Lhs::width, Rhs::height, Rhs::width, Lhs::partialFace,
      transpose, columns, rows)));
  resetMatmulThrottleState();
}
template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension, typename Input,
          typename Output>
__attribute__((noinline)) inline void reduceInitShape() {
  UNPACK((_llk_unpack_AB_reduce_init_<Pool, Dimension>(Input::tensorShape)));
  MATH((_llk_math_reduce_init_<Pool, Dimension, DST_ACCUM_MODE, MATH_FIDELITY>(
      Input::tensorShape)));
  PACK((_llk_pack_reduce_mask_config_<Dimension, ckernel::PackMode::Default>(
      Output::faceRowHeight)));
}
template <ckernel::BroadcastType Broadcast, typename Source>
inline void unaryBcastInit(Source) {
  constexpr auto copyType =
      Source::directToDestination || Broadcast == ckernel::BroadcastType::NONE
          ? ckernel::DataCopyType::A2D
          : ckernel::DataCopyType::B2D;
  UNPACK((_llk_unpack_A_init_<Broadcast, false,
                              ckernel::EltwiseBinaryReuseDestType::NONE,
                              Source::directToDestination>(
      0, 0, Source::tensorShape, Source::format, Source::unpackFormat)));
  initializeUnaryDataCopy<copyType, Broadcast, Source>();
}
template <ckernel::BroadcastType Broadcast, typename Source>
inline void unary_bcast(Source source, uint32_t tile, uint32_t destination) {
  constexpr auto copyType =
      Source::directToDestination || Broadcast == ckernel::BroadcastType::NONE
          ? ckernel::DataCopyType::A2D
          : ckernel::DataCopyType::B2D;
  UNPACK((_llk_unpack_A_<Broadcast, false,
                         ckernel::EltwiseBinaryReuseDestType::NONE,
                         Source::directToDestination>(
      source.readTile(tile), Source::format, Source::unpackFormat)));
  MATH((
      _llk_math_eltwise_unary_datacopy_<copyType, DST_SYNC_MODE, DST_ACCUM_MODE,
                                        Broadcast, Source::directToDestination>(
          destination, Source::format, Source::unpackFormat)));
}
/// Configuration is processor-local and lasts for one kernel invocation.
class ComputeContext {
  bool initialized = false;
  uint32_t sourceAFormat = 0;
  uint32_t sourceAUnpackFormat = 0;
  uint32_t sourceAPageWords = 0;
  uint32_t sourceAFaceRowHeight = 0;
  uint32_t sourceAFaceCount = 0;
  uint32_t sourceBFormat = 0;
  uint32_t sourceBUnpackFormat = 0;
  uint32_t sourceBPageWords = 0;
  uint32_t sourceBFaceRowHeight = 0;
  uint32_t sourceBFaceCount = 0;
  uint32_t outputFormat = 0;
  uint32_t outputPageWords = 0;
  uint32_t outputHeight = 0;
  uint32_t outputWidth = 0;

  template <typename SourceA, typename SourceB>
  void recordInputs() {
    sourceAFormat = SourceA::format;
    sourceAUnpackFormat = SourceA::unpackFormat;
    sourceAPageWords = SourceA::pageWords;
    sourceAFaceRowHeight = SourceA::faceRowHeight;
    sourceAFaceCount = SourceA::faceCount;
    sourceBFormat = SourceB::format;
    sourceBUnpackFormat = SourceB::unpackFormat;
    sourceBPageWords = SourceB::pageWords;
    sourceBFaceRowHeight = SourceB::faceRowHeight;
    sourceBFaceCount = SourceB::faceCount;
  }

  template <typename Output>
  void recordOutput() {
    outputFormat = Output::format;
    outputPageWords = Output::pageWords;
    outputHeight = Output::height;
    outputWidth = Output::width;
  }

public:
  template <typename SourceA, typename SourceB, typename Output>
  void configure(SourceA, SourceB, Output) {
    if (!initialized) {
      UNPACK((_llk_unpack_hw_configure_<DST_ACCUM_MODE>(
          SourceA::format, SourceB::format, SourceA::unpackFormat,
          SourceB::unpackFormat, SourceA::faceRowHeight, SourceB::faceRowHeight,
          SourceA::faceCount, SourceB::faceCount, SourceA::pageWords,
          SourceB::pageWords)));
      MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
      MATH((_llk_math_hw_configure_<DST_ACCUM_MODE>(SourceA::unpackFormat,
                                                    SourceB::unpackFormat)));
      initializePack<Output>();
      initialized = true;
      recordInputs<SourceA, SourceB>();
    } else {
      configureInputs<SourceA, SourceB>();
      if (outputHeight != Output::height || outputWidth != Output::width) {
        reconfigurePack<Output, true>();
      } else if (outputFormat != Output::format ||
                 outputPageWords != Output::pageWords) {
        reconfigurePack<Output>();
      }
    }
    recordOutput<Output>();
  }
  template <typename SourceA, typename SourceB>
  __attribute__((noinline)) void configureInputs() {
    bool sourceAGeometryChanged =
        sourceAPageWords != SourceA::pageWords ||
        sourceAFaceRowHeight != SourceA::faceRowHeight ||
        sourceAFaceCount != SourceA::faceCount;
    if (sourceAFormat != SourceA::format ||
        sourceAUnpackFormat != SourceA::unpackFormat ||
        sourceAGeometryChanged) {
      if (sourceAGeometryChanged) {
        UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<
                DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
            SourceA::format, SourceA::unpackFormat, SourceA::pageWords,
            SourceA::faceRowHeight, SourceA::faceCount)));
      } else {
        UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<
                DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            SourceA::format, SourceA::unpackFormat, SourceA::pageWords,
            SourceA::faceRowHeight, SourceA::faceCount)));
      }
      if (sourceAUnpackFormat != SourceA::unpackFormat) {
        MATH((_llk_math_reconfig_data_format_srca_<DST_ACCUM_MODE>(
            SourceA::unpackFormat)));
      }
    }
    bool sourceBGeometryChanged =
        sourceBPageWords != SourceB::pageWords ||
        sourceBFaceRowHeight != SourceB::faceRowHeight ||
        sourceBFaceCount != SourceB::faceCount;
    if (sourceBFormat != SourceB::format ||
        sourceBUnpackFormat != SourceB::unpackFormat ||
        sourceBGeometryChanged) {
      if (sourceBGeometryChanged) {
        UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<
                DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
            SourceB::format, SourceB::unpackFormat, SourceB::pageWords,
            SourceB::faceRowHeight, SourceB::faceCount)));
      } else {
        UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<
                DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            SourceB::format, SourceB::unpackFormat, SourceB::pageWords,
            SourceB::faceRowHeight, SourceB::faceCount)));
      }
      if (sourceBUnpackFormat != SourceB::unpackFormat) {
        MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE>(
            SourceB::unpackFormat)));
      }
    }
    recordInputs<SourceA, SourceB>();
  }
  template <typename Lhs, typename Rhs, typename Output>
  void matmulInit(Lhs lhs, Rhs rhs, Output output, uint32_t transpose) {
    matmulBlockInit(lhs, rhs, output, transpose, 1, 1, 1);
  }
  template <typename Lhs, typename Rhs, typename Output>
  void matmulBlockInit(Lhs lhs, Rhs rhs, Output output, uint32_t transpose,
                       uint32_t columns, uint32_t rows, uint32_t inner) {
    configure(rhs, lhs, output);
    matmulInitShape<Lhs, Rhs>(transpose, columns, rows, inner);
  }
  template <typename Lhs, typename Rhs>
  void matmulInitShort(Lhs lhs, Rhs rhs, uint32_t transpose) {
    matmulBlockInitShort(lhs, rhs, transpose, 1, 1, 1);
  }
  template <typename Lhs, typename Rhs>
  void matmulBlockInitShort(Lhs, Rhs, uint32_t transpose, uint32_t columns,
                            uint32_t rows, uint32_t inner) {
    configureInputs<Rhs, Lhs>();
    matmulInitShape<Lhs, Rhs>(transpose, columns, rows, inner);
  }
  template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension,
            typename Input, typename Scaler, typename Output>
  void reduceInit(Input input, Scaler scaler, Output output) {
    if constexpr (Dimension == ckernel::ReduceDim::REDUCE_ROW &&
                  Pool != ckernel::PoolType::MAX) {
      configure(scaler, input, output);
    } else {
      configure(input, scaler, output);
    }
    reduceInitShape<Pool, Dimension, Input, Output>();
  }
  template <ckernel::BroadcastType Broadcast, typename Source, typename Output>
  void broadcastInit(Source source, Output output) {
    configure(source, source, output);
    unaryBcastInit<Broadcast>(source);
  }
  template <typename Source, typename Output>
  void transposeInit(Source source, Output output) {
    configure(source, source, output);
    copy_tile_init(source);
    UNPACK((_llk_unpack_A_init_<ckernel::BroadcastType::NONE,
                                !Source::directToDestination,
                                ckernel::EltwiseBinaryReuseDestType::NONE,
                                Source::directToDestination>(
        true, !Source::directToDestination, Source::tensorShape, Source::format,
        Source::unpackFormat)));
    if constexpr (Source::directToDestination) {
      MATH((llk_math_transpose_dest_init<false, true>()));
    }
  }
  template <typename Source, typename Output>
  void configure(Source source, Output output) {
    configure(source, source, output);
    copy_tile_init(source);
  }
};

template <typename Source>
__attribute__((noinline)) inline void copyInitFormats() {
  UNPACK((_llk_unpack_A_init_<ckernel::BroadcastType::NONE, false,
                              ckernel::EltwiseBinaryReuseDestType::NONE,
                              Source::directToDestination>(
      0, 0, Source::tensorShape, Source::format, Source::unpackFormat)));
  initializeUnaryDataCopy<ckernel::DataCopyType::A2D,
                          ckernel::BroadcastType::NONE, Source>();
  MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
}
template <uint32_t Format, uint32_t UnpackFormat, bool Direct>
__attribute__((noinline)) inline void copyAtAddress(uint32_t address,
                                                    uint32_t destination) {
  UNPACK((_llk_unpack_A_<ckernel::BroadcastType::NONE, false,
                         ckernel::EltwiseBinaryReuseDestType::NONE, Direct>(
      address, Format, UnpackFormat)));
  MATH((_llk_math_eltwise_unary_datacopy_<ckernel::DataCopyType::A2D,
                                          DST_SYNC_MODE, DST_ACCUM_MODE,
                                          ckernel::BroadcastType::NONE, Direct>(
      destination, Format, UnpackFormat)));
}

template <typename Source>
inline void copy_tile_init(Source) {
  copyInitFormats<Source>();
}
template <typename Source>
inline void copy_tile(Source source, uint32_t tile, uint32_t destination) {
  UNPACK((copyAtAddress<Source::format, Source::unpackFormat,
                        Source::directToDestination>(source.readTile(tile),
                                                     destination)));
  MATH((copyAtAddress<Source::format, Source::unpackFormat,
                      Source::directToDestination>(0, destination)));
}

template <typename Source>
inline void transpose_wh_tile(Source source, uint32_t tile,
                              uint32_t destination) {
  copy_tile(source, tile, destination);
  if constexpr (Source::directToDestination) {
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    MATH((llk_math_transpose_dest<false, true>(destination)));
  }
}
template <ckernel::EltwiseBinaryType Operation,
          ckernel::EltwiseBinaryReuseDestType Reuse, typename Source>
inline void binary_dest_reuse_tiles_init(Source) {
  UNPACK((_llk_unpack_A_init_<ckernel::BroadcastType::NONE, true, Reuse>(
      0, 0, Source::tensorShape, Source::format, Source::unpackFormat)));
  MATH((_llk_math_eltwise_binary_init_<Operation, ckernel::BroadcastType::NONE,
                                       MATH_FIDELITY, Reuse>(
      Source::tensorShape, false)));
}
template <ckernel::EltwiseBinaryType Operation,
          ckernel::EltwiseBinaryReuseDestType Reuse, typename Source>
inline void binary_dest_reuse_tiles(Source source, uint32_t tile,
                                    uint32_t destination) {
  UNPACK((_llk_unpack_A_<ckernel::BroadcastType::NONE, true, Reuse>(
      source.readTile(tile), Source::format, Source::unpackFormat)));
  MATH((_llk_math_eltwise_binary_<
        Operation, ckernel::BroadcastType::NONE, DST_SYNC_MODE, DST_ACCUM_MODE,
        get_effective_math_fidelity<Operation, MATH_FIDELITY>(), Reuse>(
      Source::tensorShape, destination, true)));
}
template <ckernel::EltwiseBinaryType Operation, typename Source>
__attribute__((noinline)) inline void binaryInit() {
  UNPACK((_llk_unpack_AB_init_<ckernel::BroadcastType::NONE>(
      Source::tensorShape, ckernel::Transpose::None)));
  MATH((_llk_math_eltwise_binary_init_<
        Operation, ckernel::BroadcastType::NONE,
        get_effective_math_fidelity<Operation, MATH_FIDELITY>()>(
      Source::tensorShape, false)));
}
// Sharing instruction emission bounds code size independently of storage
// identities.
template <ckernel::EltwiseBinaryType Operation, typename Source>
__attribute__((noinline)) inline void
binaryAtAddresses(uint32_t sourceA, uint32_t sourceB, uint32_t destination) {
  UNPACK((_llk_unpack_AB_<ckernel::BroadcastType::NONE>(sourceA, sourceB)));
  MATH((_llk_math_eltwise_binary_<
        Operation, ckernel::BroadcastType::NONE, DST_SYNC_MODE, DST_ACCUM_MODE,
        get_effective_math_fidelity<Operation, MATH_FIDELITY>()>(
      Source::tensorShape, destination, true)));
}
template <ckernel::EltwiseBinaryType Operation, typename SourceA,
          typename SourceB>
inline void binary(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                   uint32_t tileB, uint32_t destination) {
  UNPACK((binaryAtAddresses<Operation, SourceA>(
      sourceA.readTile(tileA), sourceB.readTile(tileB), destination)));
  MATH((binaryAtAddresses<Operation, SourceA>(0, 0, destination)));
}
template <typename SourceA, typename SourceB>
inline void add_tiles_init(SourceA sourceA, SourceB sourceB) {
  binaryInit<ckernel::EltwiseBinaryType::ELWADD, SourceA>();
}
template <typename SourceA, typename SourceB>
inline void add_tiles(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                      uint32_t tileB, uint32_t destination) {
  binary<ckernel::EltwiseBinaryType::ELWADD>(sourceA, sourceB, tileA, tileB,
                                             destination);
}
template <typename SourceA, typename SourceB>
inline void sub_tiles_init(SourceA sourceA, SourceB sourceB) {
  binaryInit<ckernel::EltwiseBinaryType::ELWSUB, SourceA>();
}
template <typename SourceA, typename SourceB>
inline void sub_tiles(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                      uint32_t tileB, uint32_t destination) {
  binary<ckernel::EltwiseBinaryType::ELWSUB>(sourceA, sourceB, tileA, tileB,
                                             destination);
}
template <typename SourceA, typename SourceB>
inline void mul_tiles_init(SourceA sourceA, SourceB sourceB) {
  binaryInit<ckernel::EltwiseBinaryType::ELWMUL, SourceA>();
}
template <typename SourceA, typename SourceB>
inline void mul_tiles(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                      uint32_t tileB, uint32_t destination) {
  binary<ckernel::EltwiseBinaryType::ELWMUL>(sourceA, sourceB, tileA, tileB,
                                             destination);
}
template <typename Lhs, typename Rhs>
__attribute__((noinline)) inline void
matmulBlockAtAddresses(uint32_t lhs, uint32_t rhs, uint32_t lhsPageWords,
                       uint32_t rhsPageWords, uint32_t destination,
                       uint32_t transpose, uint32_t columns, uint32_t rows,
                       uint32_t inner) {
  UNPACK((_llk_unpack_AB_matmul_(lhs, rhs, 0, 0, lhsPageWords, rhsPageWords,
                                 Rhs::partialFace, Lhs::partialFace, columns,
                                 rows, inner)));
  executeMatmul<Lhs, Rhs>(destination, transpose, columns, rows);
}
template <typename Lhs, typename Rhs>
inline void matmul_block(Lhs lhs, Rhs rhs, uint32_t lhsTile, uint32_t rhsTile,
                         uint32_t destination, uint32_t transpose,
                         uint32_t columns, uint32_t rows, uint32_t inner) {
  UNPACK((matmulBlockAtAddresses<Lhs, Rhs>(
      lhs.readTile(lhsTile), rhs.readTile(rhsTile), Lhs::pageWords,
      Rhs::pageWords, destination, transpose, columns, rows, inner)));
  MATH((matmulBlockAtAddresses<Lhs, Rhs>(0, 0, 0, 0, destination, transpose,
                                         columns, rows, inner)));
}
template <typename Lhs, typename Rhs>
inline void matmul_tiles(Lhs lhs, Rhs rhs, uint32_t lhsTile, uint32_t rhsTile,
                         uint32_t destination) {
  matmul_block(lhs, rhs, lhsTile, rhsTile, destination, 0, 1, 1, 1);
}
template <typename Lhs, typename Rhs>
inline void
matmul_block_strided(Lhs lhs, Rhs rhs, uint32_t lhsTile, uint32_t rhsTile,
                     uint32_t destination, uint32_t transpose, uint32_t columns,
                     uint32_t rows, uint32_t inner, uint32_t rhsStride) {
  if (transpose) {
    matmulInitShape<Lhs, Rhs>(transpose, 1, 1, 1);
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t column = 0; column < columns; ++column) {
        for (uint32_t reduction = 0; reduction < inner; ++reduction) {
          matmul_block(lhs, rhs, lhsTile + row * inner + reduction,
                       rhsTile + column * inner + reduction * rhsStride,
                       destination + row * columns + column, transpose, 1, 1,
                       1);
        }
      }
    }
    return;
  }
  for (uint32_t reduction = 0; reduction < inner; ++reduction) {
    matmul_block(lhs, rhs, lhsTile + reduction, rhsTile + reduction * rhsStride,
                 destination, transpose, columns, rows, inner);
  }
}
template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension, typename Input>
__attribute__((noinline)) inline void
reduceAtAddresses(uint32_t input, uint32_t scaler, uint32_t destination) {
  UNPACK((_llk_unpack_AB_reduce_<Pool, Dimension>(input, scaler)));
  MATH((_llk_math_reduce_<Pool, Dimension, DST_ACCUM_MODE, MATH_FIDELITY>(
      destination, Input::tensorShape)));
}
template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension, typename Input,
          typename Scaler>
inline void reduce_tile(Input input, Scaler scaler, uint32_t inputTile,
                        uint32_t scalerTile, uint32_t destination) {
  UNPACK((reduceAtAddresses<Pool, Dimension, Input>(
      input.readTile(inputTile), scaler.readTile(scalerTile), destination)));
  MATH((reduceAtAddresses<Pool, Dimension, Input>(0, 0, destination)));
}
inline void reduce_uninit() {
  MATH((_llk_math_reduce_uninit_()));
  PACK((_llk_pack_reduce_mask_clear_()));
}
__attribute__((noinline)) inline void packAtAddress(uint32_t destination,
                                                    uint32_t address) {
  PACK((_llk_pack_<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::PackMode::Default>(
      destination, address)));
}
template <bool OutOfOrder, typename Output>
inline void pack_tile(uint32_t destination, Output output, uint32_t tile) {
  static_assert(OutOfOrder,
                "compiler-l1 packing requires an explicit tile index");
  PACK((packAtAddress(destination, output.writeTile(tile))));
}
template <bool OutOfOrder, typename Output>
inline void pack_waited_tile(uint32_t destination, Output output,
                             uint32_t tile) {
  static_assert(OutOfOrder,
                "compiler-l1 packing requires an explicit tile index");
  PACK((packAtAddress(destination, output.readTile(tile))));
}
} // namespace target
} // namespace ttlang::l1
#endif
