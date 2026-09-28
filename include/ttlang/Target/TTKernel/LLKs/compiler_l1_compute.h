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
/// Properties that select one LLK implementation independently of SRAM
/// placement.
template <uint32_t Format, uint32_t PageBytes, uint32_t TileHeight,
          uint32_t TileWidth, bool DirectToDestination>
struct ComputeTileMetadata {
  static constexpr uint32_t format = Format;
  static constexpr bool directToDestination = DirectToDestination;
  static constexpr uint32_t unpackFormat =
      Format == static_cast<uint32_t>(DataFormat::Float32) &&
              !DirectToDestination
          ? static_cast<uint32_t>(DataFormat::Tf32)
          : Format;
  static constexpr uint32_t pageWords = PageBytes / target::llkAddressWordBytes;
  static constexpr ckernel::TensorShape tensorShape =
      target::makeTensorShape<TileHeight, TileWidth>();
  static_assert(PageBytes % target::llkAddressWordBytes == 0,
                "compute page size must use whole LLK address words");
};

template <uint32_t Format, uint32_t PageBytes, uint32_t TileHeight,
          uint32_t TileWidth, uint32_t PagesPerBlock, uint32_t BlockCount,
          uint32_t StorageCapacityPages, uint32_t PayloadOffset,
          int32_t PayloadCommonArgIndex, bool DirectToDestination,
          bool ReloadOwnedSequence = true>
class Operand
    : public Buffer<PageBytes, PagesPerBlock, BlockCount, StorageCapacityPages,
                    PayloadOffset, PayloadCommonArgIndex, ReloadOwnedSequence> {
public:
  using BufferBase =
      Buffer<PageBytes, PagesPerBlock, BlockCount, StorageCapacityPages,
             PayloadOffset, PayloadCommonArgIndex, ReloadOwnedSequence>;
  using BufferBase::BufferBase;
  explicit Operand(const BufferBase &buffer) : BufferBase(buffer) {}
  using TileMetadata = ComputeTileMetadata<Format, PageBytes, TileHeight,
                                           TileWidth, DirectToDestination>;
  static constexpr uint32_t format = TileMetadata::format;
  static constexpr bool directToDestination = TileMetadata::directToDestination;
  static constexpr uint32_t unpackFormat = TileMetadata::unpackFormat;
  static constexpr uint32_t pageWords = TileMetadata::pageWords;
  static constexpr ckernel::TensorShape tensorShape = TileMetadata::tensorShape;
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
                     PayloadCommonArgIndex, DirectToDestination, true> {
public:
  using Operand<Format, PageBytes, TileHeight, TileWidth, PagesPerBlock,
                BlockCount, StorageCapacityPages, PayloadOffset,
                PayloadCommonArgIndex, DirectToDestination, true>::Operand;
  /// Binds this descriptor to its compile-time allocation in the core arena.
  static ComputeDFBDescriptor bind() {
    return ComputeDFBDescriptor(target::arenaBase() + StateOffset);
  }
};

namespace target {
template <typename Source>
inline void copy_tile_init(Source source);
template <typename LhsTile, typename RhsTile>
__attribute__((noinline)) inline void
matmulInitShape(uint32_t transpose, uint32_t columns, uint32_t rows,
                uint32_t inner) {
  UNPACK((_llk_unpack_AB_matmul_init_(
      transpose, columns, rows, inner, RhsTile::tensorShape.face_r_dim,
      LhsTile::tensorShape.face_r_dim, RhsTile::tensorShape.total_num_faces(),
      LhsTile::tensorShape.total_num_faces(),
      target::hasPartialFace(RhsTile::tensorShape),
      target::hasPartialFace(LhsTile::tensorShape))));
  MATH((_llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE>(
      LhsTile::tensorShape.total_row_dim(),
      LhsTile::tensorShape.total_col_dim(),
      RhsTile::tensorShape.total_row_dim(),
      RhsTile::tensorShape.total_col_dim(),
      target::hasPartialFace(LhsTile::tensorShape), transpose, columns, rows)));
  resetMatmulThrottleState();
}
template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension,
          typename InputTile, typename OutputTile>
__attribute__((noinline)) inline void reduceInitShape() {
  UNPACK(
      (_llk_unpack_AB_reduce_init_<Pool, Dimension>(InputTile::tensorShape)));
  MATH((_llk_math_reduce_init_<Pool, Dimension, DST_ACCUM_MODE, MATH_FIDELITY>(
      InputTile::tensorShape)));
  PACK((_llk_pack_reduce_mask_config_<Dimension, ckernel::PackMode::Default>(
      OutputTile::tensorShape.face_r_dim)));
}
template <ckernel::BroadcastType Broadcast, typename SourceTile>
__attribute__((noinline)) inline void unaryBcastInitTile() {
  constexpr auto copyType = SourceTile::directToDestination ||
                                    Broadcast == ckernel::BroadcastType::NONE
                                ? ckernel::DataCopyType::A2D
                                : ckernel::DataCopyType::B2D;
  UNPACK((_llk_unpack_A_init_<Broadcast, false,
                              ckernel::EltwiseBinaryReuseDestType::NONE,
                              SourceTile::directToDestination>(
      0, 0, SourceTile::tensorShape, SourceTile::format,
      SourceTile::unpackFormat)));
  initializeUnaryDataCopy<copyType, Broadcast, SourceTile>();
}
template <ckernel::BroadcastType Broadcast, typename Source>
inline void unaryBcastInit(Source) {
  unaryBcastInitTile<Broadcast, typename Source::TileMetadata>();
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
  uint64_t activeSourceAConfiguration = 0;
  uint64_t activeSourceBConfiguration = 0;
  uint64_t activeOutputConfiguration = 0;
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

  template <typename SourceTile>
  static constexpr uint64_t inputConfiguration() {
    static_assert(SourceTile::format <= UINT8_MAX &&
                  SourceTile::unpackFormat <= UINT8_MAX &&
                  SourceTile::tensorShape.face_r_dim <= UINT8_MAX &&
                  SourceTile::tensorShape.total_num_faces() <= UINT8_MAX);
    return uint64_t{SourceTile::format} |
           (uint64_t{SourceTile::unpackFormat} << 8) |
           (uint64_t{SourceTile::pageWords} << 16) |
           (uint64_t{SourceTile::tensorShape.face_r_dim} << 48) |
           (uint64_t{SourceTile::tensorShape.total_num_faces()} << 56);
  }

  template <typename OutputTile>
  static constexpr uint64_t outputConfiguration() {
    static_assert(OutputTile::format <= UINT8_MAX &&
                  OutputTile::tensorShape.total_row_dim() <= UINT8_MAX &&
                  OutputTile::tensorShape.total_col_dim() <= UINT8_MAX);
    return uint64_t{OutputTile::format} |
           (uint64_t{OutputTile::pageWords} << 8) |
           (uint64_t{OutputTile::tensorShape.total_row_dim()} << 40) |
           (uint64_t{OutputTile::tensorShape.total_col_dim()} << 48);
  }

  template <typename SourceATile, typename SourceBTile>
  void recordInputs() {
    activeSourceAConfiguration = inputConfiguration<SourceATile>();
    activeSourceBConfiguration = inputConfiguration<SourceBTile>();
    sourceAFormat = SourceATile::format;
    sourceAUnpackFormat = SourceATile::unpackFormat;
    sourceAPageWords = SourceATile::pageWords;
    sourceAFaceRowHeight = SourceATile::tensorShape.face_r_dim;
    sourceAFaceCount = SourceATile::tensorShape.total_num_faces();
    sourceBFormat = SourceBTile::format;
    sourceBUnpackFormat = SourceBTile::unpackFormat;
    sourceBPageWords = SourceBTile::pageWords;
    sourceBFaceRowHeight = SourceBTile::tensorShape.face_r_dim;
    sourceBFaceCount = SourceBTile::tensorShape.total_num_faces();
  }

  template <typename OutputTile>
  void recordOutput() {
    activeOutputConfiguration = outputConfiguration<OutputTile>();
    outputFormat = OutputTile::format;
    outputPageWords = OutputTile::pageWords;
    outputHeight = OutputTile::tensorShape.total_row_dim();
    outputWidth = OutputTile::tensorShape.total_col_dim();
  }

  template <typename SourceATile, typename SourceBTile>
  __attribute__((noinline)) void configureInputTiles() {
    bool sourceAGeometryChanged =
        sourceAPageWords != SourceATile::pageWords ||
        sourceAFaceRowHeight != SourceATile::tensorShape.face_r_dim ||
        sourceAFaceCount != SourceATile::tensorShape.total_num_faces();
    if (sourceAFormat != SourceATile::format ||
        sourceAUnpackFormat != SourceATile::unpackFormat ||
        sourceAGeometryChanged) {
      if (sourceAGeometryChanged) {
        UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<
                DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
            SourceATile::format, SourceATile::unpackFormat,
            SourceATile::pageWords, SourceATile::tensorShape.face_r_dim,
            SourceATile::tensorShape.total_num_faces())));
      } else {
        UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<
                DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            SourceATile::format, SourceATile::unpackFormat,
            SourceATile::pageWords, SourceATile::tensorShape.face_r_dim,
            SourceATile::tensorShape.total_num_faces())));
      }
      if (sourceAUnpackFormat != SourceATile::unpackFormat) {
        MATH((_llk_math_reconfig_data_format_srca_<DST_ACCUM_MODE>(
            SourceATile::unpackFormat)));
      }
    }
    bool sourceBGeometryChanged =
        sourceBPageWords != SourceBTile::pageWords ||
        sourceBFaceRowHeight != SourceBTile::tensorShape.face_r_dim ||
        sourceBFaceCount != SourceBTile::tensorShape.total_num_faces();
    if (sourceBFormat != SourceBTile::format ||
        sourceBUnpackFormat != SourceBTile::unpackFormat ||
        sourceBGeometryChanged) {
      if (sourceBGeometryChanged) {
        UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<
                DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
            SourceBTile::format, SourceBTile::unpackFormat,
            SourceBTile::pageWords, SourceBTile::tensorShape.face_r_dim,
            SourceBTile::tensorShape.total_num_faces())));
      } else {
        UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<
                DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            SourceBTile::format, SourceBTile::unpackFormat,
            SourceBTile::pageWords, SourceBTile::tensorShape.face_r_dim,
            SourceBTile::tensorShape.total_num_faces())));
      }
      if (sourceBUnpackFormat != SourceBTile::unpackFormat) {
        MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE>(
            SourceBTile::unpackFormat)));
      }
    }
    recordInputs<SourceATile, SourceBTile>();
  }

  template <typename SourceATile, typename SourceBTile, typename OutputTile>
  __attribute__((noinline)) void configureTiles() {
    if (initialized &&
        activeSourceAConfiguration == inputConfiguration<SourceATile>() &&
        activeSourceBConfiguration == inputConfiguration<SourceBTile>() &&
        activeOutputConfiguration == outputConfiguration<OutputTile>()) {
      return;
    }
    if (!initialized) {
      UNPACK((_llk_unpack_hw_configure_<DST_ACCUM_MODE>(
          SourceATile::format, SourceBTile::format, SourceATile::unpackFormat,
          SourceBTile::unpackFormat, SourceATile::tensorShape.face_r_dim,
          SourceBTile::tensorShape.face_r_dim,
          SourceATile::tensorShape.total_num_faces(),
          SourceBTile::tensorShape.total_num_faces(), SourceATile::pageWords,
          SourceBTile::pageWords)));
      MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
      MATH((_llk_math_hw_configure_<DST_ACCUM_MODE>(
          SourceATile::unpackFormat, SourceBTile::unpackFormat)));
      initializePack<OutputTile>();
      initialized = true;
      recordInputs<SourceATile, SourceBTile>();
    } else {
      configureInputTiles<SourceATile, SourceBTile>();
      if (outputHeight != OutputTile::tensorShape.total_row_dim() ||
          outputWidth != OutputTile::tensorShape.total_col_dim()) {
        reconfigurePack<OutputTile, true>();
      } else if (outputFormat != OutputTile::format ||
                 outputPageWords != OutputTile::pageWords) {
        reconfigurePack<OutputTile>();
      }
    }
    recordOutput<OutputTile>();
  }

public:
  template <typename SourceA, typename SourceB, typename Output>
  void configure(SourceA, SourceB, Output) {
    configureTiles<typename SourceA::TileMetadata,
                   typename SourceB::TileMetadata,
                   typename Output::TileMetadata>();
  }
  template <typename SourceA, typename SourceB>
  void configureInputs() {
    configureInputTiles<typename SourceA::TileMetadata,
                        typename SourceB::TileMetadata>();
  }
  template <typename Lhs, typename Rhs, typename Output>
  void matmulInit(Lhs lhs, Rhs rhs, Output output, uint32_t transpose) {
    matmulBlockInit(lhs, rhs, output, transpose, 1, 1, 1);
  }
  template <typename Lhs, typename Rhs, typename Output>
  void matmulBlockInit(Lhs lhs, Rhs rhs, Output output, uint32_t transpose,
                       uint32_t columns, uint32_t rows, uint32_t inner) {
    configure(rhs, lhs, output);
    matmulInitShape<typename Lhs::TileMetadata, typename Rhs::TileMetadata>(
        transpose, columns, rows, inner);
  }
  template <typename Lhs, typename Rhs>
  void matmulInitShort(Lhs lhs, Rhs rhs, uint32_t transpose) {
    matmulBlockInitShort(lhs, rhs, transpose, 1, 1, 1);
  }
  template <typename Lhs, typename Rhs>
  void matmulBlockInitShort(Lhs, Rhs, uint32_t transpose, uint32_t columns,
                            uint32_t rows, uint32_t inner) {
    configureInputs<Rhs, Lhs>();
    matmulInitShape<typename Lhs::TileMetadata, typename Rhs::TileMetadata>(
        transpose, columns, rows, inner);
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
    reduceInitShape<Pool, Dimension, typename Input::TileMetadata,
                    typename Output::TileMetadata>();
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

template <typename SourceTile>
__attribute__((noinline)) inline void copyInitFormats() {
  UNPACK((_llk_unpack_A_init_<ckernel::BroadcastType::NONE, false,
                              ckernel::EltwiseBinaryReuseDestType::NONE,
                              SourceTile::directToDestination>(
      0, 0, SourceTile::tensorShape, SourceTile::format,
      SourceTile::unpackFormat)));
  initializeUnaryDataCopy<ckernel::DataCopyType::A2D,
                          ckernel::BroadcastType::NONE, SourceTile>();
  MATH((ckernel::math::_configure_preserve_zero_flag_state_()));
}
template <uint32_t Format, uint32_t UnpackFormat, bool Direct>
// Inlining exposes loop-invariant hardware addresses to the RISC compiler.
__attribute__((always_inline)) inline void copyAtAddress(uint32_t address,
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
  copyInitFormats<typename Source::TileMetadata>();
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
template <ckernel::EltwiseBinaryType Operation, typename SourceTile>
__attribute__((noinline)) inline void binaryInit() {
  UNPACK((_llk_unpack_AB_init_<ckernel::BroadcastType::NONE>(
      SourceTile::tensorShape, ckernel::Transpose::None)));
  MATH((_llk_math_eltwise_binary_init_<
        Operation, ckernel::BroadcastType::NONE,
        get_effective_math_fidelity<Operation, MATH_FIDELITY>()>(
      SourceTile::tensorShape, false)));
}
// Sharing instruction emission bounds code size independently of storage
// identities.
template <ckernel::EltwiseBinaryType Operation, typename SourceTile>
__attribute__((noinline)) inline void
binaryAtAddresses(uint32_t sourceA, uint32_t sourceB, uint32_t destination) {
  UNPACK((_llk_unpack_AB_<ckernel::BroadcastType::NONE>(sourceA, sourceB)));
  MATH((_llk_math_eltwise_binary_<
        Operation, ckernel::BroadcastType::NONE, DST_SYNC_MODE, DST_ACCUM_MODE,
        get_effective_math_fidelity<Operation, MATH_FIDELITY>()>(
      SourceTile::tensorShape, destination, true)));
}
template <ckernel::EltwiseBinaryType Operation, typename SourceA,
          typename SourceB>
inline void binary(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                   uint32_t tileB, uint32_t destination) {
  UNPACK((binaryAtAddresses<Operation, typename SourceA::TileMetadata>(
      sourceA.readTile(tileA), sourceB.readTile(tileB), destination)));
  MATH((binaryAtAddresses<Operation, typename SourceA::TileMetadata>(
      0, 0, destination)));
}
template <typename SourceA, typename SourceB>
inline void add_tiles_init(SourceA sourceA, SourceB sourceB) {
  binaryInit<ckernel::EltwiseBinaryType::ELWADD,
             typename SourceA::TileMetadata>();
}
template <typename SourceA, typename SourceB>
inline void add_tiles(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                      uint32_t tileB, uint32_t destination) {
  binary<ckernel::EltwiseBinaryType::ELWADD>(sourceA, sourceB, tileA, tileB,
                                             destination);
}
template <typename SourceA, typename SourceB>
inline void sub_tiles_init(SourceA sourceA, SourceB sourceB) {
  binaryInit<ckernel::EltwiseBinaryType::ELWSUB,
             typename SourceA::TileMetadata>();
}
template <typename SourceA, typename SourceB>
inline void sub_tiles(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                      uint32_t tileB, uint32_t destination) {
  binary<ckernel::EltwiseBinaryType::ELWSUB>(sourceA, sourceB, tileA, tileB,
                                             destination);
}
template <typename SourceA, typename SourceB>
inline void mul_tiles_init(SourceA sourceA, SourceB sourceB) {
  binaryInit<ckernel::EltwiseBinaryType::ELWMUL,
             typename SourceA::TileMetadata>();
}
template <typename SourceA, typename SourceB>
inline void mul_tiles(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                      uint32_t tileB, uint32_t destination) {
  binary<ckernel::EltwiseBinaryType::ELWMUL>(sourceA, sourceB, tileA, tileB,
                                             destination);
}
template <typename LhsTile, typename RhsTile>
__attribute__((noinline)) inline void
matmulBlockAtAddresses(uint32_t lhs, uint32_t rhs, uint32_t lhsPageWords,
                       uint32_t rhsPageWords, uint32_t destination,
                       uint32_t transpose, uint32_t columns, uint32_t rows,
                       uint32_t inner) {
  UNPACK((_llk_unpack_AB_matmul_(lhs, rhs, 0, 0, lhsPageWords, rhsPageWords,
                                 target::hasPartialFace(RhsTile::tensorShape),
                                 target::hasPartialFace(LhsTile::tensorShape),
                                 columns, rows, inner)));
  executeMatmul<LhsTile, RhsTile>(destination, transpose, columns, rows);
}
template <typename Lhs, typename Rhs>
inline void matmul_block(Lhs lhs, Rhs rhs, uint32_t lhsTile, uint32_t rhsTile,
                         uint32_t destination, uint32_t transpose,
                         uint32_t columns, uint32_t rows, uint32_t inner) {
  UNPACK((matmulBlockAtAddresses<typename Lhs::TileMetadata,
                                 typename Rhs::TileMetadata>(
      lhs.readTile(lhsTile), rhs.readTile(rhsTile), Lhs::pageWords,
      Rhs::pageWords, destination, transpose, columns, rows, inner)));
  MATH((matmulBlockAtAddresses<typename Lhs::TileMetadata,
                               typename Rhs::TileMetadata>(
      0, 0, 0, 0, destination, transpose, columns, rows, inner)));
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
    matmulInitShape<typename Lhs::TileMetadata, typename Rhs::TileMetadata>(
        transpose, 1, 1, 1);
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
template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension,
          typename InputTile>
__attribute__((noinline)) inline void
reduceAtAddresses(uint32_t input, uint32_t scaler, uint32_t destination) {
  UNPACK((_llk_unpack_AB_reduce_<Pool, Dimension>(input, scaler)));
  MATH((_llk_math_reduce_<Pool, Dimension, DST_ACCUM_MODE, MATH_FIDELITY>(
      destination, InputTile::tensorShape)));
}
template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension, typename Input,
          typename Scaler>
inline void reduce_tile(Input input, Scaler scaler, uint32_t inputTile,
                        uint32_t scalerTile, uint32_t destination) {
  UNPACK((reduceAtAddresses<Pool, Dimension, typename Input::TileMetadata>(
      input.readTile(inputTile), scaler.readTile(scalerTile), destination)));
  MATH((reduceAtAddresses<Pool, Dimension, typename Input::TileMetadata>(
      0, 0, destination)));
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
