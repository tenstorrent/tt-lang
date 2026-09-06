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
template <uint32_t Format, uint32_t PageBytes, uint32_t PagesPerBlock,
          uint32_t BlockCount, uint32_t PayloadOffset, bool DirectToDestination>
class Operand
    : public Buffer<PageBytes, PagesPerBlock, BlockCount, PayloadOffset> {
public:
  using Buffer<PageBytes, PagesPerBlock, BlockCount, PayloadOffset>::Buffer;
  static constexpr uint32_t format = Format;
  static constexpr bool directToDestination = DirectToDestination;
  static constexpr uint32_t unpackFormat =
      Format == static_cast<uint32_t>(DataFormat::Float32) &&
              !DirectToDestination
          ? static_cast<uint32_t>(DataFormat::Tf32)
          : Format;
  static constexpr uint32_t pageWords = PageBytes / 16;
  uint32_t readTile(uint32_t tile) const {
    return this->get_read_ptr() / 16 + tile * pageWords - 1;
  }
  uint32_t writeTile(uint32_t tile) const {
    return this->get_write_ptr() / 16 + tile * pageWords - 1;
  }
};

namespace target {
template <typename Source>
inline void copy_tile_init(Source source);
__attribute__((noinline)) inline void matmulInitShape(uint32_t transpose,
                                                      uint32_t columns,
                                                      uint32_t rows,
                                                      uint32_t inner) {
  UNPACK((_llk_unpack_AB_matmul_init_(transpose, columns, rows, inner, 16, 16,
                                      4, 4, false, false)));
  MATH((_llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE>(
      32, 32, 32, 32, false, transpose, columns, rows)));
#if defined(ARCH_BLACKHOLE)
  MATH((ckernel::throttled_mop_status = 0));
#endif
}
template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension>
__attribute__((noinline)) inline void reduceInitShape() {
  UNPACK((_llk_unpack_AB_reduce_init_<Pool, Dimension>(
      ckernel::DEFAULT_TENSOR_SHAPE)));
  MATH((_llk_math_reduce_init_<Pool, Dimension, DST_ACCUM_MODE, MATH_FIDELITY>(
      ckernel::DEFAULT_TENSOR_SHAPE)));
  PACK((_llk_pack_reduce_mask_config_<Dimension, ckernel::PackMode::Default>(
      16)));
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
      0, 0, ckernel::DEFAULT_TENSOR_SHAPE, Source::format,
      Source::unpackFormat)));
#if defined(ARCH_BLACKHOLE)
  MATH((_llk_math_eltwise_unary_datacopy_init_<
        copyType, DST_ACCUM_MODE, Broadcast>(4, Source::unpackFormat, false)));
#else
  MATH((_llk_math_eltwise_unary_datacopy_init_<
        copyType, DST_ACCUM_MODE, Broadcast>(4, Source::unpackFormat)));
#endif
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
  uint32_t sourceADestinationFormat = 0;
  uint32_t sourceBFormat = 0;
  uint32_t sourceBDestinationFormat = 0;
  uint32_t outputFormat = 0;

public:
  template <uint32_t SourceAFormat, uint32_t SourceAUnpackFormat,
            uint32_t SourceAPageWords, uint32_t SourceBFormat,
            uint32_t SourceBUnpackFormat, uint32_t SourceBPageWords,
            uint32_t OutputFormat, uint32_t OutputPageWords>
  __attribute__((noinline)) void configureFormats() {
    if (!initialized) {
      UNPACK((_llk_unpack_hw_configure_<DST_ACCUM_MODE>(
          SourceAFormat, SourceBFormat, SourceAUnpackFormat,
          SourceBUnpackFormat, 16, 16, 4, 4, SourceAPageWords,
          SourceBPageWords)));
      MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
      MATH((_llk_math_hw_configure_<DST_ACCUM_MODE>(SourceAUnpackFormat,
                                                    SourceBUnpackFormat)));
#if defined(ARCH_BLACKHOLE)
      PACK((_llk_pack_hw_configure_<DST_ACCUM_MODE, ckernel::PackMode::Default>(
          OutputFormat, OutputFormat, OutputPageWords, 16, 32, 4, false, 0)));
      PACK((_llk_pack_init_<ckernel::PackMode::Default>(OutputFormat, 16, 32, 4,
                                                        1, false)));
      PACK((_llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE>()));
#else
      PACK((_llk_pack_hw_configure_<DST_ACCUM_MODE, ckernel::PackMode::Default>(
          OutputFormat, OutputFormat, OutputPageWords, 16, 4, false, false,
          0)));
      PACK((_llk_pack_init_<ckernel::PackMode::Default>(OutputFormat, 16, 4,
                                                        false, false, 1)));
      PACK((_llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE,
                                 ckernel::PackMode::Default>(16, false)));
#endif
      initialized = true;
    } else {
      configureInputFormats<SourceAFormat, SourceAUnpackFormat,
                            SourceAPageWords, SourceBFormat,
                            SourceBUnpackFormat, SourceBPageWords>();
      if (outputFormat != OutputFormat) {
#if defined(ARCH_BLACKHOLE)
        PACK((_llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
            OutputFormat, OutputFormat, OutputPageWords, 32, 4, false)));
#else
        PACK((_llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
            OutputFormat, OutputFormat, OutputPageWords, 16, 4, false, false)));
#endif
      }
    }
    sourceAFormat = SourceAFormat;
    sourceADestinationFormat = SourceAUnpackFormat;
    sourceBFormat = SourceBFormat;
    sourceBDestinationFormat = SourceBUnpackFormat;
    outputFormat = OutputFormat;
  }
  template <typename SourceA, typename SourceB, typename Output>
  void configure(SourceA, SourceB, Output) {
    configureFormats<SourceA::format, SourceA::unpackFormat, SourceA::pageWords,
                     SourceB::format, SourceB::unpackFormat, SourceB::pageWords,
                     Output::format, Output::pageWords>();
  }
  template <uint32_t SourceAFormat, uint32_t SourceAUnpackFormat,
            uint32_t SourceAPageWords, uint32_t SourceBFormat,
            uint32_t SourceBUnpackFormat, uint32_t SourceBPageWords>
  __attribute__((noinline)) void configureInputFormats() {
    if (sourceAFormat != SourceAFormat ||
        sourceADestinationFormat != SourceAUnpackFormat) {
      UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<
              DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
          SourceAFormat, SourceAUnpackFormat, SourceAPageWords, 16, 4)));
      MATH((_llk_math_reconfig_data_format_srca_<DST_ACCUM_MODE>(
          SourceAUnpackFormat)));
    }
    if (sourceBFormat != SourceBFormat ||
        sourceBDestinationFormat != SourceBUnpackFormat) {
      UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<
              DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
          SourceBFormat, SourceBUnpackFormat, SourceBPageWords, 16, 4)));
      MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE>(
          SourceBUnpackFormat)));
    }
    sourceAFormat = SourceAFormat;
    sourceADestinationFormat = SourceAUnpackFormat;
    sourceBFormat = SourceBFormat;
    sourceBDestinationFormat = SourceBUnpackFormat;
  }
  template <typename Lhs, typename Rhs, typename Output>
  void matmulInit(Lhs lhs, Rhs rhs, Output output, uint32_t transpose) {
    matmulBlockInit(lhs, rhs, output, transpose, 1, 1, 1);
  }
  template <typename Lhs, typename Rhs, typename Output>
  void matmulBlockInit(Lhs lhs, Rhs rhs, Output output, uint32_t transpose,
                       uint32_t columns, uint32_t rows, uint32_t inner) {
    configure(rhs, lhs, output);
    matmulInitShape(transpose, columns, rows, inner);
  }
  template <typename Lhs, typename Rhs>
  void matmulInitShort(Lhs lhs, Rhs rhs, uint32_t transpose) {
    matmulBlockInitShort(lhs, rhs, transpose, 1, 1, 1);
  }
  template <typename Lhs, typename Rhs>
  void matmulBlockInitShort(Lhs, Rhs, uint32_t transpose, uint32_t columns,
                            uint32_t rows, uint32_t inner) {
    configureInputFormats<Rhs::format, Rhs::unpackFormat, Rhs::pageWords,
                          Lhs::format, Lhs::unpackFormat, Lhs::pageWords>();
    matmulInitShape(transpose, columns, rows, inner);
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
    reduceInitShape<Pool, Dimension>();
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
        true, !Source::directToDestination, ckernel::DEFAULT_TENSOR_SHAPE,
        Source::format, Source::unpackFormat)));
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

template <uint32_t Format, uint32_t UnpackFormat, bool Direct>
__attribute__((noinline)) inline void copyInitFormats() {
  UNPACK(
      (_llk_unpack_A_init_<ckernel::BroadcastType::NONE, false,
                           ckernel::EltwiseBinaryReuseDestType::NONE, Direct>(
          0, 0, ckernel::DEFAULT_TENSOR_SHAPE, Format, UnpackFormat)));
#if defined(ARCH_BLACKHOLE)
  MATH((_llk_math_eltwise_unary_datacopy_init_<ckernel::DataCopyType::A2D,
                                               DST_ACCUM_MODE,
                                               ckernel::BroadcastType::NONE>(
      4, UnpackFormat, false)));
#else
  MATH((_llk_math_eltwise_unary_datacopy_init_<ckernel::DataCopyType::A2D,
                                               DST_ACCUM_MODE,
                                               ckernel::BroadcastType::NONE>(
      4, UnpackFormat)));
#endif
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
  copyInitFormats<Source::format, Source::unpackFormat,
                  Source::directToDestination>();
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
      0, 0, ckernel::DEFAULT_TENSOR_SHAPE, Source::format,
      Source::unpackFormat)));
  MATH((_llk_math_eltwise_binary_init_<Operation, ckernel::BroadcastType::NONE,
                                       MATH_FIDELITY, Reuse>(
      ckernel::DEFAULT_TENSOR_SHAPE, false)));
}
template <ckernel::EltwiseBinaryType Operation,
          ckernel::EltwiseBinaryReuseDestType Reuse, typename Source>
inline void binary_dest_reuse_tiles(Source source, uint32_t tile,
                                    uint32_t destination) {
  UNPACK((_llk_unpack_A_<ckernel::BroadcastType::NONE, true, Reuse>(
      source.readTile(tile), Source::format, Source::unpackFormat)));
  MATH((llk_math_eltwise_binary<Operation, ckernel::BroadcastType::NONE,
                                DST_ACCUM_MODE, MATH_FIDELITY, Reuse>(
      destination, true)));
}
template <ckernel::EltwiseBinaryType Operation>
__attribute__((noinline)) inline void binaryInit() {
  UNPACK((_llk_unpack_AB_init_<ckernel::BroadcastType::NONE>(
      ckernel::DEFAULT_TENSOR_SHAPE, ckernel::Transpose::None)));
  MATH((_llk_math_eltwise_binary_init_<
        Operation, ckernel::BroadcastType::NONE,
        get_effective_math_fidelity<Operation, MATH_FIDELITY>()>(
      ckernel::DEFAULT_TENSOR_SHAPE, false)));
}
// Sharing instruction emission bounds code size independently of storage
// identities.
template <ckernel::EltwiseBinaryType Operation>
__attribute__((noinline)) inline void
binaryAtAddresses(uint32_t sourceA, uint32_t sourceB, uint32_t destination) {
  UNPACK((_llk_unpack_AB_<ckernel::BroadcastType::NONE>(sourceA, sourceB)));
  MATH((llk_math_eltwise_binary<Operation, ckernel::BroadcastType::NONE,
                                DST_ACCUM_MODE, MATH_FIDELITY>(destination)));
}
template <ckernel::EltwiseBinaryType Operation, typename SourceA,
          typename SourceB>
inline void binary(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                   uint32_t tileB, uint32_t destination) {
  UNPACK((binaryAtAddresses<Operation>(sourceA.readTile(tileA),
                                       sourceB.readTile(tileB), destination)));
  MATH((binaryAtAddresses<Operation>(0, 0, destination)));
}
template <typename SourceA, typename SourceB>
inline void add_tiles_init(SourceA sourceA, SourceB sourceB) {
  binaryInit<ckernel::EltwiseBinaryType::ELWADD>();
}
template <typename SourceA, typename SourceB>
inline void add_tiles(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                      uint32_t tileB, uint32_t destination) {
  binary<ckernel::EltwiseBinaryType::ELWADD>(sourceA, sourceB, tileA, tileB,
                                             destination);
}
template <typename SourceA, typename SourceB>
inline void sub_tiles_init(SourceA sourceA, SourceB sourceB) {
  binaryInit<ckernel::EltwiseBinaryType::ELWSUB>();
}
template <typename SourceA, typename SourceB>
inline void sub_tiles(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                      uint32_t tileB, uint32_t destination) {
  binary<ckernel::EltwiseBinaryType::ELWSUB>(sourceA, sourceB, tileA, tileB,
                                             destination);
}
template <typename SourceA, typename SourceB>
inline void mul_tiles_init(SourceA sourceA, SourceB sourceB) {
  binaryInit<ckernel::EltwiseBinaryType::ELWMUL>();
}
template <typename SourceA, typename SourceB>
inline void mul_tiles(SourceA sourceA, SourceB sourceB, uint32_t tileA,
                      uint32_t tileB, uint32_t destination) {
  binary<ckernel::EltwiseBinaryType::ELWMUL>(sourceA, sourceB, tileA, tileB,
                                             destination);
}
__attribute__((noinline)) inline void
matmulBlockAtAddresses(uint32_t lhs, uint32_t rhs, uint32_t lhsPageWords,
                       uint32_t rhsPageWords, uint32_t destination,
                       uint32_t transpose, uint32_t columns, uint32_t rows,
                       uint32_t inner) {
  UNPACK((_llk_unpack_AB_matmul_(lhs, rhs, 0, 0, lhsPageWords, rhsPageWords,
                                 false, false, columns, rows, inner)));
#if defined(ARCH_BLACKHOLE) && defined(TRISC_MATH)
  bool throttled = (*ckernel::throttle_ptr % 2) != 0;
  if (throttled) {
    if (ckernel::throttled_mop_status != 1) {
      _llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE_MAX>(
          32, 32, 32, 32, false, transpose, columns, rows);
      ckernel::throttled_mop_status = 1;
    }
    llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(destination, columns, rows);
  } else {
    if (ckernel::throttled_mop_status != 0) {
      _llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE>(
          32, 32, 32, 32, false, transpose, columns, rows);
      ckernel::throttled_mop_status = 0;
    }
    llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(destination, columns, rows);
  }
#elif defined(TRISC_MATH)
  llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(destination, columns, rows);
#endif
}
template <typename Lhs, typename Rhs>
inline void matmul_block(Lhs lhs, Rhs rhs, uint32_t lhsTile, uint32_t rhsTile,
                         uint32_t destination, uint32_t transpose,
                         uint32_t columns, uint32_t rows, uint32_t inner) {
  UNPACK((matmulBlockAtAddresses(lhs.readTile(lhsTile), rhs.readTile(rhsTile),
                                 Lhs::pageWords, Rhs::pageWords, destination,
                                 transpose, columns, rows, inner)));
  MATH((matmulBlockAtAddresses(0, 0, 0, 0, destination, transpose, columns,
                               rows, inner)));
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
    matmulInitShape(transpose, 1, 1, 1);
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
template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension>
__attribute__((noinline)) inline void
reduceAtAddresses(uint32_t input, uint32_t scaler, uint32_t destination) {
  UNPACK((_llk_unpack_AB_reduce_<Pool, Dimension>(input, scaler)));
  MATH((_llk_math_reduce_<Pool, Dimension, DST_ACCUM_MODE, MATH_FIDELITY>(
      destination, ckernel::DEFAULT_TENSOR_SHAPE)));
}
template <ckernel::PoolType Pool, ckernel::ReduceDim Dimension, typename Input,
          typename Scaler>
inline void reduce_tile(Input input, Scaler scaler, uint32_t inputTile,
                        uint32_t scalerTile, uint32_t destination) {
  UNPACK((reduceAtAddresses<Pool, Dimension>(
      input.readTile(inputTile), scaler.readTile(scalerTile), destination)));
  MATH((reduceAtAddresses<Pool, Dimension>(0, 0, destination)));
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
} // namespace target
} // namespace ttlang::l1
#endif
