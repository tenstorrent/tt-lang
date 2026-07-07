// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_C_TTATTRS_H
#define TTLANG_C_TTATTRS_H

#include "mlir-c/AffineMap.h"
#include "ttlang-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTGridAttrGet(MlirContext ctx,
                                                     int64_t *shape,
                                                     size_t shapeSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttlangTTChipCapabilityAttrGet(MlirContext ctx, uint32_t chipCapability);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTArchAttrGet(MlirContext ctx,
                                                     uint32_t arch);

MLIR_CAPI_EXPORTED MlirAttribute
ttlangTTDataTypeAttrGet(MlirContext ctx, uint16_t *supportedDataTypes);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTChipDescAttrGet(
    MlirContext ctx, MlirAttribute arch, int64_t *grid, size_t gridSize,
    int64_t *coordTranslationOffsets, size_t coordTranslationOffsetsSize,
    unsigned l1Size, unsigned numDramChannels, unsigned dramChannelSize,
    unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
    unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
    unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
    unsigned dramUnreservedEnd, MlirAttribute *supportedDataTypes,
    MlirAttribute *supportedTileSizes, unsigned dstPhysicalSizeTiles,
    unsigned numCBs, unsigned numComputeThreads,
    unsigned numDatamovementThreads, int64_t *dramGrid, size_t dramGridSize,
    MlirAttribute *dramBankToLogicalWorkerNoc0,
    MlirAttribute *dramBankToLogicalWorkerNoc1);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTChipCoordAttrGet(
    MlirContext ctx, unsigned rack, unsigned shelf, unsigned y, unsigned x);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTChipChannelAttrGet(
    MlirContext ctx, unsigned deviceId0, int64_t *ethernetCoreCoord0,
    size_t ethernetCoreCoord0Size, unsigned deviceId1,
    int64_t *ethernetCoreCoord1, size_t ethernetCoreCoord1Size);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTSystemDescAttrGet(
    MlirContext ctx, MlirAttribute *cpuDescs, size_t cpuDescsSize,
    MlirAttribute *chipDescs, size_t chipDescsSize, unsigned *chipDescIndices,
    size_t chipDescIndicesSize, MlirAttribute *chipCapabilities,
    size_t chipCapabilitiesSize, MlirAttribute *chipCoords,
    size_t chipCoordsSize, MlirAttribute *chipChannels,
    size_t chipChannelsSize);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTMetalLayoutAttrGet(
    MlirContext ctx, intptr_t logicalRank, const int64_t *logicalShape,
    intptr_t gridRank, const int64_t *gridShape, MlirType elementType,
    intptr_t tileRank, const int64_t *tileShape, unsigned oobVal,
    unsigned memorySpace);

MLIR_CAPI_EXPORTED MlirAttribute
ttlangTTMemorySpaceAttrGet(MlirContext ctx, uint32_t memorySpace);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTOOBValAttrGet(MlirContext ctx,
                                                       uint32_t oobVal);

MLIR_CAPI_EXPORTED MlirAttribute
ttlangTTIteratorTypeAttrGet(MlirContext ctx, uint32_t iteratorType);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTIteratorTypeArrayAttrGet(
    MlirContext ctx, uint32_t *iteratorTypes, size_t iteratorTypesSize);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTTileSizeAttrGet(MlirContext ctx,
                                                         int64_t y, int64_t x);

MLIR_CAPI_EXPORTED MlirAttribute ttlangTTCoreCoordAttrGet(MlirContext ctx,
                                                          int64_t y, int64_t x);

#ifdef __cplusplus
}
#endif

#endif // TTLANG_C_TTATTRS_H
