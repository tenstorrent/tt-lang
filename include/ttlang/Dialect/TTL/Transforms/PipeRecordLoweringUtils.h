// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPERECORDLOWERINGUTILS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPERECORDLOWERINGUTILS_H

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::tt::ttl {

/// Constant tables aligned with one `PipeNetRecordsAttr` record order.
struct PipeRecordTables {
  SmallVector<int64_t> srcX;
  SmallVector<int64_t> srcY;
  SmallVector<int64_t> dstStartX;
  SmallVector<int64_t> dstStartY;
  SmallVector<int64_t> dstEndX;
  SmallVector<int64_t> dstEndY;
  SmallVector<int64_t> numDests;
  SmallVector<int64_t> srcInDstRange;
  SmallVector<int64_t> sourceDeviceIndex;
  SmallVector<int64_t> destinationDeviceIndex;
};

inline void appendPipeRecordTableEntry(PipeRecordTables &tables,
                                       PipeRecordAttr record) {
  PipeType pipeType =
      getPipeTypeFromRecord(record.getContext(), record, /*pipeNetId=*/0);
  tables.srcX.push_back(pipeType.getSrcX());
  tables.srcY.push_back(pipeType.getSrcY());
  tables.dstStartX.push_back(pipeType.getDstStartX());
  tables.dstStartY.push_back(pipeType.getDstStartY());
  tables.dstEndX.push_back(pipeType.getDstEndX());
  tables.dstEndY.push_back(pipeType.getDstEndY());
  tables.numDests.push_back(pipeType.getNumDests());
  tables.srcInDstRange.push_back(pipeType.srcInDstRange() ? 1 : 0);
  DeviceTransferAttr transfer = record.getDeviceTransfer();
  // Device-index accessors reject local record tables, so zero is an
  // unobservable placeholder that keeps intermediate lookup IR valid.
  tables.sourceDeviceIndex.push_back(
      transfer ? getLogicalDeviceIndex(transfer.getDomain(),
                                       transfer.getEdge().getSource())
               : 0);
  tables.destinationDeviceIndex.push_back(
      transfer ? getLogicalDeviceIndex(transfer.getDomain(),
                                       transfer.getEdge().getDestination())
               : 0);
}

inline PipeRecordTables
buildPipeRecordTables(ArrayRef<PipeRecordAttr> records) {
  PipeRecordTables tables;
  assert(!records.empty() && "pipe record table must not be empty");
  for (PipeRecordAttr record : records) {
    appendPipeRecordTableEntry(tables, record);
  }
  return tables;
}

inline PipeRecordTables buildPipeRecordTables(PipeNetRecordsAttr records) {
  PipeRecordTables tables;
  forEachPipeRecord(records, [&](std::uint64_t, PipeRecordAttr record) {
    appendPipeRecordTableEntry(tables, record);
  });
  return tables;
}

/// Return whether `records` contains one `(x, y) -> (x, y)` pipe per launch
/// node in row-major order.
inline bool isLaunchGridIdentityPipeMapping(ArrayRef<PipeRecordAttr> records,
                                            int64_t gridX, int64_t gridY) {
  std::optional<int64_t> gridArea = llvm::checkedMul(gridX, gridY);
  if (!gridArea || *gridArea <= 0 ||
      records.size() != static_cast<std::size_t>(*gridArea)) {
    return false;
  }
  return llvm::all_of(llvm::enumerate(records), [&](auto indexedRecord) {
    int64_t recordIndex = static_cast<int64_t>(indexedRecord.index());
    PipeRecordAttr record = indexedRecord.value();
    int64_t expectedX = recordIndex % gridX;
    int64_t expectedY = recordIndex / gridX;
    return !record.getDeviceTransfer() && !record.getIsCollective() &&
           record.getSrcX() == expectedX && record.getSrcY() == expectedY &&
           record.getDstStartX() == expectedX &&
           record.getDstStartY() == expectedY &&
           record.getDstEndX() == expectedX && record.getDstEndY() == expectedY;
  });
}

/// Build an index-valued lookup into immutable table data.
inline Value buildConstantIndexTableLookup(OpBuilder &builder, Location loc,
                                           ArrayRef<int64_t> values,
                                           Value index) {
  assert(!values.empty() && "constant table must not be empty");
  return ttkernel::ConstantTableLookupOp::create(
      builder, loc, builder.getIndexType(), index,
      builder.getDenseI64ArrayAttr(values));
}

/// Return whether a launch node equals a point coordinate.
inline Value buildNodePointMatch(OpBuilder &builder, Location loc, Value nodeX,
                                 Value nodeY, Value pointX, Value pointY) {
  Value xMatches = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         nodeX, pointX);
  Value yMatches = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         nodeY, pointY);
  return arith::AndIOp::create(builder, loc, xMatches, yMatches);
}

/// Return whether a launch node belongs to an inclusive coordinate range.
inline Value buildNodeRangeMatch(OpBuilder &builder, Location loc, Value nodeX,
                                 Value nodeY, Value minX, Value minY,
                                 Value maxX, Value maxY) {
  Value geMinX = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                       nodeX, minX);
  Value leMaxX = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle,
                                       nodeX, maxX);
  Value geMinY = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                       nodeY, minY);
  Value leMaxY = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle,
                                       nodeY, maxY);
  Value inRangeX = arith::AndIOp::create(builder, loc, geMinX, leMaxX);
  Value inRangeY = arith::AndIOp::create(builder, loc, geMinY, leMaxY);
  return arith::AndIOp::create(builder, loc, inRangeX, inRangeY);
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPERECORDLOWERINGUTILS_H
