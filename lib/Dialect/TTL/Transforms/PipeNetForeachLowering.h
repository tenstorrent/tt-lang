// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/PipeRecordLoweringUtils.h"
#include "llvm/ADT/MapVector.h"

#include <cstdint>
#include <memory>

namespace mlir::tt::ttl {

struct PipeForeachLoweringInfo;

struct GraphPipeMappingForeachPlan {
  PipeNetRecordsAttr records;
  PipeRecordTables nodePipeTables;
  std::unique_ptr<TransferGraph> graph;
  int64_t nodePipeCount = 0;
};

using GraphPipeNetForeachPlans =
    llvm::MapVector<PipeNetRecordsAttr,
                    SmallVector<GraphPipeMappingForeachPlan, 0>>;

FailureOr<GraphPipeNetForeachPlans>
buildGraphPipeNetForeachPlans(ModuleOp module);

LogicalResult
lowerPipeNetForeachOps(ModuleOp module,
                       PipeForeachLoweringInfo &foreachLoweringInfo,
                       const GraphPipeNetForeachPlans &plansByRecords);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H
