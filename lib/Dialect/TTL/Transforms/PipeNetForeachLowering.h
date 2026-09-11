// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/PipeRecordLoweringUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace mlir::tt::ttl {

struct PipeForeachLoweringInfo;

/// Immutable lowering inputs for one graph mapping. `records` contains only
/// this mapping; identity mappings derive node coordinates without tables.
struct GraphPipeMappingForeachPlan {
  PipeNetRecordsAttr records;
  PipeRecordTables nodePipeTables;
  std::unique_ptr<TransferGraph> graph;
  int64_t nodePipeCount = 0;
  int64_t launchGridX = 0;
  bool usesLaunchGridIdentity = false;
};

using GraphPipeMappingForeachPlans =
    SmallVector<GraphPipeMappingForeachPlan, 0>;
using GraphPipeNetForeachPlans = llvm::DenseMap<
    PipeNetRecordsAttr,
    llvm::DenseMap<std::pair<int64_t, int64_t>, GraphPipeMappingForeachPlans>>;

/// Validate and plan every graph PipeNet before callback IR is rewritten.
FailureOr<GraphPipeNetForeachPlans>
buildGraphPipeNetForeachPlans(ModuleOp module);

/// Lower all PipeNet callbacks using the previously validated graph plans.
LogicalResult
lowerPipeNetForeachOps(ModuleOp module,
                       PipeForeachLoweringInfo &foreachLoweringInfo,
                       const GraphPipeNetForeachPlans &plansByRecordsAndGrid);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETFOREACHLOWERING_H
