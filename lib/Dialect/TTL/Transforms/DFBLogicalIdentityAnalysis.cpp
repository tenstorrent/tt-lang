// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

namespace mlir::tt::ttl {

DFBLogicalIdentityAnalysis::DFBLogicalIdentityAnalysis(Operation *operation) {
  ModuleOp moduleOp = cast<ModuleOp>(operation);
  int64_t maxExplicitId = -1;
  int64_t compilerDeclarationCount = 0;
  BindCBOp firstCompilerDeclaration;

  // Discover the complete explicit ID range before generating IDs. Assigning
  // generated IDs during this walk could collide with a later declaration.
  WalkResult discoveryResult = moduleOp.walk([&](BindCBOp bindOp) {
    if (auto dfbId = bindOp.getDfbId()) {
      int64_t logicalId = dfbId->getSExtValue();
      maxExplicitId = std::max(maxExplicitId, logicalId);
      return WalkResult::advance();
    }
    if (!bindOp->hasAttr(kCompilerAllocatedAttrName)) {
      errorOperation = bindOp;
      errorMessage =
          "user-declared DFB requires dfb_id before physical allocation";
      return WalkResult::interrupt();
    }
    if (!firstCompilerDeclaration) {
      firstCompilerDeclaration = bindOp;
    }
    ++compilerDeclarationCount;
    return WalkResult::advance();
  });
  if (discoveryResult.wasInterrupted()) {
    return;
  }

  if (compilerDeclarationCount > 0 &&
      maxExplicitId >
          std::numeric_limits<int64_t>::max() - compilerDeclarationCount) {
    errorOperation = firstCompilerDeclaration;
    errorMessage =
        "logical DFB identifiers leave no space for compiler-created DFBs";
    return;
  }

  int64_t nextCompilerId = compilerDeclarationCount > 0 ? maxExplicitId + 1 : 0;
  llvm::DenseMap<int64_t, BindCBOp> firstDeclarationById;
  // Declarations with one logical ID describe one allocation, so they must
  // agree on the complete DFB type in every participating kernel.
  moduleOp.walk([&](BindCBOp bindOp) {
    auto dfbId = bindOp.getDfbId();
    int64_t logicalId = dfbId ? dfbId->getSExtValue() : nextCompilerId++;
    auto [firstDeclarationIt, inserted] =
        firstDeclarationById.try_emplace(logicalId, bindOp);
    if (!inserted && firstDeclarationIt->second.getResult().getType() !=
                         bindOp.getResult().getType()) {
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream
          << "logical DFB " << logicalId
          << " has inconsistent types across kernel functions: expected "
          << firstDeclarationIt->second.getResult().getType() << " but found "
          << bindOp.getResult().getType();
      errorOperation = bindOp;
      errorMessage = messageStream.str();
      return WalkResult::interrupt();
    }
    assignments.push_back({bindOp, logicalId});
    logicalIds[bindOp.getOperation()] = logicalId;
    return WalkResult::advance();
  });
}

int64_t DFBLogicalIdentityAnalysis::getLogicalId(BindCBOp declaration) const {
  auto logicalId = logicalIds.find(declaration.getOperation());
  assert(logicalId != logicalIds.end() &&
         "every DFB declaration must have a resolved logical identity");
  return logicalId->second;
}

FailureOr<int64_t> DFBLogicalIdentityAnalysis::getLogicalId(Value dfb) const {
  BindCBOp declaration = getDFBDeclaration(dfb);
  if (!declaration) {
    return failure();
  }
  auto logicalId = logicalIds.find(declaration.getOperation());
  if (logicalId == logicalIds.end()) {
    return failure();
  }
  return logicalId->second;
}

} // namespace mlir::tt::ttl
