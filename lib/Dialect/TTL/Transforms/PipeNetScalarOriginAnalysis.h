// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETSCALARORIGINANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETSCALARORIGINANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include <functional>
#include <map>
#include <optional>

namespace mlir::tt::ttl {

/// Source endpoint of one proven exact-one PipeNet transport.
struct PipeNetScalarTransportSource {
  Value dfb;
  LaunchExecutionLocation location;
  Operation *consumer = nullptr;
};

/// Resolves scalar DFB reads to their original tensor elements across proven
/// exact-one PipeNet transports.
class PipeNetScalarOriginAnalysis {
public:
  using LogicalDFBResolver = std::function<std::optional<int64_t>(Value)>;
  using OperationCountResolver = std::function<std::optional<std::uint64_t>(
      Operation *, const LaunchExecutionLocation &)>;
  using TransportSourceResolver =
      std::function<std::optional<PipeNetScalarTransportSource>(
          Operation *, const LaunchExecutionLocation &, CBPushOp)>;

  PipeNetScalarOriginAnalysis(ModuleOp module,
                              const LaunchNodeDomainState &launchDomains,
                              LogicalDFBResolver logicalDFBResolver,
                              OperationCountResolver operationCountResolver,
                              TransportSourceResolver transportSourceResolver)
      : launchDomains(launchDomains),
        logicalDFBResolver(std::move(logicalDFBResolver)),
        operationCountResolver(std::move(operationCountResolver)),
        transportSourceResolver(std::move(transportSourceResolver)) {
    module.walk([&](CBPushOp push) {
      if (std::optional<int64_t> logicalId =
              this->logicalDFBResolver(push.getCb())) {
        pushesByLogicalId[*logicalId].push_back(push);
      }
    });
  }

  bool proveEqual(Value lhs, const LaunchExecutionLocation &lhsLocation,
                  Value rhs, const LaunchExecutionLocation &rhsLocation) const {
    auto lhsRead = lhs.getDefiningOp<ReadIndexOp>();
    auto rhsRead = rhs.getDefiningOp<ReadIndexOp>();
    if (!lhsRead || !rhsRead) {
      return false;
    }
    std::optional<TensorScalarOrigin> lhsOrigin =
        traceRead(lhsRead, lhsLocation);
    std::optional<TensorScalarOrigin> rhsOrigin =
        traceRead(rhsRead, rhsLocation);
    return lhsOrigin && rhsOrigin &&
           haveSameTensorOrigin(*lhsOrigin, *rhsOrigin) &&
           lhsOrigin->tileIndices == rhsOrigin->tileIndices &&
           lhsOrigin->scalarCoords == rhsOrigin->scalarCoords &&
           lhsOrigin->location == rhsOrigin->location;
  }

private:
  struct TensorScalarOrigin {
    Value tensor;
    std::optional<int64_t> globalTensorIndex;
    SmallVector<int64_t> tileIndices;
    SmallVector<int64_t> scalarCoords;
    LaunchExecutionLocation location;
  };

  static std::optional<int64_t> getGlobalTensorIndex(Value tensor) {
    auto argument = dyn_cast<BlockArgument>(tensor);
    if (!argument) {
      return std::nullopt;
    }
    auto function =
        dyn_cast_if_present<func::FuncOp>(argument.getOwner()->getParentOp());
    if (!function || argument.getOwner() != &function.getBody().front() ||
        !function->hasAttr(kKernelThreadAttrName)) {
      return std::nullopt;
    }
    auto indices = function->getAttrOfType<ArrayAttr>(kCRTAIndicesAttrName);
    if (!indices || argument.getArgNumber() >= indices.size()) {
      return std::nullopt;
    }
    auto index = dyn_cast<IntegerAttr>(indices[argument.getArgNumber()]);
    if (!index || index.getInt() < 0) {
      return std::nullopt;
    }
    return index.getInt();
  }

  static bool haveSameTensorOrigin(const TensorScalarOrigin &lhs,
                                   const TensorScalarOrigin &rhs) {
    if (lhs.globalTensorIndex || rhs.globalTensorIndex) {
      return lhs.globalTensorIndex && rhs.globalTensorIndex &&
             lhs.globalTensorIndex == rhs.globalTensorIndex &&
             lhs.tensor.getType() == rhs.tensor.getType();
    }
    return lhs.tensor == rhs.tensor;
  }

  static bool structurallyPrecedes(Operation *before, Operation *after) {
    if (before == after || before->getParentOfType<func::FuncOp>() !=
                               after->getParentOfType<func::FuncOp>()) {
      return false;
    }
    for (Block *commonBlock = before->getBlock(); commonBlock;) {
      Operation *projectedBefore =
          before->getBlock() == commonBlock
              ? before
              : commonBlock->findAncestorOpInBlock(*before);
      Operation *projectedAfter =
          after->getBlock() == commonBlock
              ? after
              : commonBlock->findAncestorOpInBlock(*after);
      if (projectedBefore && projectedAfter &&
          projectedBefore != projectedAfter) {
        return projectedBefore->isBeforeInBlock(projectedAfter);
      }
      Operation *parent = commonBlock->getParentOp();
      commonBlock = parent ? parent->getBlock() : nullptr;
    }
    return false;
  }

  std::optional<SmallVector<int64_t>>
  evaluateCoords(ValueRange coords,
                 const LaunchExecutionLocation &location) const {
    SmallVector<int64_t> result;
    result.reserve(coords.size());
    for (Value coord : coords) {
      std::optional<llvm::APInt> value =
          evaluateIntegerAtLaunchLocation(coord, location, launchDomains);
      if (!value || value->isNegative() || value->getActiveBits() > 63) {
        return std::nullopt;
      }
      result.push_back(value->getSExtValue());
    }
    return result;
  }

  std::optional<CBPushOp>
  findUniquePush(int64_t logicalId, const LaunchExecutionLocation &location,
                 Operation *consumer) const {
    func::FuncOp consumerFunction = consumer->getParentOfType<func::FuncOp>();
    auto pushes = pushesByLogicalId.find(logicalId);
    if (pushes == pushesByLogicalId.end()) {
      return std::nullopt;
    }
    std::optional<CBPushOp> result;
    bool invalid = false;
    for (CBPushOp push : pushes->second) {
      func::FuncOp producerFunction = push->getParentOfType<func::FuncOp>();
      if (producerFunction == consumerFunction &&
          !structurallyPrecedes(push, consumer)) {
        continue;
      }
      std::optional<std::uint64_t> count =
          operationCountResolver(push, location);
      if (!count) {
        invalid = true;
        continue;
      }
      if (*count == 0) {
        continue;
      }
      if (*count != 1 || result) {
        invalid = true;
        continue;
      }
      result = push;
    }
    return invalid ? std::nullopt : result;
  }

  bool operationWritesDFB(Operation *operation, int64_t logicalId) const {
    if (isa<CBReserveOp, CBPushOp, CBWaitOp, CBPopOp, AttachCBOp, WaitOp>(
            operation)) {
      return false;
    }
    auto effects = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!effects || !effects.hasEffect<MemoryEffects::Write>()) {
      return false;
    }
    return llvm::any_of(operation->getOperands(), [&](Value operand) {
      return logicalDFBResolver(operand) == logicalId;
    });
  }

  std::optional<Operation *> findUniqueProducer(CBPushOp push,
                                                int64_t logicalId) const {
    Block *block = push->getBlock();
    Operation *reserve = nullptr;
    for (auto operation = push->getIterator(); operation != block->begin();) {
      --operation;
      if (auto priorPush = dyn_cast<CBPushOp>(&*operation)) {
        if (logicalDFBResolver(priorPush.getCb()) == logicalId) {
          return std::nullopt;
        }
      }
      if (auto candidate = dyn_cast<CBReserveOp>(&*operation)) {
        if (logicalDFBResolver(candidate.getCb()) == logicalId) {
          reserve = candidate;
          break;
        }
      }
    }
    if (!reserve) {
      return std::nullopt;
    }

    Operation *producer = nullptr;
    bool directCopyWaited = false;
    bool invalid = false;
    for (Operation *operation = reserve->getNextNode(); operation != push;
         operation = operation->getNextNode()) {
      bool writesTarget = operationWritesDFB(operation, logicalId);
      if (writesTarget) {
        if (producer) {
          invalid = true;
        } else {
          producer = operation;
        }
      }
      if (auto wait = dyn_cast<WaitOp>(operation)) {
        if (auto copy = dyn_cast_if_present<CopyOp>(producer);
            copy && wait.getXf() == copy.getResult()) {
          directCopyWaited = true;
        }
      }
    }
    if (invalid || !producer || (isa<CopyOp>(producer) && !directCopyWaited)) {
      return std::nullopt;
    }
    return producer;
  }

  std::optional<TensorScalarOrigin>
  traceDFB(Value dfb, ArrayRef<int64_t> scalarCoords,
           const LaunchExecutionLocation &location, Operation *consumer,
           SmallVector<std::pair<int64_t, LaunchExecutionLocation>, 4> visited)
      const {
    std::optional<int64_t> logicalId = logicalDFBResolver(dfb);
    if (!logicalId || llvm::any_of(visited, [&](const auto &entry) {
          return entry.first == *logicalId && entry.second == location;
        })) {
      return std::nullopt;
    }
    visited.emplace_back(*logicalId, location);
    std::optional<CBPushOp> push =
        findUniquePush(*logicalId, location, consumer);
    if (!push) {
      return std::nullopt;
    }
    std::optional<Operation *> producer = findUniqueProducer(*push, *logicalId);
    if (!producer) {
      return std::nullopt;
    }
    if (auto copy = dyn_cast<CopyOp>(*producer)) {
      if (auto tensorSlice = copy.getSrc().getDefiningOp<TensorSliceOp>()) {
        std::optional<SmallVector<int64_t>> tileIndices =
            evaluateCoords(tensorSlice.getIndices(), location);
        if (!tileIndices) {
          return std::nullopt;
        }
        Value tensor = tensorSlice.getTensor();
        return TensorScalarOrigin{tensor, getGlobalTensorIndex(tensor),
                                  *tileIndices,
                                  SmallVector<int64_t>(scalarCoords), location};
      }
    }

    std::optional<PipeNetScalarTransportSource> transportSource =
        transportSourceResolver(*producer, location, *push);
    if (!transportSource || !transportSource->consumer) {
      return std::nullopt;
    }
    return traceDFB(transportSource->dfb, scalarCoords,
                    transportSource->location, transportSource->consumer,
                    std::move(visited));
  }

  std::optional<TensorScalarOrigin>
  traceRead(ReadIndexOp read, const LaunchExecutionLocation &location) const {
    TraceKey traceKey{read.getOperation(), location};
    if (auto cached = tracedOrigins.find(traceKey);
        cached != tracedOrigins.end()) {
      return cached->second;
    }
    std::optional<SmallVector<int64_t>> scalarCoords =
        evaluateCoords(read.getCoords(), location);
    Value dfb = getAttachedCB(read.getBlock());
    if (!scalarCoords || !dfb) {
      return std::nullopt;
    }
    std::optional<TensorScalarOrigin> origin =
        traceDFB(dfb, *scalarCoords, location, read, {});
    if (origin) {
      tracedOrigins.try_emplace(traceKey, *origin);
    }
    return origin;
  }

  struct TraceKey {
    Operation *read;
    LaunchExecutionLocation location;

    bool operator<(const TraceKey &rhs) const {
      if (std::less<Operation *>{}(read, rhs.read)) {
        return true;
      }
      if (std::less<Operation *>{}(rhs.read, read)) {
        return false;
      }
      return location < rhs.location;
    }
  };

  const LaunchNodeDomainState &launchDomains;
  LogicalDFBResolver logicalDFBResolver;
  OperationCountResolver operationCountResolver;
  TransportSourceResolver transportSourceResolver;
  llvm::DenseMap<int64_t, SmallVector<CBPushOp>> pushesByLogicalId;
  mutable std::map<TraceKey, TensorScalarOrigin> tracedOrigins;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPENETSCALARORIGINANALYSIS_H
