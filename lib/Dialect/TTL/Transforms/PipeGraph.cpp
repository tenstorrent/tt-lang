// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeGraph.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::tt::ttl {

LogicalResult PipeGraph::addReceiverDFB(int64_t srcX, int64_t srcY,
                                        int64_t dstStartX, int64_t dstStartY,
                                        int64_t dstEndX, int64_t dstEndY,
                                        int64_t pipeNetId, int64_t dfbIndex,
                                        CircularBufferType dfbType,
                                        int64_t staticTileOffset,
                                        int64_t blockCount, Location loc) {
  PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY, pipeNetId};
  auto existing = receiverDFBs.find(key);
  bool hasMultipleReceivers = dstStartX != dstEndX || dstStartY != dstEndY;
  if (existing != receiverDFBs.end()) {
    if (hasMultipleReceivers &&
        (existing->second.dfbIndex != dfbIndex ||
         existing->second.dfbType != dfbType ||
         existing->second.staticTileOffset != staticTileOffset)) {
      auto diag = emitError(loc)
                  << "collective pipe receive posts publish different "
                     "destination addresses; per-receiver destination "
                     "addresses are tracked by issue #617";
      diag.attachNote(existing->second.loc)
          << "previous collective receive post for this pipe was here";
      return failure();
    }

    if (existing->second.dfbIndex != dfbIndex ||
        existing->second.dfbType != dfbType ||
        existing->second.blockCount != blockCount) {
      auto diag = emitError(loc)
                  << "conflicting receiver DFBs for the same pipe";
      diag.attachNote(existing->second.loc)
          << "previous receiver DFB for this pipe was here";
      return failure();
    }
    return success();
  }
  receiverDFBs.insert(
      {key, {dfbIndex, dfbType, staticTileOffset, 0, blockCount, loc}});
  return success();
}

void PipeGraph::assignGatherSlotIndices() {
  // (receiver, DFB index) -> slots already taken at that receiver.
  struct ReceiverKey {
    int64_t recvX, recvY, dfbIndex;
    bool operator==(const ReceiverKey &other) const {
      return recvX == other.recvX && recvY == other.recvY &&
             dfbIndex == other.dfbIndex;
    }
  };
  struct ReceiverKeyInfo {
    static unsigned getHashValue(const ReceiverKey &key) {
      return llvm::hash_combine(key.recvX, key.recvY, key.dfbIndex);
    }
    static bool isEqual(const ReceiverKey &lhs, const ReceiverKey &rhs) {
      return lhs == rhs;
    }
  };
  using ReceiverSlotMap =
      llvm::MapVector<ReceiverKey, llvm::SmallSetVector<int64_t, 4>,
                      llvm::DenseMap<ReceiverKey, unsigned, ReceiverKeyInfo>>;
  ReceiverSlotMap usedAtReceiver;

  SmallVector<PipeKey> sortedKeys;
  sortedKeys.reserve(receiverDFBs.size());
  for (const auto &entry : receiverDFBs) {
    sortedKeys.push_back(entry.first);
  }
  llvm::sort(sortedKeys, [](const PipeKey &lhs, const PipeKey &rhs) {
    return std::make_tuple(lhs.srcX, lhs.srcY, lhs.dstStartX, lhs.dstStartY,
                           lhs.dstEndX, lhs.dstEndY, lhs.pipeNetId) <
           std::make_tuple(rhs.srcX, rhs.srcY, rhs.dstStartX, rhs.dstStartY,
                           rhs.dstEndX, rhs.dstEndY, rhs.pipeNetId);
  });

  for (const PipeKey &pk : sortedKeys) {
    auto it = receiverDFBs.find(pk);
    const int64_t dfbIndex = it->second.dfbIndex;

    // Slots taken by earlier pipes at any of this pipe's receivers
    // (destination range is inclusive on both ends).
    llvm::SmallSetVector<int64_t, 4> taken;
    for (int64_t dstY = pk.dstStartY; dstY <= pk.dstEndY; ++dstY) {
      for (int64_t dstX = pk.dstStartX; dstX <= pk.dstEndX; ++dstX) {
        auto receiverIt =
            usedAtReceiver.find(ReceiverKey{dstX, dstY, dfbIndex});
        if (receiverIt == usedAtReceiver.end()) {
          continue;
        }
        for (int64_t slotIndex : receiverIt->second) {
          taken.insert(slotIndex);
        }
      }
    }

    // Lowest free slot.
    int64_t slot = 0;
    while (taken.count(slot)) {
      ++slot;
    }
    it->second.gatherSlotIdx = slot;

    // Reserve this slot at every receiver.
    for (int64_t dstY = pk.dstStartY; dstY <= pk.dstEndY; ++dstY) {
      for (int64_t dstX = pk.dstStartX; dstX <= pk.dstEndX; ++dstX) {
        usedAtReceiver[ReceiverKey{dstX, dstY, dfbIndex}].insert(slot);
      }
    }
  }
}

LogicalResult PipeGraph::verifyReceiverDFBBlockCounts() const {
  for (auto &[pk, info] : receiverDFBs) {
    int64_t requiredBlocks = info.gatherSlotIdx + 1;
    if (info.blockCount < requiredBlocks) {
      bool hasSingleReceiver =
          pk.dstStartX == pk.dstEndX && pk.dstStartY == pk.dstEndY;
      return emitError(info.loc)
             << (hasSingleReceiver ? "gather" : "collective overlap")
             << " pipe receiver DFB has block_count=" << info.blockCount
             << " but slot " << info.gatherSlotIdx
             << " is assigned to this pipe; "
             << "block_count must be >= " << requiredBlocks;
    }
  }
  return success();
}

const ReceiverDFBInfo *PipeGraph::lookupReceiverDFB(const PipeKey &key) const {
  auto it = receiverDFBs.find(key);
  if (it == receiverDFBs.end()) {
    return nullptr;
  }
  return &it->second;
}

static PipeKey getPipeKey(PipeType pipeType) {
  return {pipeType.getSrcX(),      pipeType.getSrcY(),
          pipeType.getDstStartX(), pipeType.getDstStartY(),
          pipeType.getDstEndX(),   pipeType.getDstEndY(),
          pipeType.getPipeNetId()};
}

static llvm::MapVector<PipeKey, PipeTransferContract>
collectPipeTransferContracts(ModuleOp mod) {
  llvm::MapVector<PipeKey, PipeTransferContract> contracts;
  mod.walk([&](CreatePipeOp op) {
    auto pipeType = mlir::cast<PipeType>(op.getResult().getType());
    PipeTransferContract contract = getPipeTransferContract(op);
    PipeKey key = getPipeKey(pipeType);
    auto existing = contracts.find(key);
    if (existing == contracts.end()) {
      contracts.insert({key, contract});
      return;
    }
    // Duplicate create_pipe ops for the same PipeKey can arise from cloned
    // regions. Collective is the stronger contract and must be preserved.
    if (isCollectiveTransfer(contract)) {
      existing->second = PipeTransferContract::Collective;
    }
  });
  return contracts;
}

static LogicalResult
emitUntraceableCollectiveDestinationAddress(Operation *op) {
  return op->emitError()
         << "collective pipe destination address could not be "
            "determined statically; per-receiver destination addresses are "
            "tracked by issue #617";
}

static LogicalResult addStaticCoordinates(ArrayRef<OpFoldResult> mixedOffsets,
                                          SmallVectorImpl<int64_t> &coordinates,
                                          unsigned rank) {
  if (coordinates.empty()) {
    coordinates.assign(rank, 0);
  }
  if (coordinates.size() != rank || mixedOffsets.size() != rank) {
    return failure();
  }

  for (auto [coordinate, mixedOffset] :
       llvm::zip_equal(coordinates, mixedOffsets)) {
    std::optional<int64_t> offset = getConstantIntValue(mixedOffset);
    if (!offset.has_value()) {
      return failure();
    }
    coordinate += *offset;
  }
  return success();
}

/// Return the static tile offset within the receiver DFB for a receive
/// destination. Collective lowering has one sender-visible address-table entry
/// per pipe, so each destination must publish the same static DFB address until
/// issue #617 adds explicit per-destination addresses.
static FailureOr<int64_t> getStaticDestinationTileOffset(Value dst) {
  Value view = traceUnrealizedCasts(dst);
  SmallVector<int64_t> coordinates;
  RankedTensorType rootType;
  bool sawOffset = false;

  while (true) {
    view = traceUnrealizedCasts(view);
    if (auto extract = view.getDefiningOp<tensor::ExtractOp>()) {
      auto tensorType =
          mlir::dyn_cast<RankedTensorType>(extract.getTensor().getType());
      if (!tensorType) {
        return failure();
      }
      SmallVector<OpFoldResult> mixedIndices;
      for (Value index : extract.getIndices()) {
        mixedIndices.push_back(index);
      }
      if (failed(addStaticCoordinates(mixedIndices, coordinates,
                                      tensorType.getRank()))) {
        return failure();
      }
      sawOffset = true;
      view = extract.getTensor();
      continue;
    }
    if (auto attach = view.getDefiningOp<AttachCBOp>()) {
      view = attach.getTensor();
      continue;
    }

    auto slice = view.getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice) {
      rootType = mlir::dyn_cast<RankedTensorType>(view.getType());
      break;
    }

    auto sourceType =
        mlir::dyn_cast<RankedTensorType>(slice.getSource().getType());
    if (!sourceType) {
      return failure();
    }

    if (failed(addStaticCoordinates(slice.getMixedOffsets(), coordinates,
                                    sourceType.getRank()))) {
      return failure();
    }
    sawOffset = true;
    view = slice.getSource();
  }

  if (!sawOffset) {
    return 0;
  }
  if (!rootType ||
      rootType.getRank() != static_cast<int64_t>(coordinates.size())) {
    return failure();
  }

  int64_t linearOffset = 0;
  for (auto [coordinate, dim] :
       llvm::zip_equal(coordinates, rootType.getShape())) {
    if (dim == ShapedType::kDynamic) {
      return failure();
    }
    linearOffset = linearOffset * dim + coordinate;
  }
  return linearOffset;
}

static LogicalResult addPipeReceiver(PipeGraph &graph, Operation *op,
                                     PipeType pipeType,
                                     PipeTransferContract transferContract,
                                     Value dst) {
  Value dstDFB = getAttachedCB(dst);
  if (!dstDFB) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }
  auto dfbType = mlir::dyn_cast<CircularBufferType>(dstDFB.getType());
  if (!dfbType) {
    return op->emitError("pipe receive destination is not attached to a DFB");
  }

  std::optional<int64_t> dfbIndex = getCBIndex(dstDFB);
  if (!dfbIndex.has_value()) {
    return op->emitError("could not trace pipe receiver to a DFB binding");
  }

  int64_t staticTileOffset = 0;
  if (isCollectiveTransfer(transferContract)) {
    FailureOr<int64_t> offset = getStaticDestinationTileOffset(dst);
    if (failed(offset)) {
      return emitUntraceableCollectiveDestinationAddress(op);
    }
    staticTileOffset = *offset;
  }

  return graph.addReceiverDFB(
      pipeType.getSrcX(), pipeType.getSrcY(), pipeType.getDstStartX(),
      pipeType.getDstStartY(), pipeType.getDstEndX(), pipeType.getDstEndY(),
      pipeType.getPipeNetId(), *dfbIndex, dfbType, staticTileOffset,
      dfbType.getBlockCount(), op->getLoc());
}

FailureOr<PipeGraph> PipeGraph::build(ModuleOp mod) {
  PipeGraph graph;
  llvm::MapVector<PipeKey, PipeTransferContract> transferContracts =
      collectPipeTransferContracts(mod);

  WalkResult walkResult = mod.walk([&](PipeRecvPostOp postOp) {
    auto pipeType = mlir::cast<PipeType>(postOp.getPipe().getType());
    PipeKey key = getPipeKey(pipeType);
    auto contractIt = transferContracts.find(key);
    if (contractIt == transferContracts.end()) {
      postOp.emitError("pipe receive must use a ttl.create_pipe result");
      return WalkResult::interrupt();
    }
    PipeTransferContract contract = contractIt->second;
    if (failed(addPipeReceiver(graph, postOp, pipeType, contract,
                               postOp.getDst()))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return failure();
  }

  graph.assignGatherSlotIndices();

  if (failed(graph.verifyReceiverDFBBlockCounts())) {
    return failure();
  }

  return graph;
}

} // namespace mlir::tt::ttl
