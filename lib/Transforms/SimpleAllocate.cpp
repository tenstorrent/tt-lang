#include "ttlang/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/Analysis/Allocation/Planner.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MSIMPLEALLOCATE
#include "ttlang/Transforms/Passes.h.inc"

namespace {

struct AllocInfo {
  memref::AllocOp op;
  ttcore::MemorySpace memSpace;
  allocation::Planner::AllocSizeT size;
  allocation::Planner::LiveRange range;
  int32_t varIndex = -1;
};

class D2MSimpleAllocate : public impl::D2MSimpleAllocateBase<D2MSimpleAllocate> {
public:
  using impl::D2MSimpleAllocateBase<D2MSimpleAllocate>::D2MSimpleAllocateBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.isDeclaration())
      return;

    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    ttcore::SystemDescAttr systemDesc =
        ttcore::getCurrentScopeSystemDesc(moduleOp);
    ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs().front();

    SmallVector<AllocInfo> allocs = collectAllocs(funcOp, chipDesc);

    if (failed(allocateL1(funcOp, chipDesc, allocs))) {
      return signalPassFailure();
    }

    assignAddresses(funcOp, chipDesc, allocs);
    insertDeallocs(funcOp, allocs);
  }

private:
  SmallVector<AllocInfo> collectAllocs(func::FuncOp funcOp,
                                        ttcore::ChipDescAttr chipDesc) {
    SmallVector<AllocInfo> allocs;
    mlir::Liveness liveness(funcOp);
    Block &body = funcOp.getBody().front();

    llvm::DenseMap<Operation *, allocation::Planner::SequenceT> opToSeq;
    allocation::Planner::SequenceT seq = 0;
    body.walk<WalkOrder::PreOrder>([&](Operation *op) { opToSeq[op] = seq++; });

    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    allocation::Planner::AllocSizeT l1Alignment = chipDesc.getNocL1AddressAlignBytes();
    allocation::Planner::AllocSizeT dramAlignment = chipDesc.getNocDRAMAddressAlignBytes();

    funcOp.walk([&](memref::AllocOp allocOp) {
      MemRefType type = allocOp.getType();
      ttcore::MemorySpace memSpace =
          ttcore::getMemorySpace(type, ttcore::MemorySpace::System);

      if (!ttcore::isDeviceMemorySpace(memSpace))
        return;

      int64_t sizeBytes = device.getMemrefSizeBytes(type, 0, false);
      allocation::Planner::AllocSizeT alignment =
          (memSpace == ttcore::MemorySpace::DeviceL1) ? l1Alignment : dramAlignment;
      allocation::Planner::AllocSizeT size = ttmlir::utils::alignUp(sizeBytes, alignment);

      Value result = allocOp.getResult();
      const mlir::LivenessBlockInfo *li = liveness.getLiveness(&body);
      Operation *startOp = li->getStartOperation(result);
      Operation *endOp = li->getEndOperation(result, startOp);

      // Extend through view/stream ops that alias the buffer
      endOp = extendLiveness(result, endOp, opToSeq);

      allocation::Planner::LiveRange range = {opToSeq[startOp], opToSeq[endOp]};

      allocs.push_back(
          {allocOp, memSpace, size, range, -1});
    });

    return allocs;
  }

  Operation *extendLiveness(Value val, Operation *currentEnd,
                            const llvm::DenseMap<Operation *, allocation::Planner::SequenceT> &opToSeq) {
    llvm::DenseSet<Value> visited;
    return extendLivenessImpl(val, currentEnd, opToSeq, visited);
  }

  Operation *extendLivenessImpl(
      Value val, Operation *currentEnd,
      const llvm::DenseMap<Operation *, allocation::Planner::SequenceT> &opToSeq,
      llvm::DenseSet<Value> &visited) {
    if (!visited.insert(val).second)
      return currentEnd;

    allocation::Planner::SequenceT maxSeq = opToSeq.lookup(currentEnd);

    for (Operation *user : val.getUsers()) {
      auto it = opToSeq.find(user);
      if (it != opToSeq.end()) {
        maxSeq = std::max(maxSeq, it->second);
      }

      if (auto viewOp = dyn_cast<d2m::ViewLayoutOp>(user)) {
        Operation *extendedEnd = extendLivenessImpl(
            viewOp.getResult(), currentEnd, opToSeq, visited);
        if (opToSeq.lookup(extendedEnd) > maxSeq)
          currentEnd = extendedEnd;
      } else if (auto streamOp = dyn_cast<d2m::StreamLayoutOp>(user)) {
        Operation *extendedEnd = extendLivenessImpl(
            streamOp.getResult(), currentEnd, opToSeq, visited);
        if (opToSeq.lookup(extendedEnd) > maxSeq)
          currentEnd = extendedEnd;
      } else if (auto genericOp = dyn_cast<d2m::GenericOp>(user)) {
        // Generic ops need all operands live until completion
        maxSeq = std::max(maxSeq, opToSeq.lookup(user));
        for (Value result : genericOp.getResults()) {
          Operation *extendedEnd = extendLivenessImpl(
              result, currentEnd, opToSeq, visited);
          if (opToSeq.lookup(extendedEnd) > maxSeq) {
            maxSeq = opToSeq.lookup(extendedEnd);
            currentEnd = extendedEnd;
          }
        }
      }
    }

    for (auto &[op, s] : opToSeq) {
      if (s == maxSeq)
        return op;
    }
    return currentEnd;
  }

  LogicalResult allocateL1(func::FuncOp funcOp, ttcore::ChipDescAttr chipDesc,
                            SmallVector<AllocInfo> &allocs) {
    allocation::Planner::AllocSizeT l1Base = chipDesc.getL1UnreservedBase();
    allocation::Planner::AllocSizeT l1Size = testAssumeL1Capacity > 0
                            ? (l1Base + testAssumeL1Capacity)
                            : chipDesc.getL1Size();
    allocation::Planner::AllocSizeT l1Capacity = l1Size - l1Base;

    allocation::Planner::Problem problem;

    for (auto &info : allocs) {
      if (info.memSpace == ttcore::MemorySpace::DeviceL1) {
        info.varIndex = problem.def([&](allocation::Planner::VariableBuilder &b) {
          b.request(allocation::Planner::Space::Scratch, info.size, info.range.first,
                    info.range.last);
          b.place(allocation::Planner::Space::Scratch);
        });
      }
    }

    auto stats = allocation::Planner::allocate(problem, allocation::Planner::Algorithm::Simple);

    if (stats.memUsage > l1Capacity) {
      return funcOp.emitError("L1 capacity exceeded: ")
             << stats.memUsage << " bytes required, " << l1Capacity
             << " bytes available";
    }

    for (auto &info : allocs) {
      if (info.varIndex >= 0) {
        const auto &var = problem.variable(info.varIndex);
        const auto &scratchRequests = var.domain[allocation::ordinal(allocation::Planner::Space::Scratch)];
        if (!scratchRequests.empty()) {
          auto reqIndex = *scratchRequests.begin();
          // Reuse size field to store offset (base address added later)
          info.size = problem.request(reqIndex).offset;
        }
      }
    }

    return success();
  }

  void assignAddresses(func::FuncOp funcOp,
                        ttcore::ChipDescAttr chipDesc,
                        SmallVector<AllocInfo> &allocs) {
    IRRewriter rewriter(funcOp.getContext());

    allocation::Planner::AllocSizeT l1Base = chipDesc.getL1UnreservedBase();
    allocation::Planner::AllocSizeT l1Alignment = chipDesc.getNocL1AddressAlignBytes();
    allocation::Planner::AllocSizeT dramAlignment = chipDesc.getNocDRAMAddressAlignBytes();

    for (auto &info : allocs) {
      if (info.memSpace == ttcore::MemorySpace::DeviceL1) {
        allocation::Planner::AllocSizeT address = l1Base + info.size;  // size holds offset here

        rewriter.startOpModification(info.op);
        info.op.setAlignment(l1Alignment);
        info.op->setAttr("address", rewriter.getI64IntegerAttr(address));
        rewriter.finalizeOpModification(info.op);
      } else if (info.memSpace == ttcore::MemorySpace::DeviceDRAM) {
        // DRAM addresses assigned by runtime
        rewriter.startOpModification(info.op);
        info.op.setAlignment(dramAlignment);
        rewriter.finalizeOpModification(info.op);
      }
    }
  }

  void insertDeallocs(func::FuncOp funcOp, SmallVector<AllocInfo> &allocs) {
    // All deallocs at function end (conservative but avoids aliasing issues)
    // TODO: Should DRAM allocs skip dealloc since runtime manages addresses?
    IRRewriter rewriter(funcOp.getContext());

    funcOp.walk([&](func::ReturnOp returnOp) {
      rewriter.setInsertionPoint(returnOp);
      for (auto &info : allocs) {
        rewriter.create<memref::DeallocOp>(returnOp.getLoc(),
                                            info.op.getResult());
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m
