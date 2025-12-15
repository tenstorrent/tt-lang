// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLKERNELREGIONSTOFUNCS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Map TTL ThreadType to TTKernel ThreadType.
static ttkernel::ThreadType mapThreadType(ThreadType ttlThread) {
  switch (ttlThread) {
  case ThreadType::compute:
    return ttkernel::ThreadType::Compute;
  case ThreadType::datamovement:
    return ttkernel::ThreadType::Noc;
  }
  llvm_unreachable("unhandled TTL ThreadType");
}

/// Generate a unique function name for a thread region.
static std::string generateFuncName(KernelOp kernelOp, unsigned regionIndex,
                                    ThreadType threadType) {
  std::string baseName;
  if (auto funcParent =
          kernelOp->getParentOfType<mlir::func::FuncOp>()) {
    baseName = funcParent.getName().str();
  } else {
    baseName = "kernel";
  }

  StringRef threadTypeName =
      threadType == ThreadType::compute ? "compute" : "dm";
  return baseName + "_" + threadTypeName.str() + "_" +
         std::to_string(regionIndex);
}

/// Extract a single region to a func.func.
static func::FuncOp extractRegionToFunc(OpBuilder &builder, KernelOp kernelOp,
                                        unsigned regionIndex,
                                        ThreadType threadType) {
  Region &region = kernelOp.getThreadRegions()[regionIndex];
  Block &block = region.front();

  auto funcName = generateFuncName(kernelOp, regionIndex, threadType);
  auto funcType = builder.getFunctionType(block.getArgumentTypes(), {});
  auto funcOp = builder.create<func::FuncOp>(kernelOp.getLoc(), funcName,
                                             funcType);
  funcOp.setPrivate();

  // Add ttkernel.thread attribute.
  auto tkThreadType = mapThreadType(threadType);
  funcOp->setAttr("ttkernel.thread",
                  ttkernel::ThreadTypeAttr::get(builder.getContext(),
                                                 tkThreadType));

  // Move region body into function.
  funcOp.getBody().takeBody(region);

  // Add return if block has no terminator.
  Block &funcBlock = funcOp.getBody().front();
  if (funcBlock.empty() || !funcBlock.back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&funcBlock);
    builder.create<func::ReturnOp>(kernelOp.getLoc());
  }

  return funcOp;
}

struct TTLKernelRegionsToFuncsPass
    : impl::TTLKernelRegionsToFuncsBase<TTLKernelRegionsToFuncsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    SmallVector<KernelOp> kernelOps;
    mod.walk([&](KernelOp op) { kernelOps.push_back(op); });

    for (KernelOp kernelOp : kernelOps) {
      // Insert functions at module level, before the function containing the
      // kernel op.
      if (auto parentFunc =
              kernelOp->getParentOfType<func::FuncOp>()) {
        builder.setInsertionPoint(parentFunc);
      } else {
        builder.setInsertionPointToStart(mod.getBody());
      }

      // Extract each region to a function.
      SmallVector<Attribute> newThreadAttrs;
      unsigned numRegions = kernelOp.getNumThreadRegions();

      for (unsigned i = 0; i < numRegions; ++i) {
        ThreadType threadType = kernelOp.getRegionThreadType(i);
        func::FuncOp funcOp =
            extractRegionToFunc(builder, kernelOp, i, threadType);

        // Build new ThreadAttr with symbol reference.
        auto symbolRef = FlatSymbolRefAttr::get(builder.getContext(),
                                                 funcOp.getName());
        newThreadAttrs.push_back(
            ThreadAttr::get(builder.getContext(), threadType, symbolRef));
      }

      // Update kernel op threads attribute with symbol references.
      kernelOp.setThreadsAttr(builder.getArrayAttr(newThreadAttrs));
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
