// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Dump CB Flow Graph Pass
//===----------------------------------------------------------------------===//
//
// Analysis pass that builds and dumps the CB producer/consumer flow graph.
// This enables the auto-profiler to correlate runtime barrier timings with
// source-level CB operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

#define DEBUG_TYPE "ttl-dump-cb-flow-graph"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLDUMPCBFLOWGRAPH
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Information about a CB operation for the flow graph.
struct CBOpInfo {
  std::string kernel;
  std::string thread;
  int64_t line;
  std::string op;        // "cb_wait", "cb_reserve", "copy", "wait"
  std::string direction; // "read" or "write" for copy/wait ops
};

/// Information about a circular buffer in the flow graph.
struct CBFlowInfo {
  std::optional<int64_t> dfbId;
  int64_t cbIndex;
  llvm::SmallVector<CBOpInfo> producers;
  llvm::SmallVector<CBOpInfo> consumers;
  llvm::SmallVector<CBOpInfo> dmaOps;  // copy operations
  llvm::SmallVector<CBOpInfo> waitOps; // ttl.wait operations
};

/// Extract line number from an operation's location.
static int64_t getLineNumber(Operation *op) {
  auto loc = op->getLoc();

  // Try FileLineColLoc first
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    return fileLoc.getLine();
  }

  // Try FusedLoc (may contain FileLineColLoc)
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (auto innerLoc : fusedLoc.getLocations()) {
      if (auto fileLoc = dyn_cast<FileLineColLoc>(innerLoc)) {
        return fileLoc.getLine();
      }
    }
  }

  // Try CallSiteLoc
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    if (auto fileLoc = dyn_cast<FileLineColLoc>(callLoc.getCaller())) {
      return fileLoc.getLine();
    }
  }

  return -1; // Unknown
}

/// Get the kernel name from a function.
static std::string getKernelName(func::FuncOp func) {
  return func.getName().str();
}

/// Get the thread type from a function's ttl.kernel_thread attribute.
static std::string getThreadType(func::FuncOp func) {
  if (auto threadAttr = func->getAttrOfType<tt::ttkernel::ThreadTypeAttr>(
          kKernelThreadAttrName)) {
    auto thread = threadAttr.getValue();
    switch (thread) {
    case tt::ttkernel::ThreadType::Noc:
      return "noc";
    case tt::ttkernel::ThreadType::Compute:
      return "compute";
    default:
      return "unknown";
    }
  }
  return "unknown";
}

/// Check if a type is a CB type.
static bool isCBType(Type type) { return isa<CircularBufferType>(type); }

/// Return logical identity when available so physical reuse does not merge
/// distinct DFB flows. Pre-finalization compiler DFBs fall back to their
/// provisional physical index.
static int64_t getDFBFlowKey(Value dfb) {
  if (FailureOr<int64_t> dfbId = getDFBId(dfb); succeeded(dfbId)) {
    return *dfbId;
  }
  std::optional<int64_t> cbIndex = getCBIndex(dfb);
  assert(cbIndex.has_value() && "DFB operand must trace to a declaration");
  return -*cbIndex - 1;
}

/// Get transfer direction from transfer handle type.
static std::string getTransferDirection(Type handleType) {
  if (auto thType = dyn_cast<TransferHandleType>(handleType)) {
    auto kind = thType.getKind();
    if (kind == TransferKind::read) {
      return "read";
    } else if (kind == TransferKind::write) {
      return "write";
    }
  }
  return "unknown";
}

struct TTLDumpCBFlowGraphPass
    : impl::TTLDumpCBFlowGraphBase<TTLDumpCBFlowGraphPass> {
  using TTLDumpCBFlowGraphBase::TTLDumpCBFlowGraphBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Map from logical DFB identity to flow info when identity is available.
    llvm::DenseMap<int64_t, CBFlowInfo> cbFlows;

    // Walk all functions
    mod.walk([&](func::FuncOp func) {
      std::string kernelName = getKernelName(func);
      std::string threadType = getThreadType(func);

      // Find all CB operations in this function
      func.walk([&](Operation *op) {
        if (auto bindOp = dyn_cast<BindCBOp>(op)) {
          // Initialize CB flow info
          int64_t cbIndex = bindOp.getCbIndex().getSExtValue();
          FailureOr<int64_t> dfbId = getDFBId(bindOp.getResult());
          int64_t flowKey = succeeded(dfbId) ? *dfbId : -cbIndex - 1;
          if (cbFlows.find(flowKey) == cbFlows.end()) {
            cbFlows[flowKey] =
                CBFlowInfo{succeeded(dfbId) ? std::optional<int64_t>(*dfbId)
                                            : std::nullopt,
                           cbIndex,
                           {},
                           {},
                           {},
                           {}};
          }
        } else if (auto waitOp = dyn_cast<CBWaitOp>(op)) {
          // Consumer: cb_wait
          int64_t flowKey = getDFBFlowKey(waitOp.getCb());
          CBOpInfo info{kernelName, threadType, getLineNumber(op), "cb_wait",
                        ""};
          cbFlows[flowKey].consumers.push_back(info);
        } else if (auto reserveOp = dyn_cast<CBReserveOp>(op)) {
          // Producer: cb_reserve
          int64_t flowKey = getDFBFlowKey(reserveOp.getCb());
          CBOpInfo info{kernelName, threadType, getLineNumber(op), "cb_reserve",
                        ""};
          cbFlows[flowKey].producers.push_back(info);
        } else if (auto copyOp = dyn_cast<CopyOp>(op)) {
          // DMA operation: one operand is a CB, the other a tensor.
          Value src = copyOp.getSrc();
          Value dst = copyOp.getDst();
          std::string direction;
          int64_t flowKey;

          if (isCBType(dst.getType())) {
            flowKey = getDFBFlowKey(dst);
            direction = "read"; // Reading from DRAM to CB
          } else if (isCBType(src.getType())) {
            flowKey = getDFBFlowKey(src);
            direction = "write"; // Writing from CB to DRAM
          } else {
            return; // Tensor-to-tensor copy: no CB to record.
          }

          CBOpInfo info{kernelName, threadType, getLineNumber(op), "copy",
                        direction};
          cbFlows[flowKey].dmaOps.push_back(info);
        } else if (auto waitOp = dyn_cast<WaitOp>(op)) {
          // DMA wait/barrier — trace back to the copy op to find the CB.
          auto copyOp = waitOp.getXf().getDefiningOp<CopyOp>();
          if (!copyOp) {
            return;
          }
          std::string direction =
              getTransferDirection(waitOp.getXf().getType());
          Value src = copyOp.getSrc();
          Value dst = copyOp.getDst();
          int64_t flowKey;

          if (isCBType(dst.getType())) {
            flowKey = getDFBFlowKey(dst);
          } else if (isCBType(src.getType())) {
            flowKey = getDFBFlowKey(src);
          } else {
            return; // Tensor-to-tensor copy: no CB to record.
          }

          CBOpInfo info{kernelName, threadType, getLineNumber(op), "wait",
                        direction};
          cbFlows[flowKey].waitOps.push_back(info);
        }
      });
    });

    // Print the graph (disabled by default, enable for debugging)
#if 0
    printGraph(cbFlows);
#endif

    // Output JSON if path specified
    if (!outputPath.empty()) {
      writeJSON(cbFlows);
    }
  }

  void printGraph(const llvm::DenseMap<int64_t, CBFlowInfo> &cbFlows) {
    llvm::errs() << "\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "CB Flow Graph\n";
    llvm::errs() << "========================================\n";

    for (const auto &flowEntry : cbFlows) {
      const CBFlowInfo &info = flowEntry.second;
      if (info.dfbId) {
        llvm::errs() << "\nDFB[" << *info.dfbId << "] CB[" << info.cbIndex
                     << "]:\n";
      } else {
        llvm::errs() << "\nCB[" << info.cbIndex << "]:\n";
      }

      if (!info.producers.empty()) {
        llvm::errs() << "  producers:\n";
        for (const auto &op : info.producers) {
          llvm::errs() << "    " << op.kernel << ":" << op.line << " (" << op.op
                       << ")\n";
        }
      }

      if (!info.dmaOps.empty()) {
        llvm::errs() << "  dma:\n";
        for (const auto &op : info.dmaOps) {
          llvm::errs() << "    " << op.kernel << ":" << op.line << " (copy "
                       << op.direction << ")\n";
        }
      }

      if (!info.waitOps.empty()) {
        llvm::errs() << "  barriers:\n";
        for (const auto &op : info.waitOps) {
          llvm::errs() << "    " << op.kernel << ":" << op.line << " (wait "
                       << op.direction << ")\n";
        }
      }

      if (!info.consumers.empty()) {
        llvm::errs() << "  consumers:\n";
        for (const auto &op : info.consumers) {
          llvm::errs() << "    " << op.kernel << ":" << op.line << " (" << op.op
                       << ")\n";
        }
      }
    }

    llvm::errs() << "\n========================================\n\n";
  }

  void writeJSON(const llvm::DenseMap<int64_t, CBFlowInfo> &cbFlows) {
    llvm::json::Object root;
    llvm::json::Array cbArray;

    for (const auto &flowEntry : cbFlows) {
      const CBFlowInfo &info = flowEntry.second;
      llvm::json::Object cbObj;
      cbObj["cb_index"] = info.cbIndex;
      if (info.dfbId) {
        cbObj["dfb_id"] = *info.dfbId;
      }

      auto opsToArray = [](const llvm::SmallVector<CBOpInfo> &ops) {
        llvm::json::Array arr;
        for (const auto &op : ops) {
          llvm::json::Object opObj;
          opObj["kernel"] = op.kernel;
          opObj["thread"] = op.thread;
          opObj["line"] = op.line;
          opObj["op"] = op.op;
          if (!op.direction.empty()) {
            opObj["direction"] = op.direction;
          }
          arr.push_back(std::move(opObj));
        }
        return arr;
      };

      cbObj["producers"] = opsToArray(info.producers);
      cbObj["consumers"] = opsToArray(info.consumers);
      cbObj["dma_ops"] = opsToArray(info.dmaOps);
      cbObj["wait_ops"] = opsToArray(info.waitOps);

      cbArray.push_back(std::move(cbObj));
    }

    root["circular_buffers"] = std::move(cbArray);

    std::error_code ec;
    llvm::raw_fd_ostream os(outputPath, ec);
    if (ec) {
      llvm::errs() << "Error writing CB flow JSON to " << outputPath << ": "
                   << ec.message() << "\n";
      return;
    }

    os << llvm::json::Value(std::move(root));
  }
};

} // namespace

} // namespace mlir::tt::ttl
