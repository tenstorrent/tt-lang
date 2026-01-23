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
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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
  std::string op;       // "cb_wait", "cb_reserve", "copy", "wait"
  std::string direction; // "read" or "write" for copy/wait ops
};

/// Information about a circular buffer in the flow graph.
struct CBFlowInfo {
  int64_t cbIndex;
  std::string name; // Variable name if available
  llvm::SmallVector<CBOpInfo> producers;
  llvm::SmallVector<CBOpInfo> consumers;
  llvm::SmallVector<CBOpInfo> dmaOps; // copy operations
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
  if (auto threadAttr =
          func->getAttrOfType<tt::ttkernel::ThreadTypeAttr>("ttl.kernel_thread")) {
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

/// Get the CB index from a CB value (trace to bind_cb).
static int64_t getCBIndex(Value cb) {
  cb = traceUnrealizedCasts(cb);
  if (auto bindOp = cb.getDefiningOp<BindCBOp>()) {
    return bindOp.getCbIndex().getSExtValue();
  }
  return -1;
}

/// Check if a type is a CB type.
static bool isCBType(Type type) {
  return isa<CircularBufferType>(type);
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

    // Map from CB index to flow info
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
          if (cbFlows.find(cbIndex) == cbFlows.end()) {
            cbFlows[cbIndex] = CBFlowInfo{cbIndex, "", {}, {}, {}, {}};
          }
        } else if (auto waitOp = dyn_cast<CBWaitOp>(op)) {
          // Consumer: cb_wait
          int64_t cbIndex = getCBIndex(waitOp.getCb());
          if (cbIndex >= 0) {
            CBOpInfo info{kernelName, threadType, getLineNumber(op), "cb_wait",
                          ""};
            cbFlows[cbIndex].consumers.push_back(info);
          }
        } else if (auto reserveOp = dyn_cast<CBReserveOp>(op)) {
          // Producer: cb_reserve
          int64_t cbIndex = getCBIndex(reserveOp.getCb());
          if (cbIndex >= 0) {
            CBOpInfo info{kernelName, threadType, getLineNumber(op),
                          "cb_reserve", ""};
            cbFlows[cbIndex].producers.push_back(info);
          }
        } else if (auto copyOp = dyn_cast<CopyOp>(op)) {
          // DMA operation: copy
          // Determine which operand is the CB
          Value src = copyOp.getSrc();
          Value dst = copyOp.getDst();
          std::string direction;
          int64_t cbIndex = -1;

          if (isCBType(dst.getType())) {
            // Copy TO CB (read from tensor)
            cbIndex = getCBIndex(dst);
            direction = "read"; // Reading from DRAM to CB
          } else if (isCBType(src.getType())) {
            // Copy FROM CB (write to tensor)
            cbIndex = getCBIndex(src);
            direction = "write"; // Writing from CB to DRAM
          }

          if (cbIndex >= 0) {
            CBOpInfo info{kernelName, threadType, getLineNumber(op), "copy",
                          direction};
            cbFlows[cbIndex].dmaOps.push_back(info);
          }
        } else if (auto waitOp = dyn_cast<WaitOp>(op)) {
          // DMA wait/barrier
          std::string direction = getTransferDirection(waitOp.getXf().getType());

          // Try to trace back to the copy op to get CB index
          if (auto copyOp = waitOp.getXf().getDefiningOp<CopyOp>()) {
            Value src = copyOp.getSrc();
            Value dst = copyOp.getDst();
            int64_t cbIndex = -1;

            if (isCBType(dst.getType())) {
              cbIndex = getCBIndex(dst);
            } else if (isCBType(src.getType())) {
              cbIndex = getCBIndex(src);
            }

            if (cbIndex >= 0) {
              CBOpInfo info{kernelName, threadType, getLineNumber(op), "wait",
                            direction};
              cbFlows[cbIndex].waitOps.push_back(info);
            }
          }
        }
      });
    });

    // Print the graph
    printGraph(cbFlows);

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

    for (const auto &[cbIndex, info] : cbFlows) {
      llvm::errs() << "\nCB[" << cbIndex << "]:\n";

      if (!info.producers.empty()) {
        llvm::errs() << "  producers:\n";
        for (const auto &op : info.producers) {
          llvm::errs() << "    " << op.kernel << ":" << op.line << " ("
                       << op.op << ")\n";
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
          llvm::errs() << "    " << op.kernel << ":" << op.line << " ("
                       << op.op << ")\n";
        }
      }
    }

    llvm::errs() << "\n========================================\n\n";
  }

  void writeJSON(const llvm::DenseMap<int64_t, CBFlowInfo> &cbFlows) {
    llvm::json::Object root;
    llvm::json::Array cbArray;

    for (const auto &[cbIndex, info] : cbFlows) {
      llvm::json::Object cbObj;
      cbObj["cb_index"] = cbIndex;

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
