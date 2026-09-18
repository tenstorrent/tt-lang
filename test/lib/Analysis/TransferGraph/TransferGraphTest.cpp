// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace mlir;
using namespace mlir::tt::ttl;

namespace {

// Static edge enumeration is independent of the arithmetic emitted for dynamic
// device indices. Compare both representations at every valid endpoint.
LogicalResult checkGraph(func::FuncOp fixture, TransferGraphAttr attribute) {
  auto graph = createTransferGraph(attribute);
  auto edges = graph->getEdges();
  auto edgeCount = graph->getEdgeCount();
  if (failed(edgeCount) || *edgeCount != edges.size()) {
    return fixture.emitError("edge count disagrees with enumeration");
  }
  int64_t deviceCount = 1;
  for (auto component : graph->getDomain().getComponents()) {
    for (int64_t extent : component.getExtent().asArrayRef()) {
      deviceCount *= extent;
    }
  }

  for (int64_t deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
    for (PipeRole role : {PipeRole::Source, PipeRole::Destination}) {
      SmallVector<int64_t> expected{0};
      for (auto [ordinal, edge] : llvm::enumerate(edges)) {
        int64_t source =
            getLogicalDeviceIndex(graph->getDomain(), edge.getSource());
        int64_t destination =
            getLogicalDeviceIndex(graph->getDomain(), edge.getDestination());
        if ((role == PipeRole::Source ? source : destination) == deviceIndex) {
          ++expected.front();
          expected.append({static_cast<int64_t>(ordinal), source, destination});
        }
      }

      OpBuilder builder(fixture.getContext());
      Location location = fixture.getLoc();
      OwningOpRef<ModuleOp> generated(ModuleOp::create(location));
      builder.setInsertionPointToStart(generated->getBody());
      SmallVector<Type> resultTypes(expected.size(), builder.getIndexType());
      auto function =
          func::FuncOp::create(builder, location, "generated",
                               builder.getFunctionType({}, resultTypes));
      builder.setInsertionPointToStart(function.addEntryBlock());
      Value device =
          arith::ConstantIndexOp::create(builder, location, deviceIndex);
      SmallVector<Value> actual{
          graph->buildIncidentEdgeCount(builder, location, device, role)};
      for (int64_t incidentIndex = 0; incidentIndex < expected.front();
           ++incidentIndex) {
        Value incident =
            arith::ConstantIndexOp::create(builder, location, incidentIndex);
        auto indices = graph->buildIncidentEdgeIndexValues(
            builder, location, device, incident, role);
        actual.append({indices.edgeOrdinal, indices.sourceDeviceIndex,
                       indices.destinationDeviceIndex});
      }
      auto returnOp = func::ReturnOp::create(builder, location, actual);
      RewritePatternSet patterns(fixture.getContext());
      mlir::tt::ttkernel::ConstantTableLookupOp::getCanonicalizationPatterns(
          patterns, fixture.getContext());
      if (failed(applyPatternsGreedily(*generated, std::move(patterns)))) {
        return fixture.emitError("generated index expressions did not fold");
      }
      for (auto [position, expectedIndex] : llvm::enumerate(expected)) {
        APInt folded;
        if (!matchPattern(returnOp.getOperand(position),
                          m_ConstantInt(&folded)) ||
            folded.getSExtValue() != expectedIndex) {
          auto diagnostic =
              fixture.emitError("dynamic graph mismatch: device ");
          diagnostic << deviceIndex << ", role "
                     << (role == PipeRole::Source ? "source" : "destination")
                     << ", result " << position << ", expected "
                     << expectedIndex << ", actual "
                     << returnOp.getOperand(position);
          return failure();
        }
      }
    }
  }
  llvm::outs() << fixture.getSymName() << ": " << deviceCount << " devices, "
               << edges.size() << " edges, both endpoint roles verified\n";
  return success();
}

} // namespace

int main(int argumentCount, char **argumentValues) {
  llvm::InitLLVM initLLVM(argumentCount, argumentValues);
  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                           llvm::cl::desc("<input MLIR file>"),
                                           llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(argumentCount, argumentValues);
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, TTLDialect,
                  mlir::tt::ttkernel::TTKernelDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module) {
    return 1;
  }
  unsigned graphCount = 0;
  for (auto fixture : module->getOps<func::FuncOp>()) {
    auto graph = fixture->getAttrOfType<TransferGraphAttr>("test.graph");
    if (!graph || failed(checkGraph(fixture, graph))) {
      return 1;
    }
    ++graphCount;
  }
  return graphCount == 0;
}
