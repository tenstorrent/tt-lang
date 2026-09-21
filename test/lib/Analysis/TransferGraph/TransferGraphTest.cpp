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
      // A filtered relation has no closed-form incident mapping, so its
      // contract is the per-edge predicate rather than an ordinal mapping.
      const bool filtered =
          graph->getIncidentEdgeIteration() == IncidentEdgeIteration::Filtered;
      SmallVector<int64_t> expected{0};
      for (auto [ordinal, edge] : llvm::enumerate(edges)) {
        int64_t source =
            getLogicalDeviceIndex(graph->getDomain(), edge.getSource());
        int64_t destination =
            getLogicalDeviceIndex(graph->getDomain(), edge.getDestination());
        if ((role == PipeRole::Source ? source : destination) == deviceIndex) {
          if (!filtered && graph->getIncidentEdgeOrdinal(edge, role) !=
                               static_cast<std::uint64_t>(expected.front())) {
            return fixture.emitError("static endpoint-local edge ordinal "
                                     "disagrees with enumeration");
          }
          ++expected.front();
          expected.append({static_cast<int64_t>(ordinal), source, destination});
        }
      }

      OpBuilder builder(fixture.getContext());
      Location location = fixture.getLoc();
      OwningOpRef<ModuleOp> generated(ModuleOp::create(location));
      builder.setInsertionPointToStart(generated->getBody());
      size_t resultCount = filtered ? 4 * edges.size() : expected.size();
      SmallVector<Type> resultTypes(resultCount, builder.getIndexType());
      auto function =
          func::FuncOp::create(builder, location, "generated",
                               builder.getFunctionType({}, resultTypes));
      builder.setInsertionPointToStart(function.addEntryBlock());
      Value device =
          arith::ConstantIndexOp::create(builder, location, deviceIndex);
      SmallVector<Value> actual;
      if (filtered) {
        Value zero = arith::ConstantIndexOp::create(builder, location, 0);
        Value one = arith::ConstantIndexOp::create(builder, location, 1);
        for (size_t edgeOrdinal = 0; edgeOrdinal < edges.size();
             ++edgeOrdinal) {
          Value ordinal = arith::ConstantIndexOp::create(
              builder, location, static_cast<int64_t>(edgeOrdinal));
          Value incidence = graph->buildEdgeIncidence(builder, location, device,
                                                      ordinal, role);
          auto indices =
              graph->buildEdgeIndexValues(builder, location, ordinal);
          actual.push_back(
              arith::SelectOp::create(builder, location, incidence, one, zero));
          actual.append({indices.edgeOrdinal, indices.sourceDeviceIndex,
                         indices.destinationDeviceIndex});
        }
      } else {
        actual.push_back(
            graph->buildIncidentEdgeCount(builder, location, device, role));
        for (int64_t incidentIndex = 0; incidentIndex < expected.front();
             ++incidentIndex) {
          Value incident =
              arith::ConstantIndexOp::create(builder, location, incidentIndex);
          auto indices = graph->buildIncidentEdgeIndexValues(
              builder, location, device, incident, role);
          actual.append({indices.edgeOrdinal, indices.sourceDeviceIndex,
                         indices.destinationDeviceIndex});
        }
      }
      auto returnOp = func::ReturnOp::create(builder, location, actual);
      RewritePatternSet patterns(fixture.getContext());
      mlir::tt::ttkernel::ConstantTableLookupOp::getCanonicalizationPatterns(
          patterns, fixture.getContext());
      if (failed(applyPatternsGreedily(*generated, std::move(patterns)))) {
        return fixture.emitError("generated index expressions did not fold");
      }
      SmallVector<int64_t> folded;
      folded.reserve(resultCount);
      for (size_t position = 0; position < resultCount; ++position) {
        APInt value;
        if (!matchPattern(returnOp.getOperand(position),
                          m_ConstantInt(&value))) {
          auto diagnostic = fixture.emitError("graph index did not fold: ");
          diagnostic << "device " << deviceIndex << ", role "
                     << (role == PipeRole::Source ? "source" : "destination")
                     << ", result " << position << ", actual "
                     << returnOp.getOperand(position);
          return failure();
        }
        folded.push_back(value.getSExtValue());
      }
      SmallVector<int64_t> observed;
      if (filtered) {
        observed.push_back(0);
        for (size_t edgeOrdinal = 0; edgeOrdinal < edges.size();
             ++edgeOrdinal) {
          ArrayRef<int64_t> entry =
              ArrayRef<int64_t>(folded).slice(4 * edgeOrdinal, 4);
          if (entry[0] == 0) {
            continue;
          }
          ++observed.front();
          observed.append({entry[1], entry[2], entry[3]});
        }
      } else {
        observed.assign(folded.begin(), folded.end());
      }
      if (observed != expected) {
        auto diagnostic = fixture.emitError("dynamic graph mismatch: device ");
        diagnostic << deviceIndex << ", role "
                   << (role == PipeRole::Source ? "source" : "destination")
                   << ", expected " << expected.size() << " values, observed "
                   << observed.size();
        for (auto [position, expectedIndex] : llvm::enumerate(expected)) {
          if (position >= observed.size() ||
              observed[position] != expectedIndex) {
            diagnostic << "; first difference at " << position << ", expected "
                       << expectedIndex;
            if (position < observed.size()) {
              diagnostic << ", observed " << observed[position];
            }
            break;
          }
        }
        return failure();
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
