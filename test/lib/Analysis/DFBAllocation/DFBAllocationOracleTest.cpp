// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test treats each graph vertex as one DFB and each edge as a storage
// conflict. Exhaustive index and byte-placement enumeration supplies expected
// results without calling the production search.

#include "ttlang/Dialect/TTCore/IR/TTCore.h"
#include "ttlang/Dialect/TTL/Transforms/InterferenceGraphColoring.h"
#include "ttlang/Dialect/TTL/Transforms/SRAMAllocator.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace {

using mlir::tt::ttl::ExactInterferenceGraphColoring;
using mlir::tt::ttl::ExactInterferenceGraphColoringStatus;
using mlir::tt::ttl::ExactInterferenceGraphWeightColoring;
using mlir::tt::ttl::ExactInterferenceGraphWeightStatus;
using mlir::tt::ttl::InterferenceGraph;
using mlir::tt::ttl::InterferenceGraphColorLimitResult;
using mlir::tt::ttl::InterferenceGraphColorLimitStatus;
using mlir::tt::ttl::InterferenceGraphWeightLimitResult;
using mlir::tt::ttl::SRAMAllocationProblem;
using mlir::tt::ttl::SRAMAllocationSolution;
using mlir::tt::ttl::SRAMAllocator;
using mlir::tt::ttl::SRAMAllocatorOptions;

constexpr uint64_t kUnlimitedSearchStates =
    std::numeric_limits<uint64_t>::max();

/// Verifies the target query and default system descriptor use the same
/// architecture capacities.
static bool verifyTargetDFBIndexCapacities() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  struct ExpectedCapacity {
    mlir::tt::ttcore::Arch arch;
    int32_t indexCount;
  };
  constexpr ExpectedCapacity expectedCapacities[] = {
      {mlir::tt::ttcore::Arch::WormholeB0, 32},
      {mlir::tt::ttcore::Arch::Blackhole, 64},
      {mlir::tt::ttcore::Arch::Quasar, 32},
  };
  llvm::SmallVector<int32_t, 4> observedCapacities;
  for (const ExpectedCapacity &expected : expectedCapacities) {
    module->getOperation()->setAttr(
        mlir::tt::kTargetArchAttrName,
        mlir::tt::ttcore::ArchAttr::get(&context, expected.arch));
    std::string failureReason;
    mlir::FailureOr<mlir::tt::TargetDFBIndexCapacity> capacity =
        mlir::tt::resolveTargetDFBIndexCapacity(*module, failureReason);
    if (mlir::failed(capacity) || capacity->indexCount != expected.indexCount ||
        !capacity->contains(expected.indexCount - 1) ||
        capacity->contains(expected.indexCount)) {
      llvm::errs() << "target DFB-index capacity mismatch\n";
      return false;
    }
    observedCapacities.push_back(capacity->indexCount);
  }
  module->getOperation()->removeAttr(mlir::tt::kTargetArchAttrName);
  std::string failureReason;
  mlir::FailureOr<mlir::tt::TargetDFBIndexCapacity> missingTargetCapacity =
      mlir::tt::resolveTargetDFBIndexCapacity(*module, failureReason);
  if (mlir::failed(missingTargetCapacity) ||
      missingTargetCapacity->indexCount != 32) {
    llvm::errs() << "missing-target DFB-index capacity mismatch\n";
    return false;
  }
  observedCapacities.push_back(missingTargetCapacity->indexCount);

  module->getOperation()->setAttr(mlir::tt::kTargetArchAttrName,
                                  mlir::StringAttr::get(&context, "blackhole"));
  mlir::FailureOr<mlir::tt::TargetDFBIndexCapacity> malformedTargetCapacity =
      mlir::tt::resolveTargetDFBIndexCapacity(*module, failureReason);
  if (mlir::succeeded(malformedTargetCapacity) ||
      failureReason != "ttl.target_arch must be a #ttcore.arch attribute") {
    llvm::errs() << "malformed target architecture was not rejected\n";
    return false;
  }

  struct ExpectedSystemDescCapacity {
    mlir::tt::ttcore::Arch arch;
    unsigned numCBs;
    llvm::StringRef name;
  };
  const ExpectedSystemDescCapacity expectedSystemDescCapacities[] = {
      {mlir::tt::ttcore::Arch::Blackhole, 64, "Blackhole"},
      {mlir::tt::ttcore::Arch::WormholeB0, 32, "Wormhole B0"},
  };
  llvm::SmallVector<unsigned, 2> observedSystemDescCapacities;
  for (const ExpectedSystemDescCapacity &expected :
       expectedSystemDescCapacities) {
    mlir::tt::ttcore::SystemDescAttr systemDesc =
        mlir::tt::ttcore::SystemDescAttr::getDefault(&context, expected.arch);
    if (systemDesc.getChipDescs().size() != 1 ||
        systemDesc.getChipDescs().front().getNumCBs() != expected.numCBs) {
      llvm::errs() << "default " << expected.name
                   << " system descriptor reports the wrong DFB-index "
                      "capacity\n";
      return false;
    }
    observedSystemDescCapacities.push_back(
        systemDesc.getChipDescs().front().getNumCBs());

    mlir::OwningOpRef<mlir::ModuleOp> deviceModule =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    deviceModule->getOperation()->setAttr(
        mlir::tt::ttcore::SystemDescAttr::name, systemDesc);
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToStart(deviceModule->getBody());
    mlir::tt::ttcore::DeviceOp::create(
        builder, deviceModule->getLoc(),
        mlir::tt::ttcore::getDefaultDeviceName(),
        /*symVisibility=*/nullptr,
        mlir::tt::ttcore::DeviceAttr::get(&context, systemDesc));
    mlir::FailureOr<mlir::tt::TargetDFBIndexCapacity> deviceCapacity =
        mlir::tt::resolveTargetDFBIndexCapacity(*deviceModule, failureReason);
    if (mlir::failed(deviceCapacity) ||
        deviceCapacity->indexCount != static_cast<int32_t>(expected.numCBs)) {
      llvm::errs() << "default device target resolution mismatch\n";
      return false;
    }
  }

  llvm::outs() << "target_capacities=";
  llvm::interleaveComma(observedCapacities, llvm::outs());
  llvm::outs() << "\nsystem_desc_num_cbs=";
  llvm::interleaveComma(observedSystemDescCapacities, llvm::outs());
  llvm::outs() << "\n";
  return true;
}

/// Enumerates index choices in vertex order so expected results do not depend
/// on the production search order.
static bool oracleCanColor(const InterferenceGraph &graph,
                           llvm::MutableArrayRef<unsigned> colors,
                           unsigned vertex, unsigned colorCount) {
  if (vertex == graph.size()) {
    return true;
  }
  for (unsigned color = 0; color < colorCount; ++color) {
    bool permitted = true;
    for (unsigned otherVertex = 0; otherVertex < vertex; ++otherVertex) {
      if (colors[otherVertex] == color &&
          graph.interferes(vertex, otherVertex)) {
        permitted = false;
        break;
      }
    }
    if (!permitted) {
      continue;
    }
    colors[vertex] = color;
    if (oracleCanColor(graph, colors, vertex + 1, colorCount)) {
      return true;
    }
  }
  colors[vertex] = std::numeric_limits<unsigned>::max();
  return false;
}

/// Finds the minimum feasible physical-index count by exhaustive enumeration.
static unsigned oracleMinimumIndexCount(const InterferenceGraph &graph) {
  if (graph.size() == 0) {
    return 0;
  }
  llvm::SmallVector<unsigned> colors(graph.size(),
                                     std::numeric_limits<unsigned>::max());
  for (unsigned colorCount = 1; colorCount <= graph.size(); ++colorCount) {
    std::fill(colors.begin(), colors.end(),
              std::numeric_limits<unsigned>::max());
    if (oracleCanColor(graph, colors, /*vertex=*/0, colorCount)) {
      return colorCount;
    }
  }
  llvm_unreachable("every finite graph is colorable");
}

/// Decodes a compact edge mask used to enumerate every labeled small graph.
static InterferenceGraph buildGraph(unsigned vertexCount, uint64_t edgeMask) {
  InterferenceGraph graph(vertexCount);
  unsigned edgeIndex = 0;
  for (unsigned lhsVertex = 0; lhsVertex < vertexCount; ++lhsVertex) {
    for (unsigned rhsVertex = lhsVertex + 1; rhsVertex < vertexCount;
         ++rhsVertex, ++edgeIndex) {
      if (edgeMask & (uint64_t{1} << edgeIndex)) {
        graph.addInterference(lhsVertex, rhsVertex);
      }
    }
  }
  return graph;
}

/// Checks the solver's assignment independently of its completion status.
static bool verifyColoring(const InterferenceGraph &graph,
                           llvm::ArrayRef<unsigned> colors,
                           unsigned colorCount) {
  if (colors.size() != graph.size()) {
    return false;
  }
  for (unsigned lhsVertex = 0; lhsVertex < graph.size(); ++lhsVertex) {
    if (colors[lhsVertex] >= colorCount) {
      return false;
    }
    for (unsigned rhsVertex = lhsVertex + 1; rhsVertex < graph.size();
         ++rhsVertex) {
      if (graph.interferes(lhsVertex, rhsVertex) &&
          colors[lhsVertex] == colors[rhsVertex]) {
        return false;
      }
    }
  }
  return true;
}

/// Enumerates every legal assignment and returns its minimum sum of maximum
/// vertex weight per used color.
static void oracleMinimumAllocationWeight(
    const InterferenceGraph &graph, llvm::ArrayRef<uint64_t> vertexWeights,
    llvm::MutableArrayRef<unsigned> colors,
    llvm::MutableArrayRef<uint64_t> colorWeights, unsigned vertex,
    unsigned colorLimit, uint64_t currentWeight, uint64_t &minimumWeight) {
  if (vertex == graph.size()) {
    minimumWeight = std::min(minimumWeight, currentWeight);
    return;
  }
  for (unsigned color = 0; color < colorLimit; ++color) {
    bool permitted = true;
    for (unsigned otherVertex = 0; otherVertex < vertex; ++otherVertex) {
      if (colors[otherVertex] == color &&
          graph.interferes(vertex, otherVertex)) {
        permitted = false;
        break;
      }
    }
    if (!permitted) {
      continue;
    }
    uint64_t previousColorWeight = colorWeights[color];
    uint64_t updatedColorWeight =
        std::max(previousColorWeight, vertexWeights[vertex]);
    uint64_t addedWeight = updatedColorWeight - previousColorWeight;
    if (addedWeight > minimumWeight - currentWeight) {
      continue;
    }
    colors[vertex] = color;
    colorWeights[color] = updatedColorWeight;
    oracleMinimumAllocationWeight(graph, vertexWeights, colors, colorWeights,
                                  vertex + 1, colorLimit,
                                  currentWeight + addedWeight, minimumWeight);
    colors[vertex] = std::numeric_limits<unsigned>::max();
    colorWeights[color] = previousColorWeight;
  }
}

static uint64_t
oracleMinimumAllocationWeight(const InterferenceGraph &graph,
                              llvm::ArrayRef<uint64_t> vertexWeights,
                              unsigned colorLimit) {
  llvm::SmallVector<unsigned> colors(graph.size(),
                                     std::numeric_limits<unsigned>::max());
  llvm::SmallVector<uint64_t> colorWeights(colorLimit);
  uint64_t minimumWeight = std::numeric_limits<uint64_t>::max();
  oracleMinimumAllocationWeight(graph, vertexWeights, colors, colorWeights,
                                /*vertex=*/0, colorLimit,
                                /*currentWeight=*/0, minimumWeight);
  return minimumWeight;
}

/// Compares minimum and fixed-limit production queries against exhaustive
/// expected results for every graph with at most six vertices.
static bool compareProductionSolverWithOracle() {
  uint64_t checkedGraphCount = 0;
  for (unsigned vertexCount = 0; vertexCount <= 6; ++vertexCount) {
    unsigned edgeCount = vertexCount * (vertexCount - 1) / 2;
    uint64_t graphCount = uint64_t{1} << edgeCount;
    for (uint64_t edgeMask = 0; edgeMask < graphCount; ++edgeMask) {
      InterferenceGraph graph = buildGraph(vertexCount, edgeMask);
      unsigned oracleCount = oracleMinimumIndexCount(graph);
      ExactInterferenceGraphColoring production =
          mlir::tt::ttl::colorInterferenceGraphExactly(graph,
                                                       kUnlimitedSearchStates);
      if (!production.isOptimal() || production.colorCount != oracleCount ||
          !verifyColoring(graph, production.colors, production.colorCount)) {
        llvm::errs() << "solver mismatch: vertices=" << vertexCount
                     << " edge_mask=" << edgeMask << " oracle=" << oracleCount
                     << " production=" << production.colorCount << "\n";
        return false;
      }
      for (unsigned colorLimit = 0; colorLimit <= vertexCount; ++colorLimit) {
        InterferenceGraphColorLimitResult fit =
            mlir::tt::ttl::colorInterferenceGraphWithColorLimitExactly(
                graph, colorLimit, kUnlimitedSearchStates);
        bool expectedFeasible = oracleCount <= colorLimit;
        bool resultFeasible =
            fit.status == InterferenceGraphColorLimitStatus::Feasible;
        if (resultFeasible != expectedFeasible ||
            (resultFeasible &&
             !verifyColoring(graph, fit.colors, fit.colorCount)) ||
            (!resultFeasible &&
             fit.status != InterferenceGraphColorLimitStatus::Infeasible)) {
          llvm::errs() << "fixed-limit mismatch: vertices=" << vertexCount
                       << " edge_mask=" << edgeMask
                       << " color_limit=" << colorLimit
                       << " oracle=" << oracleCount << "\n";
          return false;
        }
      }
      ++checkedGraphCount;
    }
  }
  llvm::outs() << "solver_graphs=" << checkedGraphCount << "\n";
  return true;
}

/// Confirms that a fixed-limit exact check repairs the adversarial first-fit
/// ordering used by the positive 32-index capacity test.
static bool verifyGreedyCapacityReproducer() {
  InterferenceGraph pathGraph(4);
  pathGraph.addInterference(0, 1);
  pathGraph.addInterference(1, 2);
  pathGraph.addInterference(2, 3);
  llvm::SmallVector<unsigned> adversarialOrder = {0, 3, 1, 2};
  llvm::SmallVector<unsigned> greedy =
      mlir::tt::ttl::colorInterferenceGraphFirstFit(pathGraph,
                                                    adversarialOrder);
  unsigned greedyCount = *std::max_element(greedy.begin(), greedy.end()) + 1;
  ExactInterferenceGraphColoring exact =
      mlir::tt::ttl::colorInterferenceGraphExactly(pathGraph,
                                                   kUnlimitedSearchStates);
  if (greedyCount != 3 || exact.colorCount != 2) {
    llvm::errs() << "four-vertex reproducer mismatch\n";
    return false;
  }

  InterferenceGraph capacityGraph(34);
  for (unsigned singleton = 0; singleton < 30; ++singleton) {
    for (unsigned otherVertex = singleton + 1; otherVertex < 34;
         ++otherVertex) {
      capacityGraph.addInterference(singleton, otherVertex);
    }
  }
  // A, D, B, C processing makes first-fit use three indices for the conflict
  // chain A-B-C-D after the 30 pairwise-conflicting DFBs.
  capacityGraph.addInterference(30, 32);
  capacityGraph.addInterference(32, 33);
  capacityGraph.addInterference(33, 31);
  mlir::tt::ttl::InterferenceGraphColoringBounds capacityBounds =
      mlir::tt::ttl::computeInterferenceGraphColoringBounds(capacityGraph);
  InterferenceGraphColorLimitResult capacityFit =
      mlir::tt::ttl::colorInterferenceGraphWithColorLimitExactly(
          capacityGraph, /*colorLimit=*/32, kUnlimitedSearchStates);
  if (capacityBounds.colorCount != 33 || !capacityFit.isFeasible() ||
      capacityFit.colorCount != 32) {
    llvm::errs()
        << "capacity reproducer expected first-fit 33 and fixed-limit 32, got "
        << capacityBounds.colorCount << " and " << capacityFit.colorCount
        << "\n";
    return false;
  }
  llvm::outs() << "capacity_reproducer=32\n";
  llvm::outs() << "capacity_search_states=" << capacityFit.exploredStateCount
               << "\n";
  return true;
}

/// Distinguishes an exhausted search budget from a proof of infeasibility.
static bool verifySearchLimitOutcome() {
  InterferenceGraph pathGraph(4);
  pathGraph.addInterference(0, 1);
  pathGraph.addInterference(1, 2);
  pathGraph.addInterference(2, 3);
  ExactInterferenceGraphColoring limited =
      mlir::tt::ttl::colorInterferenceGraphExactly(pathGraph,
                                                   /*searchStateLimit=*/1);
  if (limited.status !=
          ExactInterferenceGraphColoringStatus::SearchLimitReached ||
      !limited.colors.empty() || limited.colorCount != 0 ||
      limited.exploredStateCount != 1) {
    llvm::errs() << "bounded search did not report an inconclusive result\n";
    return false;
  }
  InterferenceGraphColorLimitResult fixedLimit =
      mlir::tt::ttl::colorInterferenceGraphWithColorLimitExactly(
          pathGraph, /*colorLimit=*/2, /*searchStateLimit=*/1);
  if (fixedLimit.status !=
          InterferenceGraphColorLimitStatus::SearchLimitReached ||
      !fixedLimit.colors.empty() || fixedLimit.colorCount != 0 ||
      fixedLimit.exploredStateCount != 1) {
    llvm::errs()
        << "fixed-limit search did not report an inconclusive result\n";
    return false;
  }
  llvm::outs() << "bounded_search_states=" << limited.exploredStateCount
               << "\n";
  return true;
}

/// Preserves small pairwise-conflicting sets while increasing the required
/// index count, producing cases where the lower bound is intentionally weak.
static InterferenceGraph buildMycielskian(const InterferenceGraph &graph) {
  unsigned vertexCount = graph.size();
  InterferenceGraph result(2 * vertexCount + 1);
  for (unsigned lhsVertex = 0; lhsVertex < vertexCount; ++lhsVertex) {
    for (unsigned rhsVertex = lhsVertex + 1; rhsVertex < vertexCount;
         ++rhsVertex) {
      if (!graph.interferes(lhsVertex, rhsVertex)) {
        continue;
      }
      result.addInterference(lhsVertex, rhsVertex);
      result.addInterference(lhsVertex, vertexCount + rhsVertex);
      result.addInterference(rhsVertex, vertexCount + lhsVertex);
    }
  }
  unsigned apex = 2 * vertexCount;
  for (unsigned clone = vertexCount; clone < 2 * vertexCount; ++clone) {
    result.addInterference(clone, apex);
  }
  return result;
}

/// Demonstrates why asking whether five indices fit is cheaper than proving
/// the minimum when the pairwise-conflict lower bound is weak.
static bool verifyFixedLimitAvoidsMinimumSearch() {
  InterferenceGraph completeGraph(2);
  completeGraph.addInterference(0, 1);
  InterferenceGraph cycleFive = buildMycielskian(completeGraph);
  mlir::tt::ttl::InterferenceGraphColoringBounds cycleBounds =
      mlir::tt::ttl::computeInterferenceGraphColoringBounds(cycleFive);
  InterferenceGraphColorLimitResult cycleTwoColors =
      mlir::tt::ttl::colorInterferenceGraphWithColorLimitExactly(
          cycleFive, /*colorLimit=*/2, kUnlimitedSearchStates);
  InterferenceGraphColorLimitResult cycleThreeColors =
      mlir::tt::ttl::colorInterferenceGraphWithColorLimitExactly(
          cycleFive, /*colorLimit=*/3, kUnlimitedSearchStates);
  if (cycleBounds.pairwiseConflictLowerBound != 2 ||
      cycleTwoColors.status != InterferenceGraphColorLimitStatus::Infeasible ||
      !cycleThreeColors.isFeasible()) {
    llvm::errs() << "five-cycle allocation witness mismatch\n";
    return false;
  }

  InterferenceGraph fourChromatic = buildMycielskian(cycleFive);
  InterferenceGraph fiveChromatic = buildMycielskian(fourChromatic);
  constexpr uint64_t kComparisonSearchStates = 100;
  InterferenceGraphColorLimitResult fixedLimit =
      mlir::tt::ttl::colorInterferenceGraphWithColorLimitExactly(
          fiveChromatic, /*colorLimit=*/5, kComparisonSearchStates);
  ExactInterferenceGraphColoring minimum =
      mlir::tt::ttl::colorInterferenceGraphExactly(fiveChromatic,
                                                   kComparisonSearchStates);
  if (!fixedLimit.isFeasible() ||
      minimum.status !=
          ExactInterferenceGraphColoringStatus::SearchLimitReached) {
    llvm::errs() << "fixed-limit and minimum-search comparison mismatch: fixed="
                 << static_cast<unsigned>(fixedLimit.status)
                 << " minimum=" << static_cast<unsigned>(minimum.status)
                 << " fixed_states=" << fixedLimit.exploredStateCount
                 << " minimum_states=" << minimum.exploredStateCount << "\n";
    return false;
  }
  llvm::outs() << "fixed_limit_states=" << fixedLimit.exploredStateCount << "\n"
               << "minimum_proof_states=" << minimum.exploredStateCount << "\n";
  return true;
}

/// Verifies that disconnected epoch components coordinate color permutations
/// to minimize the combined maximum allocation per physical index.
static bool verifyWeightedColoringAcrossComponents() {
  InterferenceGraph graph(4);
  graph.addInterference(0, 1);
  graph.addInterference(2, 3);
  constexpr std::uint64_t weights[] = {100, 1, 1, 100};
  InterferenceGraphWeightLimitResult exactFit =
      mlir::tt::ttl::colorInterferenceGraphWithinWeightLimitExactly(
          graph, weights, /*colorLimit=*/2, /*weightLimit=*/101,
          kUnlimitedSearchStates);
  InterferenceGraphWeightLimitResult belowFit =
      mlir::tt::ttl::colorInterferenceGraphWithinWeightLimitExactly(
          graph, weights, /*colorLimit=*/2, /*weightLimit=*/100,
          kUnlimitedSearchStates);
  InterferenceGraphWeightLimitResult limited =
      mlir::tt::ttl::colorInterferenceGraphWithinWeightLimitExactly(
          graph, weights, /*colorLimit=*/2, /*weightLimit=*/101,
          /*searchStateLimit=*/1);
  constexpr unsigned firstFitColors[] = {0, 1, 0, 1};
  ExactInterferenceGraphWeightColoring minimum =
      mlir::tt::ttl::colorInterferenceGraphMinimumWeightExactly(
          graph, weights, /*colorLimit=*/2, firstFitColors,
          kUnlimitedSearchStates);
  ExactInterferenceGraphWeightColoring minimumLimited =
      mlir::tt::ttl::colorInterferenceGraphMinimumWeightExactly(
          graph, weights, /*colorLimit=*/2, firstFitColors,
          /*searchStateLimit=*/1);
  if (!exactFit.isFeasible() || exactFit.colorCount != 2 ||
      exactFit.colors[0] != exactFit.colors[3] ||
      exactFit.colors[1] != exactFit.colors[2] ||
      belowFit.status != InterferenceGraphColorLimitStatus::Infeasible ||
      limited.status != InterferenceGraphColorLimitStatus::SearchLimitReached ||
      !minimum.isOptimal() || minimum.allocationWeight != 101 ||
      minimum.colors[0] != minimum.colors[3] ||
      minimum.colors[1] != minimum.colors[2] ||
      minimumLimited.status !=
          ExactInterferenceGraphWeightStatus::SearchLimitReached) {
    llvm::errs() << "weighted coloring witness mismatch\n";
    return false;
  }
  return true;
}

/// Compares the weighted optimizer with exhaustive enumeration for every
/// four-vertex graph, small weight tuple, and feasible color limit.
static bool compareWeightedSolverWithOracle() {
  constexpr unsigned kVertexCount = 4;
  constexpr unsigned kGraphCount = 1U << 6;
  constexpr unsigned kWeightValueCount = 3;
  constexpr unsigned kWeightTupleCount = kWeightValueCount * kWeightValueCount *
                                         kWeightValueCount * kWeightValueCount;
  uint64_t checkedCaseCount = 0;
  for (unsigned edgeMask = 0; edgeMask < kGraphCount; ++edgeMask) {
    InterferenceGraph graph = buildGraph(kVertexCount, edgeMask);
    unsigned minimumColorCount = oracleMinimumIndexCount(graph);
    for (unsigned encodedWeights = 0; encodedWeights < kWeightTupleCount;
         ++encodedWeights) {
      unsigned remainingWeights = encodedWeights;
      llvm::SmallVector<uint64_t> weights;
      for (unsigned vertex = 0; vertex < kVertexCount; ++vertex) {
        weights.push_back(1 + remainingWeights % kWeightValueCount);
        remainingWeights /= kWeightValueCount;
      }
      for (unsigned colorLimit = minimumColorCount; colorLimit <= kVertexCount;
           ++colorLimit) {
        InterferenceGraphColorLimitResult initial =
            mlir::tt::ttl::colorInterferenceGraphWithColorLimitExactly(
                graph, colorLimit, kUnlimitedSearchStates);
        ExactInterferenceGraphWeightColoring production =
            mlir::tt::ttl::colorInterferenceGraphMinimumWeightExactly(
                graph, weights, colorLimit, initial.colors,
                kUnlimitedSearchStates);
        uint64_t expected =
            oracleMinimumAllocationWeight(graph, weights, colorLimit);
        if (!initial.isFeasible() || !production.isOptimal() ||
            production.allocationWeight != expected ||
            !verifyColoring(graph, production.colors, production.colorCount)) {
          llvm::errs() << "weighted solver mismatch: edge_mask=" << edgeMask
                       << " encoded_weights=" << encodedWeights
                       << " color_limit=" << colorLimit
                       << " oracle=" << expected
                       << " production=" << production.allocationWeight << "\n";
          return false;
        }
        ++checkedCaseCount;
      }
    }
  }

  InterferenceGraph independentPair(2);
  constexpr uint64_t overflowingWeights[] = {
      std::numeric_limits<uint64_t>::max(),
      std::numeric_limits<uint64_t>::max()};
  constexpr unsigned separateColors[] = {0, 1};
  ExactInterferenceGraphWeightColoring representable =
      mlir::tt::ttl::colorInterferenceGraphMinimumWeightExactly(
          independentPair, overflowingWeights, /*colorLimit=*/2, separateColors,
          kUnlimitedSearchStates);
  independentPair.addInterference(0, 1);
  ExactInterferenceGraphWeightColoring overflow =
      mlir::tt::ttl::colorInterferenceGraphMinimumWeightExactly(
          independentPair, overflowingWeights, /*colorLimit=*/2, separateColors,
          kUnlimitedSearchStates);
  if (!representable.isOptimal() ||
      representable.allocationWeight != std::numeric_limits<uint64_t>::max() ||
      overflow.status !=
          ExactInterferenceGraphWeightStatus::AllocationWeightOverflow) {
    llvm::errs() << "weighted overflow handling mismatch\n";
    return false;
  }
  llvm::outs() << "weighted_solver_cases=" << checkedCaseCount << "\n";
  return true;
}

static void oracleMinimumArenaBytes(
    const InterferenceGraph &graph, llvm::ArrayRef<uint64_t> regionBytes,
    uint64_t alignmentBytes, uint64_t payloadBaseOffset,
    llvm::MutableArrayRef<uint64_t> offsets, unsigned regionIndex,
    uint64_t highWaterMark, uint64_t &minimumArenaBytes) {
  if (regionIndex == graph.size()) {
    minimumArenaBytes = std::min(minimumArenaBytes, highWaterMark);
    return;
  }

  uint64_t regionSize = regionBytes[regionIndex];
  for (uint64_t offset = payloadBaseOffset;
       offset < minimumArenaBytes - regionSize; offset += alignmentBytes) {
    uint64_t regionEnd = offset + regionSize;
    bool overlapsConflict = false;
    for (unsigned previousIndex = 0; previousIndex < regionIndex;
         ++previousIndex) {
      if (!graph.interferes(regionIndex, previousIndex)) {
        continue;
      }
      uint64_t previousEnd =
          offsets[previousIndex] + regionBytes[previousIndex];
      if (regionEnd > offsets[previousIndex] && previousEnd > offset) {
        overlapsConflict = true;
        break;
      }
    }
    if (overlapsConflict) {
      continue;
    }
    offsets[regionIndex] = offset;
    oracleMinimumArenaBytes(
        graph, regionBytes, alignmentBytes, payloadBaseOffset, offsets,
        regionIndex + 1, std::max(highWaterMark, regionEnd), minimumArenaBytes);
  }
}

static uint64_t oracleMinimumArenaBytes(const InterferenceGraph &graph,
                                        llvm::ArrayRef<uint64_t> regionBytes,
                                        uint64_t alignmentBytes,
                                        uint64_t payloadBaseOffset) {
  if (regionBytes.empty()) {
    return 0;
  }
  uint64_t minimumArenaBytes = payloadBaseOffset;
  for (uint64_t regionSize : regionBytes) {
    minimumArenaBytes += regionSize;
  }
  llvm::SmallVector<uint64_t> offsets(graph.size());
  oracleMinimumArenaBytes(graph, regionBytes, alignmentBytes, payloadBaseOffset,
                          offsets, /*regionIndex=*/0,
                          /*highWaterMark=*/payloadBaseOffset,
                          minimumArenaBytes);
  return minimumArenaBytes;
}

struct PlacementQuality {
  uint64_t suboptimalCount = 0;
  uint64_t excessUnits = 0;
  uint64_t maximumExcessUnits = 0;
  uint64_t optimalUnits = 0;
  uint64_t allocatedUnits = 0;
  uint64_t suboptimalOptimalUnits = 0;
  uint64_t suboptimalAllocatedUnits = 0;
  uint64_t worstOptimalUnits = 0;
  uint64_t worstAllocatedUnits = 0;

  void record(uint64_t minimumUnits, uint64_t placementUnits) {
    assert(minimumUnits > 0 && placementUnits >= minimumUnits);
    uint64_t currentExcessUnits = placementUnits - minimumUnits;
    optimalUnits += minimumUnits;
    allocatedUnits += placementUnits;
    excessUnits += currentExcessUnits;
    maximumExcessUnits = std::max(maximumExcessUnits, currentExcessUnits);
    if (currentExcessUnits == 0) {
      return;
    }

    ++suboptimalCount;
    suboptimalOptimalUnits += minimumUnits;
    suboptimalAllocatedUnits += placementUnits;
    if (worstAllocatedUnits == 0 || minimumUnits * worstAllocatedUnits <
                                        worstOptimalUnits * placementUnits) {
      worstOptimalUnits = minimumUnits;
      worstAllocatedUnits = placementUnits;
    }
  }
};

static uint64_t getEfficiencyBasisPoints(uint64_t optimalUnits,
                                         uint64_t allocatedUnits) {
  constexpr uint64_t kBasisPointScale = 10000;
  if (allocatedUnits == 0) {
    assert(optimalUnits == 0);
    return kBasisPointScale;
  }
  return optimalUnits * kBasisPointScale / allocatedUnits;
}

static bool compareMultiOrderLargeGraphs() {
  std::string reason;
  SRAMAllocatorOptions options{kUnlimitedSearchStates};
  auto first = mlir::tt::ttl::createSRAMAllocator(
      mlir::tt::ttl::kFirstFitDecreasingSRAMAllocator, options, reason);
  auto multi = mlir::tt::ttl::createSRAMAllocator(
      mlir::tt::ttl::kMultiOrderDecreasingSRAMAllocator, options, reason);
  if (mlir::failed(first) || mlir::failed(multi)) {
    return false;
  }
  uint64_t cases = 0;
  for (unsigned regionCount : {8U, 32U, 96U, 256U, 512U}) {
    for (unsigned seed = 0; seed < 32; ++seed) {
      SRAMAllocationProblem problem;
      problem.alignmentBytes = seed % 2 == 0 ? 32 : 64;
      problem.payloadBaseOffset = problem.alignmentBytes * 3;
      problem.budgetBytes = problem.payloadBaseOffset;
      problem.conflicts = InterferenceGraph(regionCount);
      for (unsigned region = 0; region < regionCount; ++region) {
        uint64_t size =
            problem.alignmentBytes *
            (seed % 4 == 0 ? 1 : 1 + (region * 17 + seed * 13) % 31);
        problem.regionBytes.push_back(size);
        problem.budgetBytes += size;
        for (unsigned other = 0; other < region; ++other) {
          unsigned hash = (region * 2654435761U) ^ (other * 2246822519U) ^
                          (seed * 3266489917U);
          if (hash % 100 < (seed % 5) * 25) {
            problem.conflicts.addInterference(region, other);
          }
        }
      }
      std::optional<unsigned> failedRegion;
      auto stable = (*first)->allocate(problem, failedRegion, reason);
      auto combined = (*multi)->allocate(problem, failedRegion, reason);
      if (mlir::failed(stable) || mlir::failed(combined) ||
          combined->arenaBytes > stable->arenaBytes ||
          (combined->arenaBytes == stable->arenaBytes &&
           combined->offsets != stable->offsets)) {
        llvm::errs() << "large multi-order regression: " << regionCount << ", "
                     << seed << "\n";
        return false;
      }
      problem.budgetBytes = combined->arenaBytes;
      auto repeated = (*multi)->allocate(problem, failedRegion, reason);
      if (mlir::failed(repeated) || repeated->offsets != combined->offsets) {
        llvm::errs() << "multi-order exact-budget determinism failed\n";
        return false;
      }
      ++cases;
    }
  }
  llvm::outs() << "sram_multi_order_large_cases=" << cases << "\n";
  return true;
}

// Observe validation ordering and exercise common validation of strategy
// results.
class DomainTestAllocator final : public SRAMAllocator {
public:
  mutable unsigned calls = 0;
  bool invalidPlacement = false;
  llvm::StringRef getName() const override { return "domain-test"; }

private:
  mlir::FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &problem,
               std::string &failureReason) const override {
    ++calls;
    SRAMAllocationSolution result;
    uint64_t nextOffset = problem.payloadBaseOffset;
    for (uint64_t size : problem.regionBytes) {
      result.offsets.push_back(invalidPlacement ? 0 : nextOffset);
      nextOffset += size;
    }
    result.arenaBytes = problem.regionBytes.empty() ? 0 : nextOffset;
    return result;
  }

  mlir::FailureOr<mlir::tt::ttl::SRAMLocationAllocationSolution>
  allocateLocationsImpl(
      const mlir::tt::ttl::SRAMLocationAllocationProblem &problem,
      std::string &failureReason) const override {
    failureReason = "location allocation is not used by this test allocator";
    return mlir::failure();
  }
};

enum class InvalidLocationSolution {
  None,
  WrongSize,
  InvalidOffset,
  FixedOffset,
  UnequalOffsets,
  Overlap,
  WrongHighWater,
};

class LocationTestAllocator final : public SRAMAllocator {
public:
  InvalidLocationSolution invalidSolution = InvalidLocationSolution::None;
  llvm::StringRef getName() const override { return "location-test"; }

private:
  mlir::FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &problem,
               std::string &failureReason) const override {
    failureReason = "scalar allocation is not used by this test allocator";
    return mlir::failure();
  }

  mlir::FailureOr<mlir::tt::ttl::SRAMLocationAllocationSolution>
  allocateLocationsImpl(
      const mlir::tt::ttl::SRAMLocationAllocationProblem &problem,
      std::string &failureReason) const override {
    mlir::tt::ttl::SRAMLocationAllocationSolution solution{
        {64, 64, 128, 192, 256}, {320, 256}};
    switch (invalidSolution) {
    case InvalidLocationSolution::None:
      break;
    case InvalidLocationSolution::WrongSize:
      solution.offsets.pop_back();
      break;
    case InvalidLocationSolution::InvalidOffset:
      solution.offsets[1] = 0;
      break;
    case InvalidLocationSolution::FixedOffset:
      solution.offsets[4] = 192;
      break;
    case InvalidLocationSolution::UnequalOffsets:
      solution.offsets[0] = 0;
      break;
    case InvalidLocationSolution::Overlap:
      solution.offsets[2] = 64;
      break;
    case InvalidLocationSolution::WrongHighWater:
      solution.highWaterBytes[0] = 319;
      break;
    }
    return solution;
  }
};

static bool verifySRAMAllocationDomains() {
  using mlir::tt::ttl::SRAMAllocationDomainFailure;
  using mlir::tt::ttl::SRAMAllocationDomainProblem;
  SRAMAllocationProblem large{{64, 128}, InterferenceGraph(2), 64, 64, 256};
  large.conflicts.addInterference(0, 1);
  SRAMAllocationProblem small{{32}, InterferenceGraph(1), 32, 32, 64};
  SRAMAllocationProblem controlOnly{{}, InterferenceGraph(0), 64, 64, 64};
  llvm::SmallVector<SRAMAllocationDomainProblem> domains{
      {large, {91, 7}}, {small, {91}}, {controlOnly, {}}};
  uint64_t cases = 0;
  for (llvm::StringRef strategy :
       {mlir::tt::ttl::kFirstFitDecreasingSRAMAllocator,
        mlir::tt::ttl::kBestFitDecreasingSRAMAllocator,
        mlir::tt::ttl::kMultiOrderDecreasingSRAMAllocator,
        mlir::tt::ttl::kExactSRAMAllocator}) {
    std::string reason;
    auto allocator = mlir::tt::ttl::createSRAMAllocator(
        strategy, {kUnlimitedSearchStates}, reason);
    if (mlir::failed(allocator)) {
      return false;
    }
    SRAMAllocationDomainFailure failureDetail;
    auto result = (*allocator)->allocateDomains(domains, failureDetail);
    if (mlir::failed(result) || result->size() != 3 ||
        (*result)[0].arenaBytes != 256 || (*result)[1].arenaBytes != 64 ||
        (*result)[2].arenaBytes != 64 || !(*result)[2].placements.empty() ||
        (*result)[0].placements[0].storageIndex != 91 ||
        (*result)[0].placements[1].storageIndex != 7 ||
        (*result)[1].placements[0].storageIndex != 91 ||
        (*result)[1].placements[0].offset != 32) {
      llvm::errs() << "domain ownership or reservation mismatch: " << strategy
                   << "\n";
      return false;
    }
    std::optional<unsigned> failedRegion;
    auto uniform = (*allocator)->allocate(large, failedRegion, reason);
    if (mlir::failed(uniform) ||
        uniform->arenaBytes != (*result)[0].arenaBytes) {
      return false;
    }
    for (unsigned region = 0; region < large.regionBytes.size(); ++region) {
      if (uniform->offsets[region] != (*result)[0].placements[region].offset) {
        return false;
      }
    }
    auto empty = (*allocator)->allocateDomains({}, failureDetail);
    if (mlir::failed(empty) || !empty->empty()) {
      return false;
    }
    auto invalid = domains;
    invalid[1].allocation.budgetBytes = 32;
    auto rejected = (*allocator)->allocateDomains(invalid, failureDetail);
    if (mlir::succeeded(rejected) || failureDetail.domainIndex != 1 ||
        failureDetail.reason.empty()) {
      return false;
    }
    ++cases;
  }

  DomainTestAllocator allocator;
  SRAMAllocationDomainFailure failureDetail;
  for (unsigned invalidKind = 0; invalidKind < 4; ++invalidKind) {
    auto invalid = domains;
    if (invalidKind == 0) {
      invalid[1].storageIndices.clear();
    } else if (invalidKind == 1) {
      invalid[1] = {large, {91, 91}};
    } else if (invalidKind == 2) {
      invalid[1].allocation.alignmentBytes = 3;
    } else {
      invalid[1].allocation.conflicts = InterferenceGraph(0);
    }
    auto rejected = allocator.allocateDomains(invalid, failureDetail);
    if (mlir::succeeded(rejected) || allocator.calls != 0 ||
        failureDetail.domainIndex != 1 || failureDetail.reason.empty()) {
      llvm::errs() << "invalid domain reached placement\n";
      return false;
    }
    ++cases;
  }
  auto invalid = domains;
  invalid[1].allocation.budgetBytes = 32;
  auto rejected = allocator.allocateDomains(invalid, failureDetail);
  if (mlir::succeeded(rejected) || allocator.calls != 2 ||
      failureDetail.domainIndex != 1 || failureDetail.storageIndex != 91) {
    return false;
  }
  allocator.invalidPlacement = true;
  rejected = allocator.allocateDomains(domains, failureDetail);
  if (mlir::succeeded(rejected) || failureDetail.domainIndex != 0 ||
      failureDetail.storageIndex != 91) {
    return false;
  }
  llvm::outs() << "sram_domain_cases=" << cases + 2 << "\n";
  return true;
}

static bool verifySRAMLocationAllocation() {
  using mlir::tt::ttl::SRAMAllocationLocation;
  using mlir::tt::ttl::SRAMAllocationRegion;
  using mlir::tt::ttl::SRAMEqualOffsetGroup;
  using mlir::tt::ttl::SRAMLocationAllocationProblem;

  SRAMLocationAllocationProblem asymmetric{
      {SRAMAllocationLocation{0, 256}, SRAMAllocationLocation{0, 256}},
      {SRAMAllocationRegion{10, 0, 64, std::nullopt},
       SRAMAllocationRegion{10, 1, 192, std::nullopt}},
      InterferenceGraph(2),
      {SRAMEqualOffsetGroup{{0, 1}}},
      64,
      {}};
  SRAMLocationAllocationProblem mixed{
      {SRAMAllocationLocation{0, 320}, SRAMAllocationLocation{64, 320}},
      {SRAMAllocationRegion{10, 0, 64, std::nullopt},
       SRAMAllocationRegion{10, 1, 128, std::nullopt},
       SRAMAllocationRegion{11, 0, 128, std::nullopt},
       SRAMAllocationRegion{12, 1, 64, std::nullopt},
       SRAMAllocationRegion{13, 0, 64, 256}},
      InterferenceGraph(5),
      {SRAMEqualOffsetGroup{{0, 1}}},
      64,
      {}};
  mixed.conflicts.addInterference(0, 2);
  mixed.conflicts.addInterference(1, 3);
  mixed.conflicts.addInterference(2, 4);

  uint64_t cases = 0;
  for (llvm::StringRef strategy :
       {mlir::tt::ttl::kFirstFitDecreasingSRAMAllocator,
        mlir::tt::ttl::kBestFitDecreasingSRAMAllocator,
        mlir::tt::ttl::kMultiOrderDecreasingSRAMAllocator,
        mlir::tt::ttl::kExactSRAMAllocator}) {
    std::string reason;
    auto allocator = mlir::tt::ttl::createSRAMAllocator(
        strategy, {kUnlimitedSearchStates}, reason);
    if (mlir::failed(allocator)) {
      return false;
    }
    std::optional<unsigned> failedRegion;
    auto asymmetricResult =
        (*allocator)->allocateLocations(asymmetric, failedRegion, reason);
    if (mlir::failed(asymmetricResult) ||
        asymmetricResult->offsets != llvm::SmallVector<uint64_t>({0, 0}) ||
        asymmetricResult->highWaterBytes !=
            llvm::SmallVector<uint64_t>({64, 192})) {
      llvm::errs() << "asymmetric equal-offset allocation mismatch: "
                   << strategy << "\n";
      return false;
    }
    auto mixedResult =
        (*allocator)->allocateLocations(mixed, failedRegion, reason);
    if (mlir::failed(mixedResult) ||
        mixedResult->offsets[0] != mixedResult->offsets[1] ||
        mixedResult->offsets[4] != 256 ||
        mixedResult->highWaterBytes !=
            llvm::SmallVector<uint64_t>({320, 256})) {
      llvm::errs() << "mixed location allocation mismatch: " << strategy
                   << "\n";
      return false;
    }
    auto repeated =
        (*allocator)->allocateLocations(mixed, failedRegion, reason);
    if (mlir::failed(repeated) || repeated->offsets != mixedResult->offsets) {
      llvm::errs() << "location allocation is not deterministic\n";
      return false;
    }
    SRAMLocationAllocationProblem empty{
        {SRAMAllocationLocation{64, 256}, SRAMAllocationLocation{128, 256}},
        {},
        InterferenceGraph(0),
        {},
        64,
        {}};
    auto emptyResult =
        (*allocator)->allocateLocations(empty, failedRegion, reason);
    if (mlir::failed(emptyResult) || !emptyResult->offsets.empty() ||
        emptyResult->highWaterBytes != llvm::SmallVector<uint64_t>({64, 128})) {
      llvm::errs() << "empty location allocation mismatch\n";
      return false;
    }
    ++cases;
  }

  std::string reason;
  auto allocator = mlir::tt::ttl::createSRAMAllocator(
      mlir::tt::ttl::kMultiOrderDecreasingSRAMAllocator,
      {kUnlimitedSearchStates}, reason);
  if (mlir::failed(allocator)) {
    return false;
  }
  std::optional<unsigned> failedRegion;
  uint64_t validationCases = 0;
  auto expectFailure = [&](const SRAMLocationAllocationProblem &problem,
                           std::optional<unsigned> expectedRegion,
                           llvm::StringRef expectedReason) {
    reason.clear();
    auto result =
        (*allocator)->allocateLocations(problem, failedRegion, reason);
    ++validationCases;
    bool matched = mlir::failed(result) && failedRegion == expectedRegion &&
                   llvm::StringRef(reason).contains(expectedReason);
    if (!matched) {
      llvm::errs() << "location validation mismatch: expected '"
                   << expectedReason << "', received '" << reason << "'\n";
    }
    return matched;
  };
  auto invalid = mixed;
  invalid.equalOffsetGroups = {SRAMEqualOffsetGroup{{0, 2}}};
  if (!expectFailure(invalid, 2, "multiple regions at one location")) {
    return false;
  }
  invalid = mixed;
  invalid.regions[0].fixedOffset = 0;
  invalid.regions[1].fixedOffset = 64;
  if (!expectFailure(invalid, 1, "incompatible fixed offsets")) {
    return false;
  }
  invalid = mixed;
  invalid.conflicts.addInterference(0, 1);
  if (!expectFailure(invalid, 1, "same location")) {
    return false;
  }
  invalid = mixed;
  invalid.regions[2].ownerIndex = 10;
  if (!expectFailure(invalid, 2, "duplicate regions")) {
    return false;
  }
  invalid = mixed;
  invalid.regions[2].bytes = 0;
  if (!expectFailure(invalid, 2, "region size")) {
    return false;
  }
  invalid = mixed;
  invalid.regions[2].locationIndex = 2;
  if (!expectFailure(invalid, 2, "unknown SRAM location")) {
    return false;
  }
  invalid = mixed;
  invalid.regions[2].fixedOffset = 1;
  if (!expectFailure(invalid, 2, "fixed region")) {
    return false;
  }
  invalid = mixed;
  invalid.equalOffsetGroups = {SRAMEqualOffsetGroup{{0}}};
  if (!expectFailure(invalid, std::nullopt, "at least two regions")) {
    return false;
  }
  invalid = mixed;
  invalid.equalOffsetGroups = {SRAMEqualOffsetGroup{{0, 5}}};
  if (!expectFailure(invalid, std::nullopt, "unknown region")) {
    return false;
  }
  invalid = mixed;
  invalid.equalOffsetGroups = {SRAMEqualOffsetGroup{{0, 1}},
                               SRAMEqualOffsetGroup{{1, 2}}};
  if (!expectFailure(invalid, 1, "multiple equal-offset groups")) {
    return false;
  }
  invalid = mixed;
  invalid.alignmentBytes = 3;
  if (!expectFailure(invalid, std::nullopt, "power of two")) {
    return false;
  }
  invalid = mixed;
  invalid.locations[0].payloadBaseOffset = 1;
  if (!expectFailure(invalid, std::nullopt, "payload base")) {
    return false;
  }
  invalid = mixed;
  invalid.locations[0].payloadBaseOffset = 384;
  if (!expectFailure(invalid, std::nullopt, "exceeds its SRAM budget")) {
    return false;
  }
  invalid = asymmetric;
  invalid.locations[0].budgetBytes = std::numeric_limits<uint64_t>::max();
  invalid.locations[1].budgetBytes = std::numeric_limits<uint64_t>::max();
  if (!expectFailure(invalid, std::nullopt, "budgets overflow")) {
    return false;
  }
  invalid = mixed;
  invalid.equalCapacityGroups = {mlir::tt::ttl::SRAMEqualCapacityGroup{{0}}};
  if (!expectFailure(invalid, std::nullopt, "at least two locations")) {
    return false;
  }
  invalid = mixed;
  invalid.equalCapacityGroups = {mlir::tt::ttl::SRAMEqualCapacityGroup{{0, 2}}};
  if (!expectFailure(invalid, std::nullopt, "unknown SRAM location")) {
    return false;
  }
  invalid = mixed;
  invalid.equalCapacityGroups = {mlir::tt::ttl::SRAMEqualCapacityGroup{{0, 1}},
                                 mlir::tt::ttl::SRAMEqualCapacityGroup{{1, 0}}};
  if (!expectFailure(invalid, std::nullopt, "multiple equal-capacity groups")) {
    return false;
  }
  invalid = asymmetric;
  invalid.locations[0] = SRAMAllocationLocation{192, 256};
  invalid.locations[1] = SRAMAllocationLocation{0, 128};
  invalid.regions.clear();
  invalid.conflicts = InterferenceGraph(0);
  invalid.equalOffsetGroups.clear();
  invalid.equalCapacityGroups = {mlir::tt::ttl::SRAMEqualCapacityGroup{{0, 1}}};
  if (!expectFailure(invalid, std::nullopt, "payload base exceeds")) {
    return false;
  }
  invalid = mixed;
  invalid.conflicts = InterferenceGraph(4);
  if (!expectFailure(invalid, std::nullopt, "conflict graph size")) {
    return false;
  }
  invalid = mixed;
  invalid.regions[2].bytes = 63;
  if (!expectFailure(invalid, 2, "region size")) {
    return false;
  }
  invalid = mixed;
  invalid.regions[2].fixedOffset = 256;
  if (!expectFailure(invalid, 2, "fixed region")) {
    return false;
  }
  invalid = asymmetric;
  invalid.locations[1].budgetBytes = 128;
  if (!expectFailure(invalid, std::nullopt, "no placement offset")) {
    return false;
  }
  auto fixedGroup = asymmetric;
  fixedGroup.regions[0].fixedOffset = 64;
  auto fixedGroupResult =
      (*allocator)->allocateLocations(fixedGroup, failedRegion, reason);
  if (mlir::failed(fixedGroupResult) ||
      fixedGroupResult->offsets != llvm::SmallVector<uint64_t>({64, 64}) ||
      fixedGroupResult->highWaterBytes !=
          llvm::SmallVector<uint64_t>({128, 256})) {
    llvm::errs() << "fixed equal-offset group allocation mismatch\n";
    return false;
  }
  SRAMLocationAllocationProblem orderingRegression{
      {SRAMAllocationLocation{0, 6}, SRAMAllocationLocation{0, 6}},
      {SRAMAllocationRegion{0, 0, 1, std::nullopt},
       SRAMAllocationRegion{0, 1, 1, std::nullopt},
       SRAMAllocationRegion{1, 0, 1, std::nullopt},
       SRAMAllocationRegion{2, 0, 1, std::nullopt},
       SRAMAllocationRegion{3, 1, 1, std::nullopt},
       SRAMAllocationRegion{4, 1, 3, std::nullopt}},
      InterferenceGraph(6),
      {SRAMEqualOffsetGroup{{0, 1}}},
      1,
      {}};
  orderingRegression.conflicts.addInterference(1, 5);
  auto exact = mlir::tt::ttl::createSRAMAllocator(
      mlir::tt::ttl::kExactSRAMAllocator, {kUnlimitedSearchStates}, reason);
  if (mlir::failed(exact)) {
    return false;
  }
  auto orderingResult =
      (*exact)->allocateLocations(orderingRegression, failedRegion, reason);
  if (mlir::failed(orderingResult) ||
      orderingResult->highWaterBytes != llvm::SmallVector<uint64_t>({1, 4})) {
    llvm::errs() << "exact location ordering regression\n";
    return false;
  }
  SRAMLocationAllocationProblem selectedGroups{
      {SRAMAllocationLocation{0, 128}, SRAMAllocationLocation{0, 128},
       SRAMAllocationLocation{0, 128}},
      {SRAMAllocationRegion{0, 0, 32, std::nullopt},
       SRAMAllocationRegion{0, 1, 64, std::nullopt},
       SRAMAllocationRegion{1, 1, 32, std::nullopt},
       SRAMAllocationRegion{1, 2, 48, std::nullopt}},
      InterferenceGraph(4),
      {SRAMEqualOffsetGroup{{0, 1}}, SRAMEqualOffsetGroup{{2, 3}}},
      16,
      {}};
  selectedGroups.conflicts.addInterference(1, 2);
  auto selectedGroupResult =
      (*exact)->allocateLocations(selectedGroups, failedRegion, reason);
  if (mlir::failed(selectedGroupResult) ||
      selectedGroupResult->offsets !=
          llvm::SmallVector<uint64_t>({32, 32, 0, 0}) ||
      selectedGroupResult->highWaterBytes !=
          llvm::SmallVector<uint64_t>({64, 96, 48})) {
    llvm::errs() << "selected equal-offset group allocation mismatch\n";
    return false;
  }
  SRAMLocationAllocationProblem capacityObjective{
      {SRAMAllocationLocation{0, 8}, SRAMAllocationLocation{0, 8}},
      {SRAMAllocationRegion{0, 0, 3, std::nullopt},
       SRAMAllocationRegion{1, 0, 1, std::nullopt},
       SRAMAllocationRegion{1, 1, 2, std::nullopt},
       SRAMAllocationRegion{2, 0, 3, std::nullopt},
       SRAMAllocationRegion{2, 1, 1, std::nullopt}},
      InterferenceGraph(5),
      {SRAMEqualOffsetGroup{{1, 2}}, SRAMEqualOffsetGroup{{3, 4}}},
      1,
      {mlir::tt::ttl::SRAMEqualCapacityGroup{{0, 1}}}};
  capacityObjective.conflicts.addInterference(0, 1);
  capacityObjective.conflicts.addInterference(0, 3);
  capacityObjective.conflicts.addInterference(1, 3);
  capacityObjective.conflicts.addInterference(2, 4);
  auto capacityResult =
      (*exact)->allocateLocations(capacityObjective, failedRegion, reason);
  if (mlir::failed(capacityResult) ||
      capacityResult->offsets != llvm::SmallVector<uint64_t>({4, 3, 3, 0, 0}) ||
      capacityResult->highWaterBytes != llvm::SmallVector<uint64_t>({7, 5})) {
    llvm::errs() << "equal-capacity reservation objective mismatch: " << reason;
    if (mlir::succeeded(capacityResult)) {
      llvm::errs() << " offsets=";
      llvm::interleaveComma(capacityResult->offsets, llvm::errs());
      llvm::errs() << " high-water=";
      llvm::interleaveComma(capacityResult->highWaterBytes, llvm::errs());
    }
    llvm::errs() << "\n";
    return false;
  }
  auto limitedExact = mlir::tt::ttl::createSRAMAllocator(
      mlir::tt::ttl::kExactSRAMAllocator, {1}, reason);
  if (mlir::failed(limitedExact)) {
    return false;
  }
  auto limitedResult =
      (*limitedExact)->allocateLocations(asymmetric, failedRegion, reason);
  if (mlir::succeeded(limitedResult) ||
      llvm::StringRef(reason).find("work items") == llvm::StringRef::npos) {
    llvm::errs() << "exact location search limit was not enforced\n";
    return false;
  }

  LocationTestAllocator testAllocator;
  for (auto [invalidSolution, expectedRegion, expectedReason] :
       {std::tuple{InvalidLocationSolution::WrongSize,
                   std::optional<unsigned>{}, "incorrect location result size"},
        std::tuple{InvalidLocationSolution::InvalidOffset,
                   std::optional<unsigned>{1}, "invalid per-location offset"},
        std::tuple{InvalidLocationSolution::FixedOffset,
                   std::optional<unsigned>{4}, "invalid per-location offset"},
        std::tuple{InvalidLocationSolution::UnequalOffsets,
                   std::optional<unsigned>{1}, "equal-offset constraint"},
        std::tuple{InvalidLocationSolution::Overlap, std::optional<unsigned>{2},
                   "overlapped conflicting"},
        std::tuple{InvalidLocationSolution::WrongHighWater,
                   std::optional<unsigned>{}, "high-water marks"}}) {
    testAllocator.invalidSolution = invalidSolution;
    reason.clear();
    auto invalidResult =
        testAllocator.allocateLocations(mixed, failedRegion, reason);
    if (mlir::succeeded(invalidResult) || failedRegion != expectedRegion ||
        !llvm::StringRef(reason).contains(expectedReason)) {
      llvm::errs() << "invalid location solution passed common validation\n";
      return false;
    }
    ++validationCases;
  }

  auto capacityValidation = mixed;
  capacityValidation.locations[1].budgetBytes = 256;
  capacityValidation.equalCapacityGroups = {
      mlir::tt::ttl::SRAMEqualCapacityGroup{{0, 1}}};
  testAllocator.invalidSolution = InvalidLocationSolution::None;
  reason.clear();
  auto invalidCapacity =
      testAllocator.allocateLocations(capacityValidation, failedRegion, reason);
  if (mlir::succeeded(invalidCapacity) || failedRegion ||
      !llvm::StringRef(reason).contains("equal capacity")) {
    llvm::errs() << "invalid equal-capacity solution passed validation\n";
    return false;
  }
  ++validationCases;

  llvm::outs() << "sram_location_cases=" << cases + validationCases + 5 << "\n";
  return true;
}

static bool compareExactSRAMLocationAllocationWithOracle() {
  using mlir::tt::ttl::SRAMAllocationLocation;
  using mlir::tt::ttl::SRAMAllocationRegion;
  using mlir::tt::ttl::SRAMEqualOffsetGroup;
  using mlir::tt::ttl::SRAMLocationAllocationProblem;

  std::string reason;
  auto allocator = mlir::tt::ttl::createSRAMAllocator(
      mlir::tt::ttl::kExactSRAMAllocator, {kUnlimitedSearchStates}, reason);
  if (mlir::failed(allocator)) {
    return false;
  }
  uint64_t cases = 0;
  for (uint64_t firstBytes = 1; firstBytes <= 3; ++firstBytes) {
    for (uint64_t secondBytes = 1; secondBytes <= 3; ++secondBytes) {
      for (uint64_t thirdBytes = 1; thirdBytes <= 3; ++thirdBytes) {
        for (uint64_t fourthBytes = 1; fourthBytes <= 3; ++fourthBytes) {
          for (unsigned conflictMask = 0; conflictMask < 4; ++conflictMask) {
            SRAMLocationAllocationProblem problem{
                {SRAMAllocationLocation{0, 6}, SRAMAllocationLocation{0, 6}},
                {SRAMAllocationRegion{0, 0, firstBytes, std::nullopt},
                 SRAMAllocationRegion{0, 1, secondBytes, std::nullopt},
                 SRAMAllocationRegion{1, 0, thirdBytes, std::nullopt},
                 SRAMAllocationRegion{2, 1, fourthBytes, std::nullopt}},
                InterferenceGraph(4),
                {SRAMEqualOffsetGroup{{0, 1}}},
                1,
                {}};
            if (conflictMask & 1) {
              problem.conflicts.addInterference(0, 2);
            }
            if (conflictMask & 2) {
              problem.conflicts.addInterference(1, 3);
            }

            uint64_t oracleCost = std::numeric_limits<uint64_t>::max();
            for (uint64_t sharedOffset = 0; sharedOffset < 6; ++sharedOffset) {
              if (sharedOffset + firstBytes > 6 ||
                  sharedOffset + secondBytes > 6) {
                continue;
              }
              for (uint64_t thirdOffset = 0; thirdOffset + thirdBytes <= 6;
                   ++thirdOffset) {
                bool firstConflict = (conflictMask & 1) &&
                                     sharedOffset < thirdOffset + thirdBytes &&
                                     thirdOffset < sharedOffset + firstBytes;
                if (firstConflict) {
                  continue;
                }
                for (uint64_t fourthOffset = 0; fourthOffset + fourthBytes <= 6;
                     ++fourthOffset) {
                  bool secondConflict =
                      (conflictMask & 2) &&
                      sharedOffset < fourthOffset + fourthBytes &&
                      fourthOffset < sharedOffset + secondBytes;
                  if (secondConflict) {
                    continue;
                  }
                  oracleCost = std::min(
                      oracleCost, std::max(sharedOffset + firstBytes,
                                           thirdOffset + thirdBytes) +
                                      std::max(sharedOffset + secondBytes,
                                               fourthOffset + fourthBytes));
                }
              }
            }
            std::optional<unsigned> failedRegion;
            auto solution = allocator.value()->allocateLocations(
                problem, failedRegion, reason);
            if (mlir::failed(solution)) {
              llvm::errs() << "exact location allocation unexpectedly failed\n";
              return false;
            }
            uint64_t actualCost =
                solution->highWaterBytes[0] + solution->highWaterBytes[1];
            if (actualCost != oracleCost) {
              llvm::errs() << "exact location oracle mismatch: " << actualCost
                           << " != " << oracleCost << "\n";
              return false;
            }
            ++cases;
          }
        }
      }
    }
  }
  llvm::outs() << "sram_location_oracle_cases=" << cases << "\n";
  return true;
}

static bool compareSRAMPlacementWithOracle() {
  constexpr unsigned kVertexCount = 4;
  constexpr unsigned kGraphCount = 1U << 6;
  constexpr unsigned kSizeValueCount = 3;
  constexpr unsigned kSizeTupleCount =
      kSizeValueCount * kSizeValueCount * kSizeValueCount * kSizeValueCount;
  constexpr uint64_t kAlignmentBytes = 4;
  constexpr uint64_t kPayloadBaseOffset = 12;
  SRAMAllocatorOptions options{kUnlimitedSearchStates};
  std::string failureReason;
  mlir::FailureOr<std::unique_ptr<SRAMAllocator>> exactAllocator =
      mlir::tt::ttl::createSRAMAllocator(mlir::tt::ttl::kExactSRAMAllocator,
                                         options, failureReason);
  mlir::FailureOr<std::unique_ptr<SRAMAllocator>> firstFitAllocator =
      mlir::tt::ttl::createSRAMAllocator(
          mlir::tt::ttl::kFirstFitDecreasingSRAMAllocator, options,
          failureReason);
  mlir::FailureOr<std::unique_ptr<SRAMAllocator>> bestFitAllocator =
      mlir::tt::ttl::createSRAMAllocator(
          mlir::tt::ttl::kBestFitDecreasingSRAMAllocator, options,
          failureReason);
  if (mlir::failed(exactAllocator) || mlir::failed(firstFitAllocator) ||
      mlir::failed(bestFitAllocator)) {
    llvm::errs() << "failed to create SRAM allocators: " << failureReason
                 << "\n";
    return false;
  }

  auto multiAllocator = mlir::tt::ttl::createSRAMAllocator(
      mlir::tt::ttl::kMultiOrderDecreasingSRAMAllocator, options,
      failureReason);
  if (mlir::failed(multiAllocator)) {
    return false;
  }
  uint64_t checkedCaseCount = 0;
  PlacementQuality multiQuality;
  PlacementQuality firstFitQuality;
  PlacementQuality bestFitQuality;
  for (unsigned edgeMask = 0; edgeMask < kGraphCount; ++edgeMask) {
    InterferenceGraph graph = buildGraph(kVertexCount, edgeMask);
    for (unsigned encodedSizes = 0; encodedSizes < kSizeTupleCount;
         ++encodedSizes) {
      unsigned remainingSizes = encodedSizes;
      llvm::SmallVector<uint64_t> regionBytes;
      uint64_t budgetBytes = kPayloadBaseOffset;
      for (unsigned regionIndex = 0; regionIndex < kVertexCount;
           ++regionIndex) {
        uint64_t regionSize =
            (1 + remainingSizes % kSizeValueCount) * kAlignmentBytes;
        remainingSizes /= kSizeValueCount;
        regionBytes.push_back(regionSize);
        budgetBytes += regionSize;
      }
      SRAMAllocationProblem problem{regionBytes, graph, kAlignmentBytes,
                                    kPayloadBaseOffset, budgetBytes};
      std::optional<unsigned> failureRegionIndex;
      mlir::FailureOr<SRAMAllocationSolution> exact =
          (*exactAllocator)
              ->allocate(problem, failureRegionIndex, failureReason);
      mlir::FailureOr<SRAMAllocationSolution> repeated =
          (*exactAllocator)
              ->allocate(problem, failureRegionIndex, failureReason);
      mlir::FailureOr<SRAMAllocationSolution> firstFit =
          (*firstFitAllocator)
              ->allocate(problem, failureRegionIndex, failureReason);
      mlir::FailureOr<SRAMAllocationSolution> bestFit =
          (*bestFitAllocator)
              ->allocate(problem, failureRegionIndex, failureReason);
      auto multi = (*multiAllocator)
                       ->allocate(problem, failureRegionIndex, failureReason);
      if (mlir::failed(multi) || mlir::failed(firstFit) ||
          multi->arenaBytes > firstFit->arenaBytes ||
          (multi->arenaBytes == firstFit->arenaBytes &&
           multi->offsets != firstFit->offsets)) {
        llvm::errs()
            << "multi-order placement regressed or changed a tied layout\n";
        return false;
      }
      uint64_t expectedArenaBytes = oracleMinimumArenaBytes(
          graph, regionBytes, kAlignmentBytes, kPayloadBaseOffset);
      if (mlir::failed(exact) || mlir::failed(repeated) ||
          mlir::failed(firstFit) || mlir::failed(bestFit) ||
          exact->arenaBytes != expectedArenaBytes ||
          repeated->arenaBytes != exact->arenaBytes ||
          repeated->offsets != exact->offsets ||
          firstFit->arenaBytes < exact->arenaBytes ||
          bestFit->arenaBytes < exact->arenaBytes) {
        llvm::errs() << "SRAM placement mismatch: edge_mask=" << edgeMask
                     << " encoded_sizes=" << encodedSizes
                     << " expected=" << expectedArenaBytes;
        if (mlir::succeeded(exact)) {
          llvm::errs() << " exact=" << exact->arenaBytes;
        }
        llvm::errs() << " failure=" << failureReason << "\n";
        return false;
      }

      uint64_t optimalUnits =
          (exact->arenaBytes - kPayloadBaseOffset) / kAlignmentBytes;
      uint64_t firstFitUnits =
          (firstFit->arenaBytes - kPayloadBaseOffset) / kAlignmentBytes;
      uint64_t bestFitUnits =
          (bestFit->arenaBytes - kPayloadBaseOffset) / kAlignmentBytes;
      multiQuality.record(optimalUnits,
                          (multi->arenaBytes - kPayloadBaseOffset) /
                              kAlignmentBytes);
      firstFitQuality.record(optimalUnits, firstFitUnits);
      bestFitQuality.record(optimalUnits, bestFitUnits);
      ++checkedCaseCount;
    }
  }

  llvm::outs() << "sram_multi_order_suboptimal_cases="
               << multiQuality.suboptimalCount << "\n"
               << "sram_multi_order_excess_units=" << multiQuality.excessUnits
               << "\n"
               << "sram_multi_order_aggregate_efficiency_basis_points="
               << getEfficiencyBasisPoints(multiQuality.optimalUnits,
                                           multiQuality.allocatedUnits)
               << "\n"
               << "sram_multi_order_worst_efficiency_basis_points="
               << getEfficiencyBasisPoints(multiQuality.worstOptimalUnits,
                                           multiQuality.worstAllocatedUnits)
               << "\n";
  llvm::outs()
      << "sram_placement_cases=" << checkedCaseCount << "\n"
      << "sram_first_fit_suboptimal_cases=" << firstFitQuality.suboptimalCount
      << "\n"
      << "sram_best_fit_suboptimal_cases=" << bestFitQuality.suboptimalCount
      << "\n"
      << "sram_first_fit_excess_units=" << firstFitQuality.excessUnits << "\n"
      << "sram_best_fit_excess_units=" << bestFitQuality.excessUnits << "\n"
      << "sram_first_fit_max_excess_units="
      << firstFitQuality.maximumExcessUnits << "\n"
      << "sram_best_fit_max_excess_units=" << bestFitQuality.maximumExcessUnits
      << "\n"
      << "sram_first_fit_aggregate_efficiency_basis_points="
      << getEfficiencyBasisPoints(firstFitQuality.optimalUnits,
                                  firstFitQuality.allocatedUnits)
      << "\n"
      << "sram_best_fit_aggregate_efficiency_basis_points="
      << getEfficiencyBasisPoints(bestFitQuality.optimalUnits,
                                  bestFitQuality.allocatedUnits)
      << "\n"
      << "sram_first_fit_suboptimal_efficiency_basis_points="
      << getEfficiencyBasisPoints(firstFitQuality.suboptimalOptimalUnits,
                                  firstFitQuality.suboptimalAllocatedUnits)
      << "\n"
      << "sram_best_fit_suboptimal_efficiency_basis_points="
      << getEfficiencyBasisPoints(bestFitQuality.suboptimalOptimalUnits,
                                  bestFitQuality.suboptimalAllocatedUnits)
      << "\n"
      << "sram_first_fit_worst_efficiency_basis_points="
      << getEfficiencyBasisPoints(firstFitQuality.worstOptimalUnits,
                                  firstFitQuality.worstAllocatedUnits)
      << "\n"
      << "sram_best_fit_worst_efficiency_basis_points="
      << getEfficiencyBasisPoints(bestFitQuality.worstOptimalUnits,
                                  bestFitQuality.worstAllocatedUnits)
      << "\n";
  return compareMultiOrderLargeGraphs();
}

/// Exhaustively measures the assignment-count penalty of uniform assignment
/// relative to per-node and two-group contracts.
static bool compareAssignmentContracts() {
  constexpr unsigned kVertexCount = 4;
  constexpr unsigned kPossibleEdgeCount = 6;
  constexpr unsigned kGraphCount = 1U << kPossibleEdgeCount;
  unsigned minimumIndexCounts[kGraphCount];
  for (unsigned edgeMask = 0; edgeMask < kGraphCount; ++edgeMask) {
    minimumIndexCounts[edgeMask] =
        oracleMinimumIndexCount(buildGraph(kVertexCount, edgeMask));
  }

  uint64_t caseCount = 0;
  uint64_t perNodeImprovementCount = 0;
  uint64_t twoGroupImprovementCount = 0;
  unsigned maximumUniformPenalty = 0;
  for (unsigned firstNode = 0; firstNode < kGraphCount; ++firstNode) {
    for (unsigned secondNode = 0; secondNode < kGraphCount; ++secondNode) {
      for (unsigned thirdNode = 0; thirdNode < kGraphCount; ++thirdNode) {
        unsigned uniformCount =
            minimumIndexCounts[firstNode | secondNode | thirdNode];
        unsigned perNodeCount = std::max({minimumIndexCounts[firstNode],
                                          minimumIndexCounts[secondNode],
                                          minimumIndexCounts[thirdNode]});
        unsigned twoGroupCount =
            std::min({std::max(minimumIndexCounts[firstNode | secondNode],
                               minimumIndexCounts[thirdNode]),
                      std::max(minimumIndexCounts[firstNode | thirdNode],
                               minimumIndexCounts[secondNode]),
                      std::max(minimumIndexCounts[secondNode | thirdNode],
                               minimumIndexCounts[firstNode])});
        ++caseCount;
        if (perNodeCount < uniformCount) {
          ++perNodeImprovementCount;
          maximumUniformPenalty =
              std::max(maximumUniformPenalty, uniformCount - perNodeCount);
        }
        if (twoGroupCount < uniformCount) {
          ++twoGroupImprovementCount;
        }
      }
    }
  }

  if (caseCount != 262144 || perNodeImprovementCount != 149268 ||
      twoGroupImprovementCount != 142536 || maximumUniformPenalty != 2) {
    llvm::errs() << "assignment-contract comparison mismatch\n";
    return false;
  }
  llvm::outs() << "contract_cases=" << caseCount << "\n"
               << "per_node_improvements=" << perNodeImprovementCount << "\n"
               << "two_group_improvements=" << twoGroupImprovementCount << "\n"
               << "maximum_uniform_penalty=" << maximumUniformPenalty << "\n";
  return true;
}

} // namespace

int main() {
  return compareProductionSolverWithOracle() &&
                 verifyGreedyCapacityReproducer() &&
                 verifySearchLimitOutcome() &&
                 verifyFixedLimitAvoidsMinimumSearch() &&
                 verifyWeightedColoringAcrossComponents() &&
                 compareWeightedSolverWithOracle() &&
                 compareSRAMPlacementWithOracle() &&
                 verifySRAMAllocationDomains() &&
                 verifySRAMLocationAllocation() &&
                 compareExactSRAMLocationAllocationWithOracle() &&
                 verifyTargetDFBIndexCapacities() &&
                 compareAssignmentContracts()
             ? 0
             : 1;
}
