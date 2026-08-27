// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test treats each graph vertex as one DFB, each edge as a required
// distinct-index relation, and each color as one physical index. Exhaustive
// assignment enumeration supplies expected results without calling the
// production search.

#include "ttlang/Dialect/TTCore/IR/TTCore.h"
#include "ttlang/Dialect/TTL/Transforms/InterferenceGraphColoring.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

using mlir::tt::ttl::ExactInterferenceGraphColoring;
using mlir::tt::ttl::ExactInterferenceGraphColoringStatus;
using mlir::tt::ttl::ExactInterferenceGraphWeightColoring;
using mlir::tt::ttl::ExactInterferenceGraphWeightStatus;
using mlir::tt::ttl::InterferenceGraph;
using mlir::tt::ttl::InterferenceGraphColorLimitResult;
using mlir::tt::ttl::InterferenceGraphColorLimitStatus;
using mlir::tt::ttl::InterferenceGraphWeightLimitResult;

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
                 verifyTargetDFBIndexCapacities() &&
                 compareAssignmentContracts()
             ? 0
             : 1;
}
