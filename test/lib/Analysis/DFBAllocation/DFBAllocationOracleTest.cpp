// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test uses exhaustive assignment enumeration as an independent oracle.
// It does not call the production search while computing expected minima.

#include "ttlang/Dialect/TTL/Transforms/InterferenceGraphColoring.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

using mlir::tt::ttl::ExactInterferenceGraphColoring;
using mlir::tt::ttl::ExactInterferenceGraphColoringStatus;
using mlir::tt::ttl::InterferenceGraph;

constexpr uint64_t kUnlimitedSearchStates =
    std::numeric_limits<uint64_t>::max();

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

static unsigned oracleChromaticNumber(const InterferenceGraph &graph) {
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

static bool verifyColoring(const InterferenceGraph &graph,
                           const ExactInterferenceGraphColoring &coloring) {
  if (coloring.colors.size() != graph.size()) {
    return false;
  }
  for (unsigned lhsVertex = 0; lhsVertex < graph.size(); ++lhsVertex) {
    if (coloring.colors[lhsVertex] >= coloring.colorCount) {
      return false;
    }
    for (unsigned rhsVertex = lhsVertex + 1; rhsVertex < graph.size();
         ++rhsVertex) {
      if (graph.interferes(lhsVertex, rhsVertex) &&
          coloring.colors[lhsVertex] == coloring.colors[rhsVertex]) {
        return false;
      }
    }
  }
  return true;
}

static bool compareProductionSolverWithOracle() {
  uint64_t checkedGraphCount = 0;
  for (unsigned vertexCount = 0; vertexCount <= 6; ++vertexCount) {
    unsigned edgeCount = vertexCount * (vertexCount - 1) / 2;
    uint64_t graphCount = uint64_t{1} << edgeCount;
    for (uint64_t edgeMask = 0; edgeMask < graphCount; ++edgeMask) {
      InterferenceGraph graph = buildGraph(vertexCount, edgeMask);
      unsigned oracleCount = oracleChromaticNumber(graph);
      ExactInterferenceGraphColoring production =
          mlir::tt::ttl::colorInterferenceGraphExactly(graph,
                                                       kUnlimitedSearchStates);
      if (!production.isOptimal() || production.colorCount != oracleCount ||
          !verifyColoring(graph, production)) {
        llvm::errs() << "solver mismatch: vertices=" << vertexCount
                     << " edge_mask=" << edgeMask << " oracle=" << oracleCount
                     << " production=" << production.colorCount << "\n";
        return false;
      }
      ++checkedGraphCount;
    }
  }
  llvm::outs() << "solver_graphs=" << checkedGraphCount << "\n";
  return true;
}

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
  // Vertex order A, D, B, C makes first-fit use three colors for the path
  // A-B-C-D after the 30-color clique.
  capacityGraph.addInterference(30, 32);
  capacityGraph.addInterference(32, 33);
  capacityGraph.addInterference(33, 31);
  mlir::tt::ttl::InterferenceGraphColoringBounds capacityBounds =
      mlir::tt::ttl::computeInterferenceGraphColoringBounds(capacityGraph);
  ExactInterferenceGraphColoring capacity =
      mlir::tt::ttl::colorInterferenceGraphExactly(capacityGraph,
                                                   kUnlimitedSearchStates);
  if (capacityBounds.colorCount != 33 || capacity.colorCount != 32) {
    llvm::errs() << "capacity reproducer expected greedy 33 and exact 32, got "
                 << capacityBounds.colorCount << " and " << capacity.colorCount
                 << "\n";
    return false;
  }
  llvm::outs() << "capacity_reproducer=32\n";
  llvm::outs() << "capacity_search_states=" << capacity.exploredStateCount
               << "\n";
  return true;
}

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
  llvm::outs() << "bounded_search_states=" << limited.exploredStateCount
               << "\n";
  return true;
}

static bool compareAssignmentContracts() {
  constexpr unsigned kVertexCount = 4;
  constexpr unsigned kPossibleEdgeCount = 6;
  constexpr unsigned kGraphCount = 1U << kPossibleEdgeCount;
  unsigned chromaticNumbers[kGraphCount];
  for (unsigned edgeMask = 0; edgeMask < kGraphCount; ++edgeMask) {
    chromaticNumbers[edgeMask] =
        oracleChromaticNumber(buildGraph(kVertexCount, edgeMask));
  }

  uint64_t caseCount = 0;
  uint64_t perNodeImprovementCount = 0;
  uint64_t twoGroupImprovementCount = 0;
  unsigned maximumUniformPenalty = 0;
  for (unsigned firstNode = 0; firstNode < kGraphCount; ++firstNode) {
    for (unsigned secondNode = 0; secondNode < kGraphCount; ++secondNode) {
      for (unsigned thirdNode = 0; thirdNode < kGraphCount; ++thirdNode) {
        unsigned uniformCount =
            chromaticNumbers[firstNode | secondNode | thirdNode];
        unsigned perNodeCount =
            std::max({chromaticNumbers[firstNode], chromaticNumbers[secondNode],
                      chromaticNumbers[thirdNode]});
        unsigned twoGroupCount =
            std::min({std::max(chromaticNumbers[firstNode | secondNode],
                               chromaticNumbers[thirdNode]),
                      std::max(chromaticNumbers[firstNode | thirdNode],
                               chromaticNumbers[secondNode]),
                      std::max(chromaticNumbers[secondNode | thirdNode],
                               chromaticNumbers[firstNode])});
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
                 verifySearchLimitOutcome() && compareAssignmentContracts()
             ? 0
             : 1;
}
