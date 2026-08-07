// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/InterferenceGraphColoring.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#include <algorithm>
#include <limits>

namespace mlir::tt::ttl {

namespace {

/// Applies one state limit across all independently solved candidate groups so
/// splitting a graph cannot multiply the configured compile-time budget.
class ExactColoringSearchBudget {
public:
  explicit ExactColoringSearchBudget(std::uint64_t stateLimit)
      : stateLimit(stateLimit) {}

  bool consumeState() {
    if (exploredStateCount == stateLimit) {
      return false;
    }
    ++exploredStateCount;
    return true;
  }

  std::uint64_t getExploredStateCount() const { return exploredStateCount; }

private:
  std::uint64_t stateLimit = 0;
  std::uint64_t exploredStateCount = 0;
};

enum class FixedColorCountSearchStatus {
  Feasible,
  Infeasible,
  SearchLimitReached,
};

struct FixedColorCountSearchResult {
  FixedColorCountSearchStatus status = FixedColorCountSearchStatus::Infeasible;
  llvm::SmallVector<unsigned> colors;
};

/// Separates candidate groups that have no conflicts between them. Independent
/// groups can reuse the same resource slots and are cheaper to search alone.
static llvm::SmallVector<llvm::SmallVector<unsigned>>
getConnectedComponents(const InterferenceGraph &graph) {
  llvm::SmallVector<llvm::SmallVector<unsigned>> components;
  llvm::BitVector visited(graph.size());
  for (unsigned root = 0; root < graph.size(); ++root) {
    if (visited.test(root)) {
      continue;
    }
    llvm::SmallVector<unsigned> component;
    llvm::SmallVector<unsigned> pending = {root};
    visited.set(root);
    while (!pending.empty()) {
      unsigned vertex = pending.pop_back_val();
      component.push_back(vertex);
      for (int neighbor = graph.getNeighbors(vertex).find_first();
           neighbor >= 0;
           neighbor = graph.getNeighbors(vertex).find_next(neighbor)) {
        unsigned neighborVertex = static_cast<unsigned>(neighbor);
        if (!visited.test(neighborVertex)) {
          visited.set(neighborVertex);
          pending.push_back(neighborVertex);
        }
      }
    }
    llvm::sort(component);
    components.push_back(std::move(component));
  }
  return components;
}

/// Finds a pairwise-conflicting set whose size proves a required slot count.
///
/// The degree-first greedy selection may miss a larger set, but every set it
/// returns contains only members that conflict with every other member. Its
/// size is therefore always a valid lower bound on the required slots.
static unsigned
getPairwiseConflictLowerBound(const InterferenceGraph &graph,
                              llvm::ArrayRef<unsigned> component) {
  llvm::SmallVector<unsigned> candidates(component.begin(), component.end());
  llvm::sort(candidates, [&](unsigned lhsVertex, unsigned rhsVertex) {
    unsigned lhsDegree = graph.degree(lhsVertex);
    unsigned rhsDegree = graph.degree(rhsVertex);
    return lhsDegree != rhsDegree ? lhsDegree > rhsDegree
                                  : lhsVertex < rhsVertex;
  });

  llvm::SmallVector<unsigned> conflictingSet;
  for (unsigned candidate : candidates) {
    if (llvm::all_of(conflictingSet, [&](unsigned member) {
          return graph.interferes(candidate, member);
        })) {
      conflictingSet.push_back(candidate);
    }
  }
  return std::max(1U, static_cast<unsigned>(conflictingSet.size()));
}

/// Processes highly constrained candidates first to improve the first-fit
/// assignment while retaining candidate number as a deterministic tie-breaker.
static llvm::SmallVector<unsigned>
getDegreePriorityOrder(const InterferenceGraph &graph,
                       llvm::ArrayRef<unsigned> component) {
  llvm::SmallVector<unsigned> order(component.begin(), component.end());
  llvm::sort(order, [&](unsigned lhsVertex, unsigned rhsVertex) {
    unsigned lhsDegree = graph.degree(lhsVertex);
    unsigned rhsDegree = graph.degree(rhsVertex);
    return lhsDegree != rhsDegree ? lhsDegree > rhsDegree
                                  : lhsVertex < rhsVertex;
  });
  return order;
}

/// Fixing the slot count answers one capacity question without first proving
/// that every smaller count fails.
class FixedColorCountSearch {
public:
  FixedColorCountSearch(const InterferenceGraph &graph,
                        llvm::ArrayRef<unsigned> component, unsigned colorCount,
                        ExactColoringSearchBudget &searchBudget)
      : graph(graph), component(component.begin(), component.end()),
        colorCount(colorCount), colors(graph.size(), kUnassigned),
        active(graph.size()), searchBudget(searchBudget) {
    for (unsigned vertex : component) {
      active.set(vertex);
    }
  }

  FixedColorCountSearchResult solve() {
    FixedColorCountSearchStatus status =
        assign(/*assignedCount=*/0, /*usedColorCount=*/0);
    if (status != FixedColorCountSearchStatus::Feasible) {
      return {status, {}};
    }
    return {status, colors};
  }

private:
  static constexpr unsigned kUnassigned = std::numeric_limits<unsigned>::max();

  /// Implements deterministic DSATUR selection. A candidate's saturation is
  /// the number of different slots already used by its assigned conflicts;
  /// higher saturation means fewer remaining choices and is searched first.
  unsigned selectVertex() const {
    unsigned selected = kUnassigned;
    unsigned selectedSaturation = 0;
    unsigned selectedDegree = 0;
    for (unsigned vertex : component) {
      if (colors[vertex] != kUnassigned) {
        continue;
      }
      llvm::BitVector neighborColors(colorCount);
      unsigned activeDegree = 0;
      for (int neighbor = graph.getNeighbors(vertex).find_first();
           neighbor >= 0;
           neighbor = graph.getNeighbors(vertex).find_next(neighbor)) {
        unsigned neighborVertex = static_cast<unsigned>(neighbor);
        if (!active.test(neighborVertex)) {
          continue;
        }
        ++activeDegree;
        if (colors[neighborVertex] != kUnassigned) {
          neighborColors.set(colors[neighborVertex]);
        }
      }
      unsigned saturation = neighborColors.count();
      if (selected == kUnassigned || saturation > selectedSaturation ||
          (saturation == selectedSaturation &&
           (activeDegree > selectedDegree ||
            (activeDegree == selectedDegree && vertex < selected)))) {
        selected = vertex;
        selectedSaturation = saturation;
        selectedDegree = activeDegree;
      }
    }
    assert(selected != kUnassigned && "search must select an uncolored vertex");
    return selected;
  }

  bool canUseColor(unsigned vertex, unsigned color) const {
    for (int neighbor = graph.getNeighbors(vertex).find_first(); neighbor >= 0;
         neighbor = graph.getNeighbors(vertex).find_next(neighbor)) {
      if (colors[static_cast<unsigned>(neighbor)] == color) {
        return false;
      }
    }
    return true;
  }

  /// Backtracks over legal slots while removing equivalent slot renamings.
  ///
  /// At most one previously unused slot is tried at a state. Any other unused
  /// slot differs only by its number and would produce an equivalent subtree.
  FixedColorCountSearchStatus assign(unsigned assignedCount,
                                     unsigned usedColorCount) {
    if (!searchBudget.consumeState()) {
      return FixedColorCountSearchStatus::SearchLimitReached;
    }
    if (assignedCount == component.size()) {
      return FixedColorCountSearchStatus::Feasible;
    }

    unsigned vertex = selectVertex();
    unsigned candidateColorCount = std::min(colorCount, usedColorCount + 1);
    for (unsigned color = 0; color < candidateColorCount; ++color) {
      if (!canUseColor(vertex, color)) {
        continue;
      }
      colors[vertex] = color;
      unsigned nextUsedColorCount = std::max(usedColorCount, color + 1);
      FixedColorCountSearchStatus status =
          assign(assignedCount + 1, nextUsedColorCount);
      if (status == FixedColorCountSearchStatus::Feasible) {
        return status;
      }
      colors[vertex] = kUnassigned;
      if (status == FixedColorCountSearchStatus::SearchLimitReached) {
        return status;
      }
    }
    return FixedColorCountSearchStatus::Infeasible;
  }

  const InterferenceGraph &graph;
  llvm::SmallVector<unsigned> component;
  unsigned colorCount = 0;
  llvm::SmallVector<unsigned> colors;
  llvm::BitVector active;
  ExactColoringSearchBudget &searchBudget;
};

} // namespace

InterferenceGraphColoringBounds
computeInterferenceGraphColoringBounds(const InterferenceGraph &graph) {
  InterferenceGraphColoringBounds bounds;
  if (graph.size() == 0) {
    return bounds;
  }
  llvm::SmallVector<unsigned> vertices =
      llvm::to_vector(llvm::seq<unsigned>(0, graph.size()));
  bounds.pairwiseConflictLowerBound =
      getPairwiseConflictLowerBound(graph, vertices);
  bounds.colors = colorInterferenceGraphFirstFit(graph, vertices);
  for (unsigned color : bounds.colors) {
    bounds.colorCount = std::max(bounds.colorCount, color + 1);
  }
  return bounds;
}

InterferenceGraphColorLimitResult
colorInterferenceGraphWithColorLimitExactly(const InterferenceGraph &graph,
                                            unsigned colorLimit,
                                            std::uint64_t searchStateLimit) {
  InterferenceGraphColorLimitResult result;
  if (graph.size() == 0) {
    result.status = InterferenceGraphColorLimitStatus::Feasible;
    return result;
  }
  if (colorLimit == 0) {
    return result;
  }

  result.colors.assign(graph.size(), 0);
  ExactColoringSearchBudget searchBudget(searchStateLimit);
  for (llvm::ArrayRef<unsigned> component : getConnectedComponents(graph)) {
    // More pairwise-conflicting candidates than available slots prove failure
    // without backtracking.
    if (getPairwiseConflictLowerBound(graph, component) > colorLimit) {
      result.colors.clear();
      result.exploredStateCount = searchBudget.getExploredStateCount();
      return result;
    }

    FixedColorCountSearch search(graph, component, colorLimit, searchBudget);
    FixedColorCountSearchResult searchResult = search.solve();
    if (searchResult.status ==
        FixedColorCountSearchStatus::SearchLimitReached) {
      result.status = InterferenceGraphColorLimitStatus::SearchLimitReached;
      result.colors.clear();
      result.exploredStateCount = searchBudget.getExploredStateCount();
      return result;
    }
    if (searchResult.status == FixedColorCountSearchStatus::Infeasible) {
      result.colors.clear();
      result.exploredStateCount = searchBudget.getExploredStateCount();
      return result;
    }
    for (unsigned vertex : component) {
      result.colors[vertex] = searchResult.colors[vertex];
      result.colorCount =
          std::max(result.colorCount, searchResult.colors[vertex] + 1);
    }
  }
  result.status = InterferenceGraphColorLimitStatus::Feasible;
  result.exploredStateCount = searchBudget.getExploredStateCount();
  return result;
}

ExactInterferenceGraphColoring
colorInterferenceGraphExactly(const InterferenceGraph &graph,
                              std::uint64_t searchStateLimit) {
  ExactInterferenceGraphColoring result;
  result.colors.assign(graph.size(), 0);
  if (graph.size() == 0) {
    return result;
  }

  ExactColoringSearchBudget searchBudget(searchStateLimit);

  for (llvm::ArrayRef<unsigned> component : getConnectedComponents(graph)) {
    unsigned pairwiseConflictLowerBound =
        getPairwiseConflictLowerBound(graph, component);
    result.pairwiseConflictLowerBound =
        std::max(result.pairwiseConflictLowerBound, pairwiseConflictLowerBound);

    // First-fit supplies a known-valid maximum for the increasing exact
    // searches, so at least one tested slot count must succeed.
    llvm::SmallVector<unsigned> globalPriorityOrder =
        getDegreePriorityOrder(graph, component);
    InterferenceGraph componentGraph(component.size());
    for (unsigned lhsIndex = 0; lhsIndex < component.size(); ++lhsIndex) {
      for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < component.size();
           ++rhsIndex) {
        if (graph.interferes(component[lhsIndex], component[rhsIndex])) {
          componentGraph.addInterference(lhsIndex, rhsIndex);
        }
      }
    }
    llvm::SmallVector<unsigned> componentPriorityOrder;
    componentPriorityOrder.reserve(component.size());
    for (unsigned globalVertex : globalPriorityOrder) {
      componentPriorityOrder.push_back(llvm::find(component, globalVertex) -
                                       component.begin());
    }
    llvm::SmallVector<unsigned> upperColors =
        colorInterferenceGraphFirstFit(componentGraph, componentPriorityOrder);
    unsigned upperBound = 0;
    for (unsigned color : upperColors) {
      upperBound = std::max(upperBound, color + 1);
    }

    llvm::SmallVector<unsigned> exactColors;
    for (unsigned candidateCount = pairwiseConflictLowerBound;
         candidateCount <= upperBound; ++candidateCount) {
      FixedColorCountSearch search(graph, component, candidateCount,
                                   searchBudget);
      FixedColorCountSearchResult searchResult = search.solve();
      if (searchResult.status ==
          FixedColorCountSearchStatus::SearchLimitReached) {
        result.status =
            ExactInterferenceGraphColoringStatus::SearchLimitReached;
        result.colors.clear();
        result.colorCount = 0;
        result.exploredStateCount = searchBudget.getExploredStateCount();
        return result;
      }
      if (searchResult.status == FixedColorCountSearchStatus::Feasible) {
        exactColors = std::move(searchResult.colors);
        result.colorCount = std::max(result.colorCount, candidateCount);
        break;
      }
    }
    assert(!exactColors.empty() && "first-fit upper bound must be feasible");
    for (unsigned vertex : component) {
      result.colors[vertex] = exactColors[vertex];
    }
  }
  result.exploredStateCount = searchBudget.getExploredStateCount();
  return result;
}

} // namespace mlir::tt::ttl
