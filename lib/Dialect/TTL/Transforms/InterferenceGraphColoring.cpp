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

static unsigned getGreedyCliqueLowerBound(const InterferenceGraph &graph,
                                          llvm::ArrayRef<unsigned> component) {
  llvm::SmallVector<unsigned> candidates(component.begin(), component.end());
  llvm::sort(candidates, [&](unsigned lhsVertex, unsigned rhsVertex) {
    unsigned lhsDegree = graph.degree(lhsVertex);
    unsigned rhsDegree = graph.degree(rhsVertex);
    return lhsDegree != rhsDegree ? lhsDegree > rhsDegree
                                  : lhsVertex < rhsVertex;
  });

  llvm::SmallVector<unsigned> clique;
  for (unsigned candidate : candidates) {
    if (llvm::all_of(clique, [&](unsigned member) {
          return graph.interferes(candidate, member);
        })) {
      clique.push_back(candidate);
    }
  }
  return std::max(1U, static_cast<unsigned>(clique.size()));
}

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
  bounds.cliqueLowerBound = getGreedyCliqueLowerBound(graph, vertices);
  bounds.colors = colorInterferenceGraphFirstFit(graph, vertices);
  for (unsigned color : bounds.colors) {
    bounds.colorCount = std::max(bounds.colorCount, color + 1);
  }
  return bounds;
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
    unsigned cliqueLowerBound = getGreedyCliqueLowerBound(graph, component);
    result.cliqueLowerBound =
        std::max(result.cliqueLowerBound, cliqueLowerBound);

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
    for (unsigned candidateCount = cliqueLowerBound;
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
