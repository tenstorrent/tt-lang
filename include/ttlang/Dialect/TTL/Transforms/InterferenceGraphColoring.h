// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_INTERFERENCEGRAPHCOLORING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_INTERFERENCEGRAPHCOLORING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>

namespace mlir::tt::ttl {

/// Undirected graph whose edges prohibit assigning the same color to both
/// endpoint vertices.
class InterferenceGraph {
public:
  /// Creates a graph with vertices numbered `[0, vertexCount)`.
  explicit InterferenceGraph(unsigned vertexCount)
      : adjacency(vertexCount, llvm::BitVector(vertexCount)) {}

  /// Returns the number of vertices in the graph.
  unsigned size() const { return adjacency.size(); }

  /// Records that two distinct vertices cannot receive the same color.
  void addInterference(unsigned lhs, unsigned rhs) {
    assert(lhs < size() && rhs < size() && lhs != rhs);
    adjacency[lhs].set(rhs);
    adjacency[rhs].set(lhs);
  }

  /// Returns true when two vertices require distinct colors.
  bool interferes(unsigned lhs, unsigned rhs) const {
    assert(lhs < size() && rhs < size());
    return adjacency[lhs].test(rhs);
  }

  /// Returns the neighbors of one vertex.
  const llvm::BitVector &getNeighbors(unsigned vertex) const {
    assert(vertex < size());
    return adjacency[vertex];
  }

  /// Returns the degree of one vertex.
  unsigned degree(unsigned vertex) const {
    return getNeighbors(vertex).count();
  }

private:
  llvm::SmallVector<llvm::BitVector> adjacency;
};

/// Visits vertices in priority order and assigns the lowest available color.
///
/// Production DFB allocation accepts this assignment when it satisfies the
/// hardware limits and otherwise uses exact coloring below.
inline llvm::SmallVector<unsigned>
colorInterferenceGraphFirstFit(const InterferenceGraph &graph,
                               llvm::ArrayRef<unsigned> priorityOrder) {
  assert(priorityOrder.size() == graph.size());

  llvm::BitVector visited(graph.size());
  llvm::SmallVector<unsigned> vertexColors(graph.size());
  llvm::SmallVector<llvm::SmallVector<unsigned>> colorUsers;
  for (unsigned vertex : priorityOrder) {
    assert(vertex < graph.size() && !visited.test(vertex));
    visited.set(vertex);

    unsigned selectedColor = 0;
    for (; selectedColor < colorUsers.size(); ++selectedColor) {
      bool hasInterference = false;
      for (unsigned assignedVertex : colorUsers[selectedColor]) {
        if (graph.interferes(vertex, assignedVertex)) {
          hasInterference = true;
          break;
        }
      }
      if (!hasInterference) {
        break;
      }
    }
    if (selectedColor == colorUsers.size()) {
      colorUsers.emplace_back();
    }
    colorUsers[selectedColor].push_back(vertex);
    vertexColors[vertex] = selectedColor;
  }
  assert(visited.all() && "priority order must contain every graph vertex");
  return vertexColors;
}

/// Completion status for bounded exact coloring.
enum class ExactInterferenceGraphColoringStatus {
  Optimal,
  SearchLimitReached,
};

/// Result of bounded exact coloring of an interference graph.
struct ExactInterferenceGraphColoring {
  ExactInterferenceGraphColoringStatus status =
      ExactInterferenceGraphColoringStatus::Optimal;

  /// Dense zero-based color indexed by vertex number.
  ///
  /// Present only when `status` is `Optimal`.
  llvm::SmallVector<unsigned> colors;

  /// Proven chromatic number, or zero when search was inconclusive.
  unsigned colorCount = 0;

  /// Sound clique lower bound used to begin exact search.
  unsigned cliqueLowerBound = 0;

  /// Recursive search states examined before completion or termination.
  std::uint64_t exploredStateCount = 0;

  bool isOptimal() const {
    return status == ExactInterferenceGraphColoringStatus::Optimal;
  }
};

/// Deterministic feasible coloring with a sound clique lower bound.
struct InterferenceGraphColoringBounds {
  llvm::SmallVector<unsigned> colors;
  unsigned colorCount = 0;
  unsigned cliqueLowerBound = 0;

  bool provesMinimum() const { return colorCount == cliqueLowerBound; }
};

/// Computes a deterministic greedy upper bound and explicit clique lower bound.
InterferenceGraphColoringBounds
computeInterferenceGraphColoringBounds(const InterferenceGraph &graph);

/// Computes a deterministic minimum coloring with bounded DSATUR search.
///
/// Connected components are solved independently and reuse the same color
/// range. Exhaustive infeasibility checks below the returned color count prove
/// that an `Optimal` result is minimum. `SearchLimitReached` contains no
/// coloring and cannot justify a capacity diagnostic.
ExactInterferenceGraphColoring
colorInterferenceGraphExactly(const InterferenceGraph &graph,
                              std::uint64_t searchStateLimit);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_INTERFERENCEGRAPHCOLORING_H
