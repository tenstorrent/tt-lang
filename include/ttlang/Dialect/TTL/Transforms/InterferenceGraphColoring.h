// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_INTERFERENCEGRAPHCOLORING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_INTERFERENCEGRAPHCOLORING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>

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

private:
  llvm::SmallVector<llvm::BitVector> adjacency;
};

/// Strategy interface for coloring an interference graph.
///
/// Implementations must assign different colors to interfering vertices and
/// return dense, zero-based colors. `priorityOrder` lists every vertex exactly
/// once from highest to lowest priority.
class InterferenceGraphColoring {
public:
  virtual ~InterferenceGraphColoring() = default;

  /// Returns one color per vertex, indexed by vertex number.
  virtual llvm::SmallVector<unsigned>
  color(const InterferenceGraph &graph,
        llvm::ArrayRef<unsigned> priorityOrder) const = 0;
};

/// Visits vertices in priority order and assigns the lowest available color.
class GreedyFirstFitInterferenceGraphColoring final
    : public InterferenceGraphColoring {
public:
  llvm::SmallVector<unsigned>
  color(const InterferenceGraph &graph,
        llvm::ArrayRef<unsigned> priorityOrder) const final {
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
};

/// Returns the shared deterministic greedy first-fit coloring strategy.
inline const InterferenceGraphColoring &
getGreedyFirstFitInterferenceGraphColoring() {
  static const GreedyFirstFitInterferenceGraphColoring coloring;
  return coloring;
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_INTERFERENCEGRAPHCOLORING_H
