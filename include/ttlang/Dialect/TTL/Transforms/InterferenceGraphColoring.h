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

/// Allocation constraints represented as an undirected graph.
///
/// Each vertex is one allocation candidate. An edge means its two candidates
/// cannot share a resource. A color is one resource slot, so a valid coloring
/// assigns different colors to the endpoints of every edge. For DFB allocation,
/// vertices are logical DFBs and colors are physical DFB indices.
class InterferenceGraph {
public:
  /// Creates a graph with vertices numbered `[0, vertexCount)`.
  explicit InterferenceGraph(unsigned vertexCount)
      : adjacency(vertexCount, llvm::BitVector(vertexCount)) {}

  /// Returns the number of vertices in the graph.
  unsigned size() const { return adjacency.size(); }

  /// Records that two distinct candidates cannot share a resource slot.
  void addInterference(unsigned lhs, unsigned rhs) {
    assert(lhs < size() && rhs < size() && lhs != rhs);
    adjacency[lhs].set(rhs);
    adjacency[rhs].set(lhs);
  }

  /// Returns true when two candidates require distinct resource slots.
  bool interferes(unsigned lhs, unsigned rhs) const {
    assert(lhs < size() && rhs < size());
    return adjacency[lhs].test(rhs);
  }

  /// Returns the candidates that conflict with one candidate.
  const llvm::BitVector &getNeighbors(unsigned vertex) const {
    assert(vertex < size());
    return adjacency[vertex];
  }

  /// Returns the number of candidates that conflict with one candidate.
  unsigned degree(unsigned vertex) const {
    return getNeighbors(vertex).count();
  }

private:
  llvm::SmallVector<llvm::BitVector> adjacency;
};

/// Assigns each candidate the lowest-numbered slot not used by a conflict.
///
/// Candidates are processed in `priorityOrder`, so the result is deterministic
/// but may use more slots than necessary. The returned valid assignment gives
/// an upper bound on the minimum required slot count.
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

/// Completion status for a bounded minimum-slot search.
enum class ExactInterferenceGraphColoringStatus {
  Optimal,
  SearchLimitReached,
};

/// Result of proving the minimum number of slots required by a graph.
struct ExactInterferenceGraphColoring {
  ExactInterferenceGraphColoringStatus status =
      ExactInterferenceGraphColoringStatus::Optimal;

  /// Dense zero-based resource slot indexed by candidate number.
  ///
  /// Present only when `status` is `Optimal`.
  llvm::SmallVector<unsigned> colors;

  /// Proven minimum slot count, or zero when search was inconclusive.
  unsigned colorCount = 0;

  /// Size of a found set whose members all conflict with each other.
  ///
  /// Such a set is a clique. Every member needs a distinct slot, so its size is
  /// a proved lower bound on the required slot count. The found clique need not
  /// be the largest clique in the graph.
  unsigned pairwiseConflictLowerBound = 0;

  /// Recursive search states examined before completion or termination.
  std::uint64_t exploredStateCount = 0;

  bool isOptimal() const {
    return status == ExactInterferenceGraphColoringStatus::Optimal;
  }
};

/// A valid first-fit assignment and a proved lower bound on required slots.
struct InterferenceGraphColoringBounds {
  /// Dense zero-based resource slot indexed by candidate number.
  llvm::SmallVector<unsigned> colors;

  /// Slots used by `colors`, which is an upper bound on the minimum.
  unsigned colorCount = 0;

  /// Size of a found clique, which is a lower bound on the minimum.
  unsigned pairwiseConflictLowerBound = 0;

  /// Equal valid upper and proved lower bounds establish the minimum.
  bool provesMinimum() const {
    return colorCount == pairwiseConflictLowerBound;
  }
};

/// Computes a first-fit assignment and a pairwise-conflict lower bound.
InterferenceGraphColoringBounds
computeInterferenceGraphColoringBounds(const InterferenceGraph &graph);

/// Completion status for deciding whether a fixed slot limit is sufficient.
enum class InterferenceGraphColorLimitStatus {
  Feasible,
  Infeasible,
  SearchLimitReached,
};

/// Result of deciding whether an allocation graph fits a fixed slot limit.
struct InterferenceGraphColorLimitResult {
  InterferenceGraphColorLimitStatus status =
      InterferenceGraphColorLimitStatus::Infeasible;

  /// Dense zero-based resource slot indexed by candidate number.
  ///
  /// Present only when `status` is `Feasible`.
  llvm::SmallVector<unsigned> colors;

  /// Slots used by the feasible assignment, or zero otherwise.
  unsigned colorCount = 0;

  /// Recursive search states examined before completion or termination.
  std::uint64_t exploredStateCount = 0;

  bool isFeasible() const {
    return status == InterferenceGraphColorLimitStatus::Feasible;
  }
};

/// Decides whether `graph` can use at most `colorLimit` resource slots.
///
/// The deterministic backtracking search selects the unassigned candidate
/// constrained by the most different slots among its already assigned
/// conflicts, then tries legal slots in numeric order. This DSATUR ordering
/// reduces branching by resolving candidates with the fewest remaining choices
/// first; it does not change which assignments are feasible. `Infeasible` is
/// returned only after the search exhausts every assignment. Reaching the state
/// limit is inconclusive and cannot justify a capacity rejection.
InterferenceGraphColorLimitResult
colorInterferenceGraphWithColorLimitExactly(const InterferenceGraph &graph,
                                            unsigned colorLimit,
                                            std::uint64_t searchStateLimit);

/// Result of deciding whether a weighted allocation fits fixed slot and byte
/// limits.
using InterferenceGraphWeightLimitResult = InterferenceGraphColorLimitResult;

/// Finds a valid coloring whose sum of maximum vertex weight per used color is
/// at most `weightLimit` while using at most `colorLimit` colors.
///
/// This searches the complete graph because disconnected components still
/// interact through color weights: permuting one component's colors can change
/// the combined allocation size. Reaching the state limit is inconclusive.
InterferenceGraphWeightLimitResult
colorInterferenceGraphWithinWeightLimitExactly(
    const InterferenceGraph &graph, llvm::ArrayRef<std::uint64_t> vertexWeights,
    unsigned colorLimit, std::uint64_t weightLimit,
    std::uint64_t searchStateLimit);

enum class ExactInterferenceGraphWeightStatus {
  Optimal,
  AllocationWeightOverflow,
  SearchLimitReached,
};

/// Result of minimizing the sum of maximum vertex weight per used color.
struct ExactInterferenceGraphWeightColoring {
  ExactInterferenceGraphWeightStatus status =
      ExactInterferenceGraphWeightStatus::Optimal;
  llvm::SmallVector<unsigned> colors;
  unsigned colorCount = 0;
  std::uint64_t allocationWeight = 0;
  std::uint64_t exploredStateCount = 0;

  bool isOptimal() const {
    return status == ExactInterferenceGraphWeightStatus::Optimal;
  }
};

/// Proves the minimum weighted allocation within `colorLimit` colors.
///
/// `initialColors` must be a valid coloring and supplies the first upper bound.
/// Reaching `searchStateLimit` is inconclusive and returns no coloring.
ExactInterferenceGraphWeightColoring colorInterferenceGraphMinimumWeightExactly(
    const InterferenceGraph &graph, llvm::ArrayRef<std::uint64_t> vertexWeights,
    unsigned colorLimit, llvm::ArrayRef<unsigned> initialColors,
    std::uint64_t searchStateLimit);

/// Computes a deterministic assignment with the minimum resource-slot count.
///
/// Disconnected candidate groups have no conflicts between them, so they are
/// solved independently and reuse the same slots. The search begins at the
/// pairwise-conflict lower bound and tests increasing slot counts. Exhaustively
/// rejecting every smaller count proves that an `Optimal` result is minimum.
/// `SearchLimitReached` contains no assignment and cannot justify a capacity
/// diagnostic.
ExactInterferenceGraphColoring
colorInterferenceGraphExactly(const InterferenceGraph &graph,
                              std::uint64_t searchStateLimit);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_INTERFERENCEGRAPHCOLORING_H
