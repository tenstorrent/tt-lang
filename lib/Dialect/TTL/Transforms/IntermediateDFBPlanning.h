// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_INTERMEDIATEDFBPLANNING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_INTERMEDIATEDFBPLANNING_H

//===----------------------------------------------------------------------===//
// Intermediate DFB Planning
//===----------------------------------------------------------------------===//
//
// Intermediate DFB insertion rebuilds computes and rewrites consumer
// operands. Its materialization requirements and their correctness evidence
// must therefore be computed from immutable IR. This file provides a read-only
// plan so mutation cannot change which source values, releases, or output
// transactions justify a materialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "ttlang/Analysis/PlanningResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace mlir::tt::ttl {

class DFBValueLifetimeAnalysis;

/// Proven reason for materializing one consumer operand.
enum class IntermediateDFBReason {
  /// The consumer operation requires a DFB-attached operand.
  RequiredDFBOperand,

  /// The consumer's attached input may be released before it executes.
  DFBInputMayBeReleased,

  /// A partial expression would otherwise read a released DFB input.
  ExpressionInputMayBeReleased,

  /// Moving tensor evaluation to an output store would read a released input.
  ComputeOpInputMayBeReleased,

  /// One result is published through several reserves of the same DFB.
  MultipleOutputTransactions,

  /// A ttl.compute at the final output store would not dominate a consumer.
  ComputeOpWouldNotDominateUse,

  /// Creation would reorder instrumentation with another operation, so the
  /// consumer must read materialized storage at the tensor SSA frontier.
  ComputeOpInstrumentationWouldBeReordered,

  /// Another use requires this ttl.compute result to be materialized. All
  /// consumers must read that shared DFB because the result has no storage.
  ComputeResultHasMaterializedUse,

  /// A consumer cannot absorb a producer with its own standalone compute
  /// recipe, so the producer result must become a DFB input to that consumer.
  ComputeOpRequiresMaterializedInput,
};

/// Evidence supporting one intermediate DFB requirement.
struct IntermediateDFBEvidence {
  /// Proven condition requiring the materialization.
  IntermediateDFBReason reason;

  /// Values whose availability was queried for a release-based requirement.
  SmallVector<Value> inputs;

  /// Operation before which `inputs` must remain available.
  Operation *observation = nullptr;

  /// Output DFB whose reserve transactions require separation, when relevant.
  std::optional<Value> outputDFB;
};

/// One consumer operand replaced with an attached intermediate DFB value.
struct IntermediateDFBRequirement {
  /// Operation whose operand is replaced.
  Operation *consumer = nullptr;

  /// Operand position within `consumer`.
  unsigned operandIndex = 0;

  /// Original operand value used to verify the plan before mutation.
  Value value;

  /// Independent proofs that require this materialization.
  SmallVector<IntermediateDFBEvidence> evidence;
};

/// Whole-kernel operand decisions produced before DFB insertion modifies IR.
///
/// This state is the DFB analogue of One-Shot Bufferize's analysis state: each
/// decision marks one tensor operand whose value must be transferred through
/// a compiler DFB. Decisions grow monotonically and are consumed by a separate
/// materialization plan.
///
/// See the upstream analyze-then-apply contract:
/// https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/mlir/include/mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h
class DFBMaterializationAnalysisState {
public:
  /// Returns whether `operand` must be replaced with a DFB-attached value.
  bool requiresMaterialization(const OpOperand &operand) const;

  /// Records an independent proof that `operand` requires materialization.
  /// Returns true when this is the first decision for the operand.
  bool requireMaterialization(OpOperand &operand,
                              IntermediateDFBEvidence evidence);

  ArrayRef<IntermediateDFBRequirement> getRequirements() const {
    return requirements;
  }

  SmallVector<IntermediateDFBRequirement> takeRequirements() && {
    return std::move(requirements);
  }

private:
  SmallVector<IntermediateDFBRequirement> requirements;
};

/// One result of an existing compute that needs an additional DFB output.
struct ComputeResultDFBMaterializationPlan {
  /// Result position in `producer` and its corresponding output DFB.
  unsigned resultIndex = 0;

  /// Static tensor type used to construct the additional DFB output.
  RankedTensorType tensorType;

  /// Existing output DFB whose tile stores are replicated.
  Value sourceDFB;

  /// Indices into `IntermediateDFBPlan::getRequirements()`.
  SmallVector<unsigned> requirementIndices;
};

/// Atomic rebuild of one existing compute with additional DFB outputs.
struct ComputeDFBMaterializationPlan {
  /// Compute rebuilt atomically with all additional outputs in `results`.
  ComputeOp producer;

  /// Original results that require compiler-created DFB outputs.
  SmallVector<ComputeResultDFBMaterializationPlan> results;
};

/// One tensor definition stored to a compiler-created DFB.
struct StandaloneDFBMaterializationPlan {
  /// Tensor value routed through a compiler-created DFB.
  Value source;

  /// Static tensor type used by the new DFB lifecycle.
  RankedTensorType tensorType;

  /// Operation after which the compiler DFB store is inserted.
  ///
  /// This is a later output store that properly dominates every rewritten
  /// consumer, or the source definition when there is no applicable output
  /// plan or every source use is rewritten. Each condition ensures final
  /// `ComputeOp` creation executes before the compiler DFB wait.
  Operation *insertionAnchor = nullptr;

  /// Indices into `IntermediateDFBPlan::getRequirements()`.
  SmallVector<unsigned> requirementIndices;
};

/// Immutable intermediate DFB requirements for one kernel.
///
/// Operation and value handles remain valid only while the analyzed kernel is
/// unchanged. The insertion pass consumes the complete plan after all
/// requirements have reached a fixed point.
class IntermediateDFBPlan {
public:
  ArrayRef<IntermediateDFBRequirement> getRequirements() const {
    return requirements;
  }

  /// Returns compute rebuilds in SSA producer-before-consumer order.
  ArrayRef<ComputeDFBMaterializationPlan> getComputeMaterializations() const {
    return computeMaterializations;
  }

  ArrayRef<StandaloneDFBMaterializationPlan>
  getStandaloneMaterializations() const {
    return standaloneMaterializations;
  }

private:
  friend class IntermediateDFBPlanner;
  explicit IntermediateDFBPlan(
      SmallVector<IntermediateDFBRequirement> requirements,
      SmallVector<ComputeDFBMaterializationPlan> computeMaterializations,
      SmallVector<StandaloneDFBMaterializationPlan> standaloneMaterializations);

  SmallVector<IntermediateDFBRequirement> requirements;
  SmallVector<ComputeDFBMaterializationPlan> computeMaterializations;
  SmallVector<StandaloneDFBMaterializationPlan> standaloneMaterializations;
};

/// Computes intermediate DFB requirements without modifying IR.
///
/// Every release result is conservative: an unproven value lifetime requires
/// the affected consumer operand to be materialized. Repeated analysis
/// terminates because each iteration can add only an operand from the finite
/// input kernel.
class IntermediateDFBPlanner {
public:
  IntermediateDFBPlanner(func::FuncOp kernel,
                         const DFBValueLifetimeAnalysis &lifetimes)
      : kernel(kernel), lifetimes(lifetimes) {}

  PlanningResult<IntermediateDFBPlan> build() const;

private:
  PlanningResult<IntermediateDFBPlan> buildMaterializationRecords(
      SmallVector<IntermediateDFBRequirement> requirements) const;

  func::FuncOp kernel;
  const DFBValueLifetimeAnalysis &lifetimes;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_INTERMEDIATEDFBPLANNING_H
