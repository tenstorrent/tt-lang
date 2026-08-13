// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEOPCREATIONPLANNING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEOPCREATIONPLANNING_H

//===----------------------------------------------------------------------===//
// ComputeOp Creation Planning
//===----------------------------------------------------------------------===//
//
// ComputeOp creation moves tensor evaluation to an output-store position and
// may absorb several source operations. Legality and recipe decisions
// must therefore be computed from immutable IR: a rewrite cannot safely infer
// inputs, DFB availability, indexing, publication, or instrumentation placement
// after an earlier rewrite has changed those facts. The records in this file
// freeze those decisions for mechanical application during one pass
// invocation. They are invalid after their source kernel is modified outside
// the recorded application order.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Analysis/PlanningResult.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/ComputeTarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir::tt::ttl {

class DFBValueLifetimeAnalysis;

/// Strategy for creating a `ComputeOp`, selected before IR mutation begins.
enum class ComputeOpCreationKind {
  /// Lower the source operation with its declared DFB input operands.
  Direct,

  /// Absorb a fusable expression and read its traced DFB roots.
  Fused,

  /// Replace an operation with a semantically equivalent input value.
  Elide,
};

/// Reason a structurally complete creation is not selected.
enum class ComputeOpCreationRejectionKind {
  /// The creation satisfies every common legality condition.
  None,

  /// The operation or expression has no supported tile recipe.
  UnsupportedCandidate,

  /// A supported operation requires a DFB input not present in the current IR.
  UnmaterializedInput,

  /// Valid output control flow cannot be represented by one current plan.
  UnsupportedOutputPublication,

  /// One output DFB would publish several reserve transactions.
  MultipleOutputTransactions,

  /// At least one input may have been released at the insertion anchor.
  InputMayBeReleased,

  /// The insertion anchor does not dominate a surviving result use.
  ResultUseNotDominated,

  /// Creation would move instrumentation across a non-reorderable operation.
  InstrumentationWouldBeReordered,

  /// An enclosing rejected creation must be split before converting this
  /// source.
  DeferredDependency,
};

/// Non-fatal code-generation change caused by preserving source semantics.
enum class ComputeOpCreationWarningKind {
  /// Instrumentation prevents combining a matmul and its accumulator add.
  InstrumentationPreventsMatmulAccumulator,
};

/// Warning emitted before applying a legal `ComputeOp` creation plan.
struct ComputeOpCreationWarning {
  /// Instrumentation operation that caused the code-generation change.
  Operation *operation = nullptr;

  /// Typed warning condition.
  ComputeOpCreationWarningKind kind;
};

StringRef getComputeOpCreationWarningMessage(ComputeOpCreationWarningKind kind);

/// Operation-specific tile recipe selected before IR mutation.
enum class ComputeOpCreationRecipe {
  /// Emit the tile operation corresponding to a unary or binary tensor op.
  Elementwise,

  /// Emit inter-tile indexing and, when required, an intra-tile broadcast.
  BlockBroadcast,

  /// Emit one block matrix multiplication.
  Matmul,

  /// Emit one reduction with its scaler input.
  Reduce,

  /// Emit elementwise multiplication by a compile-time scalar.
  MulUnaryConst,

  /// Emit a constant tile.
  Fill,

  /// Emit an element-type conversion.
  Typecast,

  /// Emit a width-height tile transpose.
  Transpose,

  /// Emit the recorded expression-level tile recipes.
  Fused,

  /// Replace an identity operation with its input without creating a
  /// `ComputeOp`.
  Elide,
};

/// Indexing and iteration semantics for one created `ComputeOp`.
///
/// Recording these attributes prevents application patterns from deriving a
/// different iteration domain or broadcast/reduction mapping after mutation.
struct ComputeIterationPlan {
  /// One indexing map for each compute input, in input order.
  SmallVector<AffineMap> inputMaps;

  /// Indexing map shared by every output DFB in the publication plan.
  AffineMap outputMap;

  /// Iterator semantics in compute iteration-domain order.
  SmallVector<utils::IteratorType> iteratorTypes;
};

/// Placement of instrumentation copied into a created `ComputeOp` body.
///
/// Fusion relocates evaluation into a new region. This record is required to
/// preserve each observation point relative to the absorbed operations.
struct ComputeInstrumentationPlacement {
  /// Instrumentation operation copied into the compute body and then erased.
  Operation *operation = nullptr;

  /// Preceding moved operation or output store, or null for a leading effect.
  Operation *after = nullptr;
};

/// Affine indexing role of one external input to a fused expression.
enum class FusedInputRole {
  /// Elementwise input indexed by the output's parallel dimensions.
  Parallel,

  /// Matmul left operand indexed by `[M, K]`.
  MatmulLeft,

  /// Matmul right operand indexed by `[K, N]`.
  MatmulRight,

  /// Transposed matmul right operand indexed by `[N, K]`.
  MatmulTransposedRight,
};

/// Tile recipe selected for one operation in a fused expression.
enum class FusedOperationRecipe {
  /// Emit the tile operation corresponding to the source tensor operation.
  TileOperation,

  /// Reuse the input tile; the broadcast is represented by its indexing map.
  InterTileBroadcast,

  /// Emit `ttl.tile_bcast` for a broadcast within one hardware tile.
  TileBroadcast,

  /// Emit a standalone `ttl.tile_matmul_block`.
  Matmul,

  /// Emit no operation because a later accumulator recipe folds this matmul.
  DeferredMatmul,

  /// Emit one accumulating matmul for a previously deferred matmul and add.
  MatmulAccumulator,

  /// Emit no multiply because a following exp consumes its constant as scale.
  DeferredExpScale,
};

/// One tensor operand consumed by a fused tile-level recipe.
struct FusedOperationOperand {
  /// Tensor value named by the source expression.
  Value value;

  /// Compute input slot for an external root; absent for an earlier fused
  /// result produced inside the compute body.
  std::optional<unsigned> rootInputIndex;
};

/// Hardware configuration captured for an exp tile recipe.
struct ExpFlagsPlan {
  BoolAttr approx;
  FloatAttr scale;
  InputClampingAttr inputClamping;
  IntegerAttr iterations;
};

/// Immutable tile recipe for one absorbed tensor operation.
///
/// Each operand identifies either an external compute input slot or a prior
/// fused result. A deferred matmul is emitted by its `MatmulAccumulator` user
/// so the hardware accumulator implements `accumulator + lhs * rhs` without a
/// separate add. Recording these relations avoids consulting use-lists after
/// earlier creations have removed users.
struct FusedOperationPlan {
  /// Tensor operation represented by this recipe.
  Operation *source = nullptr;

  /// Original operands used to detect invalidation before application.
  SmallVector<Value> sourceOperands;

  /// Tile recipe selected by the planner.
  FusedOperationRecipe recipe = FusedOperationRecipe::TileOperation;

  /// Tensor dependencies and their external compute input slots.
  SmallVector<FusedOperationOperand> operands;

  /// Exact tile result type recorded from the source result.
  ttcore::TileType resultTileType;

  /// Hardware broadcast kind for a tile-broadcast recipe.
  std::optional<BcastType> tileBroadcast;

  /// Matmul emitted by a later accumulator recipe.
  std::optional<MatmulOp> foldedMatmul;

  /// Non-matmul operand used to initialize an accumulating matmul.
  std::optional<Value> accumulator;

  /// Whether a matmul recipe reads its right operand transposed.
  bool transposeRhs = false;

  /// Scalar used by fill and multiply-constant recipes.
  std::optional<FloatAttr> constantValue;

  /// Hardware flags used by an exp recipe.
  std::optional<ExpFlagsPlan> expFlags;
};

/// Stable identity of one source-result use during plan application.
///
/// Operand storage may move when another rewrite changes an owner's operand
/// list. The owner and operand number identify the use without retaining an
/// `OpOperand *` into that storage.
struct ComputeOpCreationUse {
  /// Operation containing the recorded operand.
  Operation *owner = nullptr;

  /// Operand position within `owner`.
  unsigned operandIndex = 0;

  bool matches(OpOperand &operand) const {
    return owner == operand.getOwner() &&
           operandIndex == operand.getOperandNumber();
  }
};

/// Stores and optional publication for one acquired output DFB view.
///
/// Every store uses a view derived from `acquire`. Reserve-backed transactions
/// may carry a following `ttl.cb_push`. Wait-backed transactions replace an
/// occupied consumer slot and retain the existing `ttl.cb_pop` release.
struct OutputDFBTransaction {
  /// DFB whose acquired pages are written by the transaction.
  Value dfb;

  /// `ttl.cb_reserve` or `ttl.cb_wait` that provides every store view.
  Operation *acquire = nullptr;

  /// Stores using views derived from `acquire`, in block order.
  SmallVector<StoreOp> stores;

  /// First matching `ttl.cb_push` after every reserve-backed store, when one
  /// exists. Wait-backed transactions never publish.
  std::optional<CBPushOp> push;

  bool isWaitedMutation() const { return isa<CBWaitOp>(acquire); }
};

/// Read-only plan for storing and pushing one created `ComputeOp` result.
///
/// Stores, transactions, and pushes retain block order. `dfbs` contains each
/// output DFB once in first-store order. A DFB in `multiTransactionDFBs` is
/// written through more than one acquisition; combining those transactions
/// into one compute would lose the acquisition boundary.
struct OutputPublicationPlan {
  /// Unique output DFBs in first-store order.
  SmallVector<Value> dfbs;

  /// All stores of the source result, in block order.
  SmallVector<StoreOp> stores;

  /// Existing `ttl.cb_push` operations matched to transactions, in block order.
  SmallVector<CBPushOp> pushes;

  /// Acquire-delimited output transactions in first-store order.
  SmallVector<OutputDFBTransaction> transactions;

  /// DFBs written through more than one acquisition operation.
  SmallVector<Value> multiTransactionDFBs;

  /// Operation immediately before which the created `ComputeOp` executes.
  ///
  /// The final store is selected because all output stores are in one block
  /// and each acquisition dominates its store. It is therefore dominated by
  /// every acquisition needed to create the `ComputeOp` outputs.
  StoreOp insertionAnchor;

  bool hasMultipleTransactionsForOneDFB() const {
    return !multiTransactionDFBs.empty();
  }

  bool hasMultipleTransactions(Value dfb) const {
    return llvm::is_contained(multiTransactionDFBs, dfb);
  }
};

/// Immutable proof for one straight-line replacement of a waited DFB block.
///
/// The initial contract accepts one complete one-block consumer acquisition
/// whose original value is a direct input to the replacement compute. All
/// fields refer to the unmodified kernel used to build the creation plan.
struct WaitedDFBMutationPlan {
  /// Tensor store that replaces the acquired pages.
  StoreOp store;

  /// Consumer acquisition that owns the destination pages.
  CBWaitOp wait;

  /// Consumer release after every read of the replacement generation.
  CBPopOp release;

  /// Logical DFB whose read and write cursors coincide while the ring is full.
  Value dfb;

  /// Complete-ring transaction and capacity, in tiles.
  int64_t transactionTiles = 0;
  int64_t capacityTiles = 0;
};

/// Reason output publication cannot be planned for a legal source operation.
enum class OutputPublicationRejectionKind {
  /// The source does not have exactly one tensor result.
  UnsupportedResultCount,

  /// The source result has no `ttl.store` user.
  MissingStore,

  /// Result stores do not share one block and insertion position.
  StoresInDifferentBlocks,
};

/// Typed explanation for an unsupported output-publication candidate.
struct OutputPublicationRejection {
  /// Candidate whose output publication is unsupported.
  Operation *source = nullptr;

  /// Unsupported publication condition.
  OutputPublicationRejectionKind kind =
      OutputPublicationRejectionKind::UnsupportedResultCount;

  /// Explanation suitable for planner debug output.
  std::string message;
};

/// Returns whether creation preserves one use of the source result.
///
/// A recorded output-store tensor use is erased by creation. Every other use
/// must be dominated by `insertionAnchor` because the compute is inserted
/// immediately before that operation. Incomparable or unmodeled region order
/// returns false, which requires materialization and preserves correctness.
bool isComputeOpCreationUsePreserved(const OutputPublicationPlan &outputs,
                                     OpOperand &use,
                                     const DominanceInfo &dominanceInfo);

/// Returns whether creation replaces `source` without moving its evaluation.
///
/// A typecast whose input and result types match is replaced at its original
/// position. Other supported recipes execute at their output insertion anchor.
bool isComputeOpCreationElision(Operation *source);

/// Returns whether `source` has an operation-specific ttl.compute recipe.
///
/// The query assumes direct tensor operands will become DFB-attached and does
/// not require an output store. It proves only that materializing the source
/// result can separate this operation from a fused consumer; creation
/// planning still validates input availability and output publication.
bool hasStandaloneComputeOpCreationRecipe(Operation *source);

/// Builds the output transaction plan for `source` without modifying IR.
///
/// `source` must have exactly one result with at least one `ttl.store` user,
/// all stores must be in one block, and every store view must originate from
/// `ttl.cb_reserve` or `ttl.cb_wait`. Valid SSA ensures each acquisition
/// dominates its stores. These conditions provide one insertion position and
/// explicit transaction identities. Unsupported valid forms return a typed
/// rejection. A malformed store/publication transaction returns an invalid-IR
/// diagnostic.
PlanningResult<OutputPublicationPlan, OutputPublicationRejection>
buildOutputPublicationPlan(Operation *source);

/// Resolves publication operations for previously analyzed transactions.
///
/// Rewriting another source may relocate a push shared by multiple store
/// sequences. A store consumes only its source's SSA result, so rewriting a
/// different source cannot erase that store or its reserve. Resolving the first
/// current push after each store therefore preserves the analyzed transaction
/// contract without repeating creation or lifetime analysis.
PlanningResult<OutputPublicationPlan>
resolveOutputPublicationOperations(const OutputPublicationPlan &analyzed);

/// One operation that a creation would reorder with instrumentation.
struct ComputeOpCreationInstrumentationBoundary {
  /// Operation whose relative order must remain unchanged.
  Operation *operation = nullptr;

  /// Uses that would move from after `operation` into the created `ComputeOp`.
  SmallVector<ComputeOpCreationUse> crossingUses;
};

/// Finds exact SSA uses needed to preserve instrumentation order.
///
/// The query applies upstream `isPure` semantics to operations that remain
/// outside the created `ComputeOp`. When recorded movable instrumentation
/// precedes such an operation but the created `ComputeOp` would follow it,
/// each returned use identifies a consumer operand to materialize. The
/// producer computation then remains before that operation, which preserves
/// the original observation order. Output reserves are excluded because
/// creation must execute after the reserve supplying each formal output view.
///
/// `isMaterializationPlanned` identifies uses already selected as split
/// points. `outputs` must be a valid publication plan for the source's single
/// tensor result. Failure means the source has no supported direct or fused
/// compute semantics, or a source outside the publication block cannot be
/// recomputed under MLIR's purity contract.
FailureOr<SmallVector<ComputeOpCreationInstrumentationBoundary>>
collectComputeOpCreationInstrumentationBoundaries(
    Operation *source, const OutputPublicationPlan &outputs,
    llvm::function_ref<bool(OpOperand &)> isMaterializationPlanned);

/// Returns current-storage inputs that constrain future `ComputeOp` creation.
///
/// Direct creation contributes every unmaterialized tensor operand in
/// lowering order, including duplicates. When direct creation is
/// unavailable, fusable expressions contribute their distinct unmaterialized
/// DFB roots. Failure means `source` has no known `ComputeOp` creation
/// semantics. An empty result is valid when no existing storage constrains
/// execution, including `ttl.fill` and creations whose inputs are all
/// materialized.
///
/// `isMaterializationPlanned` identifies operands anywhere in the fusable
/// expression that will be replaced by compiler-DFB values before creation.
/// The replacement DFB is released after its planned uses, so the original
/// operand's storage lifetime does not constrain the creation position.
FailureOr<SmallVector<Value>> collectComputeOpCreationLifetimeInputs(
    Operation *source,
    llvm::function_ref<bool(OpOperand &)> isMaterializationPlanned);

/// Complete common legality plan for creating a `ttl.compute` from one source.
///
/// The plan refers to source IR. Dependency-safe application preserves every
/// traced operation until its plan is consumed. Rewrites may relocate shared
/// pushes; `resolveOutputPublicationOperations` remaps only those publication
/// operations while retaining the proven reserve/store transactions.
struct ComputeOpCreationPlan {
  /// Source operation represented by this plan.
  Operation *source = nullptr;

  /// Operands expected when this plan is applied.
  ///
  /// Most plans record the analyzed operands. An identity-typecast plan records
  /// the input produced by its prerequisite identity elisions instead.
  SmallVector<Value> applicationOperands;

  /// Complete ordered tensor inputs read by the created `ComputeOp`.
  SmallVector<Value> inputs;

  /// Exact tile type for each entry in `inputs`.
  ///
  /// Empty for an identity elision, which creates no `ComputeOp`.
  SmallVector<ttcore::TileType> inputTileTypes;

  /// Indexing role for each fused input. A value appears more than once when
  /// distinct uses require different affine maps. Empty for direct creation.
  SmallVector<FusedInputRole> fusedInputRoles;

  /// Creation strategy whose input contract is recorded in `inputs`.
  ComputeOpCreationKind kind = ComputeOpCreationKind::Direct;

  /// Operation-specific tile recipe used by mechanical application.
  ComputeOpCreationRecipe recipe = ComputeOpCreationRecipe::Elementwise;

  /// Result tensor type used by the created `ComputeOp`.
  RankedTensorType resultType;

  /// Exact result tile type recorded during planning.
  ///
  /// Null for an identity elision, which creates no `ComputeOp`.
  ttcore::TileType resultTileType;

  /// Precomputed indexing maps and iterator types.
  ComputeIterationPlan iteration;

  /// Intra-tile broadcast operation, absent for inter-tile-only broadcast.
  std::optional<BcastType> tileBroadcast;

  /// Hardware reduction dimension for a reduce recipe.
  std::optional<ttkernel::ReduceDim> reduceDimension;

  /// Reduction function for a reduce recipe.
  std::optional<ReduceType> reduceType;

  /// Transposition selected for a direct matmul recipe.
  bool transposeRhs = false;

  /// Scalar constant used by fill and multiply-constant recipes.
  std::optional<FloatAttr> constantValue;

  /// Hardware flags used by a direct exp recipe.
  std::optional<ExpFlagsPlan> expFlags;

  /// Expression absorbed by fused creation. Empty for direct creation.
  FusionTraceResult trace;

  /// Instrumentation copied into the created `ComputeOp` in source order.
  SmallVector<ComputeInstrumentationPlacement> instrumentation;

  /// Non-fatal code-generation changes reported before application.
  SmallVector<ComputeOpCreationWarning> warnings;

  /// Tile-level recipes in expression dependency order.
  SmallVector<FusedOperationPlan> fusedOperations;

  /// Original result uses used to verify plan/application consistency.
  SmallVector<ComputeOpCreationUse> resultUses;

  /// Result uses that must survive earlier overlapping creations.
  ///
  /// An outer fused creation may include this source but cannot erase it when
  /// one of these uses remains. Application verifies the recorded use before
  /// consuming this plan, making the dependency-safe ordering assumption
  /// explicit and fail-closed if another rewrite invalidates it.
  SmallVector<ComputeOpCreationUse> preservingUses;

  /// Uses that a verified earlier creation must remove.
  ///
  /// The compute insertion anchor does not dominate these consumers in the
  /// original IR. Kernel planning may still select the creation when each
  /// consumer is an earlier fused source that erases the recorded operand use.
  /// Application fails closed if any recorded use remains.
  SmallVector<ComputeOpCreationUse> preCreationRemovedUses;

  /// Reserve, store, and publication transactions affected by creation.
  OutputPublicationPlan outputs;

  /// Consumer-owned DFB replacements proved before IR mutation.
  SmallVector<WaitedDFBMutationPlan> waitedMutations;

  /// Typed reason for an unselected structurally complete creation.
  ComputeOpCreationRejectionKind rejectionKind =
      ComputeOpCreationRejectionKind::None;

  /// Violated common check when `rejectionKind` is not `None`.
  std::string rejectionReason;

  bool isLegal() const {
    return rejectionKind == ComputeOpCreationRejectionKind::None;
  }
};

/// Typed explanation for a source rejected during `ComputeOp` creation
/// planning.
struct ComputeOpCreationRejection {
  /// Source operation whose `ComputeOp` creation was rejected.
  Operation *source = nullptr;

  /// Common or operation-specific rejection category.
  ComputeOpCreationRejectionKind kind =
      ComputeOpCreationRejectionKind::UnsupportedCandidate;

  /// Explanation suitable for planner debug output.
  std::string message;

  /// Complete candidate retained when dependency selection needs its trace.
  std::optional<ComputeOpCreationPlan> candidate;
};

/// Immutable plan for a DFB-to-DFB passthrough store compute.
///
/// A passthrough store creates a `ComputeOp` even though the store has no
/// tensor result and is not a normal creation candidate. Keeping it in the
/// kernel plan ensures its input lifetime, indexing, output transaction, and
/// affected associations are validated before any candidate rewrite begins.
struct PassthroughStorePlan {
  /// Store replaced by the passthrough compute.
  StoreOp store;

  /// Tensor read by `store` in the analyzed IR.
  Value originalInput;

  /// DFB-backed tensor read after planned identity typecasts are elided.
  Value input;

  /// Producer acquisition that provides `outputView`.
  CBReserveOp reserve;

  /// Reserved tensor view written by the tile store.
  Value outputView;

  /// DFB associated with `outputView`.
  Value outputDFB;

  /// Tensor type shared by the passthrough input and result.
  RankedTensorType tensorType;

  /// Exact tile type shared by the passthrough input and result.
  ttcore::TileType tileType;

  /// Identity iteration used by the passthrough compute.
  ComputeIterationPlan iteration;

  /// Associations whose results must be replaced by the compute result.
  SmallVector<AttachCBOp> outputAssociations;
};

/// Kernel-local creation plans computed before conversion modifies IR.
///
/// Every structurally supported source operation receives one complete plan.
/// Plans rejected by common legality checks remain available for dependency
/// analysis, but rewrite patterns can access only legal plans. This prevents
/// producer creation from destroying an expression that later DFB
/// materialization must split. Application patterns select only the recorded
/// tile operation; they cannot derive different inputs, indexing maps, or
/// iterator semantics, or bypass DFB lifetime and output transaction checks.
class KernelComputeOpCreationPlan {
public:
  /// Returns the legal creation plan for `source`, when one exists.
  FailureOr<const ComputeOpCreationPlan *> get(Operation *source) const;

  /// Returns why `source` has no legal creation plan.
  StringRef getRejectionReason(Operation *source) const;

  /// Returns the typed rejection for `source`, when planning recorded one.
  std::optional<ComputeOpCreationRejectionKind>
  getRejectionKind(Operation *source) const;

  /// Returns whether `source` has a complete creation record.
  bool hasCreationRecord(Operation *source) const {
    return creations.contains(source);
  }

  /// Returns candidates in dependency-safe conversion order. A candidate is
  /// listed before any other candidate whose expression it absorbs.
  ArrayRef<Operation *> getCreationOrder() const { return creationOrder; }

  /// Returns every analyzed source operation in kernel walk order.
  ArrayRef<Operation *> getAnalyzedSources() const { return analyzedSources; }

  /// Returns the creation record for an analyzed source, including rejected
  /// records. The caller must select `source` from `getAnalyzedSources()`.
  const ComputeOpCreationPlan &getAnalyzedCreation(Operation *source) const;

  /// Returns the passthrough-store plan for `store`, when selected.
  FailureOr<const PassthroughStorePlan *> get(StoreOp store) const;

  /// Returns stores not assigned to a selected creation or passthrough plan.
  ///
  /// Producer creation may leave these stores for intermediate DFB
  /// materialization. Final conversion must reject them before mutation.
  ArrayRef<StoreOp> getUnassignedStores() const { return unassignedStores; }

  /// Returns the diagnostic that explains why `store` is unassigned.
  ///
  /// A source-level rejection from an attempted `ComputeOp` creation takes
  /// precedence because it explains why the store remains unassigned. A
  /// generic unsupported-candidate rejection does not override a more precise
  /// passthrough-store failure. The diagnostic is anchored at the source only
  /// when an unmaterialized input is the relevant operation; other failures
  /// remain anchored at the store that final conversion cannot lower.
  PlanningDiagnostic getUnassignedStoreDiagnostic(StoreOp store) const;

  /// Returns the rejected source operation whose reason explains why `store`
  /// remains unassigned. Unsupported tensor sources return `std::nullopt`
  /// because their stores are governed by passthrough-specific diagnostics.
  std::optional<Operation *> getUnassignedStoreSource(StoreOp store) const;

private:
  friend class ComputeOpCreationPlanner;

  DenseMap<Operation *, ComputeOpCreationPlan> creations;
  DenseMap<Operation *, PassthroughStorePlan> passthroughStores;
  DenseMap<Operation *, std::string> rejectionReasons;
  DenseMap<Operation *, ComputeOpCreationRejectionKind> rejectionKinds;
  SmallVector<Operation *> analyzedSources;
  SmallVector<Operation *> creationOrder;
  SmallVector<StoreOp> unassignedStores;
};

/// Builds complete kernel `ComputeOp` creation plans without modifying IR.
///
/// The immutable plan and separate application follow LLVM VPlan's planning
/// model while recording TTL-specific DFB transactions and tile recipes:
/// https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/llvm/lib/Transforms/Vectorize/VPlan.h
class ComputeOpCreationPlanner {
public:
  ComputeOpCreationPlanner(func::FuncOp kernel,
                           const DFBValueLifetimeAnalysis &lifetimes,
                           const ComputeTargetEnvironment &target)
      : kernel(kernel), lifetimes(lifetimes), target(target) {}

  PlanningResult<KernelComputeOpCreationPlan> build() const;

private:
  func::FuncOp kernel;
  const DFBValueLifetimeAnalysis &lifetimes;
  const ComputeTargetEnvironment &target;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_COMPUTEOPCREATIONPLANNING_H
