// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/ExecutionCountAnalysis.h"

#include "ttlang/Analysis/LoopIterationUtils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>

namespace mlir::tt::execution_count_detail {

struct BlockFlowGraph;

/// A block and its possible control-flow edges for one induction environment.
struct BlockFlowNode {
  BlockFlowGraph *parent = nullptr;
  Block *block = nullptr;
  SmallVector<BlockFlowNode *> successors;
  SmallVector<BlockFlowNode *> predecessors;

  /// Position in the parent's node list. LLVM's dominator tree indexes its
  /// internal vectors by this, so it must be dense over [0, nodes.size()).
  unsigned number = 0;

  /// Supply the parent required by LLVM's generic dominator tree.
  BlockFlowGraph *getParent() const { return parent; }

  /// Print the corresponding MLIR block in LLVM graph diagnostics.
  void printAsOperand(llvm::raw_ostream &output, bool printType) const {
    block->printAsOperand(output, printType);
  }
};

/// The possible block CFG for one region and induction environment.
struct BlockFlowGraph {
  using NodeList = SmallVector<BlockFlowNode>;

  NodeList nodes;
  BlockFlowNode *entry = nullptr;

  /// Supply the entry required by LLVM's generic dominator tree.
  BlockFlowNode &front() { return *entry; }
};

} // namespace mlir::tt::execution_count_detail

namespace llvm {

/// Restrict LLVM graph traversal to successors possible in this context.
template <>
struct GraphTraits<mlir::tt::execution_count_detail::BlockFlowNode *> {
  using NodeRef = mlir::tt::execution_count_detail::BlockFlowNode *;
  using ChildIteratorType = SmallVector<NodeRef>::const_iterator;

  static NodeRef getEntryNode(NodeRef node) { return node; }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->successors.begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->successors.end();
  }
  static unsigned getNumber(NodeRef node) { return node->number; }
};

/// Supply the numbering the dominator tree reads through a const node.
template <>
struct GraphTraits<const mlir::tt::execution_count_detail::BlockFlowNode *> {
  using NodeRef = const mlir::tt::execution_count_detail::BlockFlowNode *;
  using ChildIteratorType = SmallVector<
      mlir::tt::execution_count_detail::BlockFlowNode *>::const_iterator;

  static NodeRef getEntryNode(NodeRef node) { return node; }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->successors.begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->successors.end();
  }
  static unsigned getNumber(NodeRef node) { return node->number; }
};

/// Provide the reverse contextual edges required by post-dominance.
template <>
struct GraphTraits<Inverse<mlir::tt::execution_count_detail::BlockFlowNode *>> {
  using NodeRef = mlir::tt::execution_count_detail::BlockFlowNode *;
  using ChildIteratorType = SmallVector<NodeRef>::const_iterator;

  static NodeRef getEntryNode(Inverse<NodeRef> graph) { return graph.Graph; }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->predecessors.begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->predecessors.end();
  }
};

/// Let post-dominance inspect every node, including disconnected blocks.
template <>
struct GraphTraits<mlir::tt::execution_count_detail::BlockFlowGraph *>
    : GraphTraits<mlir::tt::execution_count_detail::BlockFlowNode *> {
  using GraphType = mlir::tt::execution_count_detail::BlockFlowGraph *;
  using NodeRef = mlir::tt::execution_count_detail::BlockFlowNode *;
  using nodes_iterator = pointer_iterator<
      mlir::tt::execution_count_detail::BlockFlowGraph::NodeList::iterator>;

  static NodeRef getEntryNode(GraphType graph) { return graph->entry; }
  static nodes_iterator nodes_begin(GraphType graph) {
    return nodes_iterator(graph->nodes.begin());
  }
  static nodes_iterator nodes_end(GraphType graph) {
    return nodes_iterator(graph->nodes.end());
  }

  static unsigned getMaxNumber(GraphType graph) { return graph->nodes.size(); }
  /// Numbers are assigned once at graph construction and never reassigned, so
  /// the dominator tree can never observe a stale numbering.
  static unsigned getNumberEpoch(GraphType) { return 0; }
};

} // namespace llvm

namespace mlir::tt {

namespace {

/// One nesting level between the root region and the queried operation. The
/// block contains either the query or its next inner enclosing operation.
struct ControlFrame {
  Operation *parent = nullptr;
  Region *region = nullptr;
  Block *targetBlock = nullptr;
};

/// Record whether each successor edge is possible, ordered by block and
/// successor position.
using BlockFlowKey = SmallVector<std::uint8_t>;

/// Exact block counts for one possible block CFG.
struct BlockCountResult {
  /// Map every block to zero, one, or an unknown invocation count.
  llvm::DenseMap<Block *, std::optional<std::uint64_t>> blockCounts;
};

/// Return whether the parent may transition between the two child regions.
bool isRegionReachable(RegionBranchOpInterface branch, Region &sourceRegion,
                       Region &targetRegion) {
  SmallVector<Region *> worklist{&sourceRegion};
  llvm::SmallPtrSet<Region *, 4> visited;
  visited.insert(&sourceRegion);
  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(*region, successors);
    for (RegionSuccessor successor : successors) {
      Region *successorRegion = successor.getSuccessor();
      if (!successorRegion) {
        continue;
      }
      if (successorRegion == &targetRegion) {
        return true;
      }
      if (visited.insert(successorRegion).second) {
        worklist.push_back(successorRegion);
      }
    }
  }
  return false;
}

} // namespace

class ExecutionCountAnalysis::Impl {
public:
  Impl(Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator,
       RegionInvocationCountEvaluator regionInvocationCountEvaluator,
       Options options)
      : rootRegion(rootRegion),
        symbolValueEvaluator(std::move(symbolValueEvaluator)),
        regionInvocationCountEvaluator(
            std::move(regionInvocationCountEvaluator)),
        options(options) {
    Operation *parent = rootRegion.getParentOp();
    if (!parent) {
      return;
    }
    dataFlowSolver = std::make_unique<DataFlowSolver>();
    dataflow::loadBaselineAnalyses(*dataFlowSolver);
    if (failed(dataFlowSolver->initializeAndRun(parent))) {
      dataFlowSolver.reset();
    }
  }

  std::optional<std::uint64_t> getExecutionCount(Operation *operation) {
    if (!operation) {
      return std::nullopt;
    }
    Block *operationBlock = operation->getBlock();
    if (!operationBlock) {
      return std::nullopt;
    }
    // Every operation in a block executes once per block invocation, so reuse
    // the control-flow proof for all operations in that block.
    auto cached = blockExecutionCountCache.find(operationBlock);
    if (cached != blockExecutionCountCache.end()) {
      return cached->second;
    }

    SmallVector<ControlFrame> frames;
    Block *rootBlock = nullptr;
    if (!collectControlFrames(operation, frames, rootBlock)) {
      blockExecutionCountCache.try_emplace(operationBlock, std::nullopt);
      return std::nullopt;
    }

    LoopInductionBindings inductionValues;
    EnumerationBudget enumerationBudget(options.maxEnumeratedIterations);
    std::optional<std::uint64_t> maybeRootBlockCount =
        getExactBlockInvocationCount(rootRegion, rootBlock, inductionValues);
    std::optional<std::uint64_t> maybeCount;
    if (!maybeRootBlockCount || *maybeRootBlockCount == 0) {
      maybeCount = maybeRootBlockCount;
    } else {
      std::optional<std::uint64_t> maybeNestedCount =
          countExecutions(frames, 0, inductionValues, enumerationBudget);
      maybeCount = maybeNestedCount
                       ? llvm::checkedMulUnsigned(*maybeRootBlockCount,
                                                  *maybeNestedCount)
                       : std::nullopt;
    }
    blockExecutionCountCache.try_emplace(operationBlock, maybeCount);
    return maybeCount;
  }

private:
  bool collectControlFrames(Operation *operation,
                            SmallVectorImpl<ControlFrame> &frames,
                            Block *&rootBlock) const {
    Region *parentRegion = operation->getParentRegion();
    if (!parentRegion || !rootRegion.isAncestor(parentRegion)) {
      return false;
    }

    for (Operation *child = operation;
         child->getParentRegion() != &rootRegion;) {
      Region *region = child->getParentRegion();
      Operation *parent = region->getParentOp();
      if (!parent) {
        return false;
      }
      frames.push_back({parent, region, child->getBlock()});
      child = parent;
    }
    Operation *rootChild = frames.empty() ? operation : frames.back().parent;
    rootBlock = rootChild->getBlock();
    if (!rootBlock || rootBlock->getParent() != &rootRegion) {
      return false;
    }
    std::reverse(frames.begin(), frames.end());
    return true;
  }

  IntegerExpressionEvaluator
  createIntegerEvaluator(const LoopInductionBindings &inductionValues) const {
    return createLoopIntegerEvaluator(inductionValues, [this](Value value) {
      return evaluateContextValue(value);
    });
  }

  /// Evaluate values established outside the current induction bindings.
  std::optional<llvm::APInt> evaluateContextValue(Value value) const {
    if (std::optional<llvm::APInt> maybeConstant =
            getDataFlowIntegerConstant(value)) {
      return maybeConstant;
    }
    return symbolValueEvaluator ? symbolValueEvaluator(value) : std::nullopt;
  }

  /// Return an integer constant propagated through block and region arguments.
  std::optional<llvm::APInt> getDataFlowIntegerConstant(Value value) const {
    if (!dataFlowSolver) {
      return std::nullopt;
    }
    using ConstantLattice = dataflow::Lattice<dataflow::ConstantValue>;
    const auto *lattice = dataFlowSolver->lookupState<ConstantLattice>(value);
    if (!lattice || lattice->getValue().isUninitialized()) {
      return std::nullopt;
    }
    auto integerAttr =
        dyn_cast_or_null<IntegerAttr>(lattice->getValue().getConstantValue());
    return integerAttr ? std::optional(integerAttr.getValue()) : std::nullopt;
  }

  SmallVector<Attribute>
  evaluateOperands(Operation *operation,
                   const LoopInductionBindings &inductionValues) const {
    SmallVector<Attribute> operands;
    operands.reserve(operation->getNumOperands());
    IntegerExpressionEvaluator integerEvaluator =
        createIntegerEvaluator(inductionValues);
    for (Value operand : operation->getOperands()) {
      std::optional<llvm::APInt> maybeInteger =
          operand.getType().isIntOrIndex() ? integerEvaluator.evaluate(operand)
                                           : std::nullopt;
      if (maybeInteger) {
        operands.push_back(IntegerAttr::get(operand.getType(), *maybeInteger));
        continue;
      }
      Attribute constant;
      operands.push_back(matchPattern(operand, m_Constant(&constant))
                             ? constant
                             : Attribute());
    }
    return operands;
  }

  bool isKnownDead(Block *block) const {
    if (!dataFlowSolver) {
      return false;
    }
    ProgramPoint *blockStart = dataFlowSolver->getProgramPointBefore(block);
    const auto *executable =
        dataFlowSolver->lookupState<dataflow::Executable>(blockStart);
    return executable && !executable->isLive();
  }

  bool isKnownDead(Block *source, Block *target) const {
    if (!dataFlowSolver) {
      return false;
    }
    dataflow::CFGEdge *edge =
        dataFlowSolver->getLatticeAnchor<dataflow::CFGEdge>(source, target);
    const auto *executable =
        dataFlowSolver->lookupState<dataflow::Executable>(edge);
    return executable && !executable->isLive();
  }

  /// Return the target block's exact count from its possible control-flow CFG.
  std::optional<std::uint64_t>
  getExactBlockInvocationCount(Region &region, Block *targetBlock,
                               const LoopInductionBindings &inductionValues) {
    if (!targetBlock || targetBlock->getParent() != &region || region.empty()) {
      return std::nullopt;
    }

    FailureOr<BlockFlowKey> maybeFlowKey =
        getBlockFlowKey(region, inductionValues);
    if (failed(maybeFlowKey)) {
      return std::nullopt;
    }
    auto &regionCache = blockCountCache[&region];
    auto resultIt = regionCache.find(*maybeFlowKey);
    if (resultIt == regionCache.end()) {
      BlockCountResult result = classifyBlockCounts(region, *maybeFlowKey);
      auto [insertedIt, inserted] =
          regionCache.try_emplace(std::move(*maybeFlowKey), std::move(result));
      assert(inserted && "new block CFG must be inserted");
      resultIt = insertedIt;
    }

    BlockCountResult &result = resultIt->second;
    auto countIt = result.blockCounts.find(targetBlock);
    assert(countIt != result.blockCounts.end() &&
           "block-count result must contain every block");
    return countIt->second;
  }

  /// Encode the successor edges possible under the current induction values.
  FailureOr<BlockFlowKey>
  getBlockFlowKey(Region &region,
                  const LoopInductionBindings &inductionValues) const {
    BlockFlowKey flowKey;
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      if (!terminator) {
        return failure();
      }
      if (terminator->getNumSuccessors() == 0) {
        continue;
      }
      auto branch = dyn_cast<BranchOpInterface>(terminator);
      if (!branch) {
        return failure();
      }

      SmallVector<Attribute> operands =
          evaluateOperands(terminator, inductionValues);
      Block *selectedSuccessor = branch.getSuccessorForOperands(operands);
      if (selectedSuccessor &&
          !llvm::is_contained(terminator->getSuccessors(), selectedSuccessor)) {
        return failure();
      }

      for (Block *successor : terminator->getSuccessors()) {
        bool isPossible =
            !isKnownDead(&block) && !isKnownDead(successor) &&
            !isKnownDead(&block, successor) &&
            (!selectedSuccessor || successor == selectedSuccessor);
        flowKey.push_back(isPossible);
      }
    }
    return flowKey;
  }

  /// Classify every block in one invocation of the possible block CFG.
  BlockCountResult classifyBlockCounts(Region &region,
                                       ArrayRef<std::uint8_t> flowKey) const {
    using execution_count_detail::BlockFlowGraph;
    using execution_count_detail::BlockFlowNode;

    BlockCountResult result;
    BlockFlowGraph graph;
    // Successor lists store node pointers, so allocate every node first.
    graph.nodes.reserve(std::distance(region.begin(), region.end()));
    llvm::DenseMap<Block *, BlockFlowNode *> nodesByBlock;
    for (Block &block : region) {
      BlockFlowNode &node = graph.nodes.emplace_back();
      node.parent = &graph;
      node.block = &block;
      node.number = graph.nodes.size() - 1;
      nodesByBlock.try_emplace(&block, &node);
    }
    graph.entry = nodesByBlock.lookup(&region.front());

    std::size_t edgePosition = 0;
    for (Block &block : region) {
      BlockFlowNode *sourceNode = nodesByBlock.lookup(&block);
      Operation *terminator = block.getTerminator();
      for (Block *successor : terminator->getSuccessors()) {
        assert(edgePosition < flowKey.size() && "block-flow key is too short");
        if (!flowKey[edgePosition++]) {
          continue;
        }
        BlockFlowNode *successorNode = nodesByBlock.lookup(successor);
        assert(successorNode && "successor must belong to the same region");
        if (!llvm::is_contained(sourceNode->successors, successorNode)) {
          sourceNode->successors.push_back(successorNode);
          successorNode->predecessors.push_back(sourceNode);
        }
      }
    }
    assert(edgePosition == flowKey.size() && "block-flow key is too long");

    llvm::DenseSet<BlockFlowNode *> reachableNodes;
    SmallVector<BlockFlowNode *> worklist{graph.entry};
    reachableNodes.insert(graph.entry);
    while (!worklist.empty()) {
      BlockFlowNode *node = worklist.pop_back_val();
      for (BlockFlowNode *successor : node->successors) {
        if (reachableNodes.insert(successor).second) {
          worklist.push_back(successor);
        }
      }
    }

    // Blocks in a possible cycle may repeat, and its successors may never run.
    llvm::DenseSet<BlockFlowNode *> cycleAffectedNodes;
    for (auto sccIt = llvm::scc_begin(&graph); !sccIt.isAtEnd(); ++sccIt) {
      if (!sccIt.hasCycle()) {
        continue;
      }
      for (BlockFlowNode *node : *sccIt) {
        if (cycleAffectedNodes.insert(node).second) {
          worklist.push_back(node);
        }
      }
    }
    while (!worklist.empty()) {
      BlockFlowNode *node = worklist.pop_back_val();
      for (BlockFlowNode *successor : node->successors) {
        if (cycleAffectedNodes.insert(successor).second) {
          worklist.push_back(successor);
        }
      }
    }

    // In the remaining acyclic graph, a reachable block executes exactly once
    // iff every terminating or nonterminating continuation visits it.
    llvm::PostDomTreeBase<BlockFlowNode> postDominance;
    postDominance.recalculate(graph);
    for (BlockFlowNode &node : graph.nodes) {
      std::optional<std::uint64_t> maybeCount;
      if (!reachableNodes.contains(&node)) {
        maybeCount = 0;
      } else if (!cycleAffectedNodes.contains(&node) &&
                 postDominance.dominates(&node, graph.entry)) {
        maybeCount = 1;
      }
      result.blockCounts.try_emplace(node.block, maybeCount);
    }
    return result;
  }

  std::optional<std::uint64_t> getExactRegionInvocationCount(
      ControlFrame frame, const LoopInductionBindings &inductionValues) const {
    if (regionInvocationCountEvaluator) {
      if (std::optional<std::uint64_t> maybeCount =
              regionInvocationCountEvaluator(*frame.region)) {
        return maybeCount;
      }
    }

    auto branch = dyn_cast<RegionBranchOpInterface>(frame.parent);
    if (!branch) {
      return std::nullopt;
    }

    SmallVector<Attribute> operands =
        evaluateOperands(frame.parent, inductionValues);

    SmallVector<InvocationBounds> bounds;
    branch.getRegionInvocationBounds(operands, bounds);
    std::size_t regionNumber = frame.region->getRegionNumber();
    if (regionNumber >= bounds.size()) {
      return std::nullopt;
    }
    const InvocationBounds &regionBounds = bounds[regionNumber];
    auto maybeUpperBound = regionBounds.getUpperBound();
    SmallVector<RegionSuccessor> entrySuccessors;
    branch.getEntrySuccessorRegions(operands, entrySuccessors);
    if (entrySuccessors.size() == 1) {
      Region *selectedRegion = entrySuccessors.front().getSuccessor();
      if (!selectedRegion) {
        return 0;
      }
      if (selectedRegion != frame.region) {
        if (maybeUpperBound && *maybeUpperBound == 0) {
          return 0;
        }
        if (!isRegionReachable(branch, *selectedRegion, *frame.region)) {
          return 0;
        }
        return std::nullopt;
      }
      if (maybeUpperBound && *maybeUpperBound == 1) {
        return 1;
      }

      SmallVector<RegionSuccessor> regionSuccessors;
      branch.getSuccessorRegions(*selectedRegion, regionSuccessors);
      if (llvm::all_of(regionSuccessors, [](RegionSuccessor successor) {
            return successor.isOperation();
          })) {
        return 1;
      }
      return std::nullopt;
    }

    if (!maybeUpperBound || regionBounds.getLowerBound() != *maybeUpperBound) {
      return std::nullopt;
    }
    return *maybeUpperBound;
  }

  std::optional<std::uint64_t>
  countInsideFrame(ArrayRef<ControlFrame> frames, std::size_t frameIndex,
                   LoopInductionBindings &inductionValues,
                   EnumerationBudget &enumerationBudget) {
    ControlFrame frame = frames[frameIndex];
    std::optional<std::uint64_t> maybeBlockCount = getExactBlockInvocationCount(
        *frame.region, frame.targetBlock, inductionValues);
    if (!maybeBlockCount || *maybeBlockCount == 0) {
      return maybeBlockCount;
    }
    std::optional<std::uint64_t> maybeNestedCount = countExecutions(
        frames, frameIndex + 1, inductionValues, enumerationBudget);
    return maybeNestedCount
               ? llvm::checkedMulUnsigned(*maybeBlockCount, *maybeNestedCount)
               : std::nullopt;
  }

  std::optional<std::uint64_t>
  countExecutions(ArrayRef<ControlFrame> frames, std::size_t frameIndex,
                  LoopInductionBindings &inductionValues,
                  EnumerationBudget &enumerationBudget) {
    if (frameIndex == frames.size()) {
      return 1;
    }

    ControlFrame frame = frames[frameIndex];
    auto loop = dyn_cast<LoopLikeOpInterface>(frame.parent);
    SmallVector<Region *> loopRegions;
    if (loop) {
      loopRegions = loop.getLoopRegions();
    }
    if (!loop || !llvm::is_contained(loopRegions, frame.region)) {
      std::optional<std::uint64_t> maybeInvocationCount =
          getExactRegionInvocationCount(frame, inductionValues);
      if (!maybeInvocationCount) {
        return std::nullopt;
      }
      if (*maybeInvocationCount == 0) {
        return 0;
      }
      std::optional<std::uint64_t> maybeNestedCount = countInsideFrame(
          frames, frameIndex, inductionValues, enumerationBudget);
      return maybeNestedCount ? llvm::checkedMulUnsigned(*maybeInvocationCount,
                                                         *maybeNestedCount)
                              : std::nullopt;
    }

    std::optional<std::uint64_t> maybeTripCount =
        tt::getLoopTripCount(loop, inductionValues, [this](Value value) {
          return evaluateContextValue(value);
        });
    if (!maybeTripCount) {
      return std::nullopt;
    }

    // A trip count does not define the invocation count of each region in a
    // multi-region loop.
    if (loopRegions.size() != 1) {
      return std::nullopt;
    }
    if (*maybeTripCount == 0) {
      return 0;
    }

    // Try multiplication before enumeration. Passing the same budget charges
    // nested enumeration even when this attempt cannot prove a count.
    std::optional<std::uint64_t> maybeNestedCount = countInsideFrame(
        frames, frameIndex, inductionValues, enumerationBudget);
    if (maybeNestedCount) {
      return llvm::checkedMulUnsigned(*maybeTripCount, *maybeNestedCount);
    }

    if (!enumerationBudget.canConsume(*maybeTripCount)) {
      return std::nullopt;
    }
    std::uint64_t total = 0;
    SmallVector<LoopLikeOpInterface, 1> loops{loop};
    LogicalResult enumerationResult = enumerateLoopNest(
        loops, inductionValues, enumerationBudget,
        [&](const LoopInductionBindings &) {
          maybeNestedCount = countInsideFrame(
              frames, frameIndex, inductionValues, enumerationBudget);
          if (!maybeNestedCount) {
            return failure();
          }
          std::optional<std::uint64_t> maybeNextTotal =
              llvm::checkedAddUnsigned(total, *maybeNestedCount);
          if (!maybeNextTotal) {
            return failure();
          }
          total = *maybeNextTotal;
          return success();
        },
        [this](Value value) { return evaluateContextValue(value); });
    if (failed(enumerationResult)) {
      return std::nullopt;
    }
    return total;
  }

  Region &rootRegion;
  SymbolValueEvaluator symbolValueEvaluator;
  RegionInvocationCountEvaluator regionInvocationCountEvaluator;
  Options options;
  std::unique_ptr<DataFlowSolver> dataFlowSolver;
  llvm::DenseMap<Block *, std::optional<std::uint64_t>>
      blockExecutionCountCache;
  /// Reuse block counts when different induction values select the same edges.
  llvm::DenseMap<Region *, llvm::DenseMap<BlockFlowKey, BlockCountResult>>
      blockCountCache;
};

ExecutionCountAnalysis::ExecutionCountAnalysis(
    Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator,
    RegionInvocationCountEvaluator regionInvocationCountEvaluator)
    : ExecutionCountAnalysis(rootRegion, std::move(symbolValueEvaluator),
                             std::move(regionInvocationCountEvaluator),
                             Options{}) {}

ExecutionCountAnalysis::ExecutionCountAnalysis(
    Region &rootRegion, SymbolValueEvaluator symbolValueEvaluator,
    RegionInvocationCountEvaluator regionInvocationCountEvaluator,
    Options options)
    : impl(std::make_unique<Impl>(rootRegion, std::move(symbolValueEvaluator),
                                  std::move(regionInvocationCountEvaluator),
                                  options)) {}

ExecutionCountAnalysis::~ExecutionCountAnalysis() = default;

ExecutionCountAnalysis::ExecutionCountAnalysis(
    ExecutionCountAnalysis &&) noexcept = default;

ExecutionCountAnalysis &
ExecutionCountAnalysis::operator=(ExecutionCountAnalysis &&) noexcept = default;

std::optional<std::uint64_t>
ExecutionCountAnalysis::getExecutionCount(Operation *operation) {
  return impl->getExecutionCount(operation);
}

} // namespace mlir::tt
