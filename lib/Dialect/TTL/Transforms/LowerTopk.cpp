// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Lower TopK
//===----------------------------------------------------------------------===//
//
// Lowers `ttl.topk` to the single-core local bitonic sequence. `stable`
// selects fused keys: fuse runs in the section that first writes those keys,
// merge and rebuild copy them, and defuse runs in the section that splits
// them back into values and indices. The packed form is the dataflow buffer's
// representation, recorded as `ttl.topk_payload`, not a property of one
// destination-register section.
//
// Tile loops are unrolled. Only the row loop remains. Two scratch buffers
// alternate so a stage packs into a buffer other than the one it reads.
// Tiles a stage does not update are copied across. The final extract
// transposes the column-layout result back to row layout, matching the
// metal final kernel rather than the intermediate local kernel.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBMaterialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <optional>

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLLOWERTOPK
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct TileUpdate {
  int64_t left = 0;
  int64_t right = 0;
  bool skipSecond = false;
};

struct Bank {
  Value handle;
  RankedTensorType type;
};

class TopkLowering {
public:
  TopkLowering(TopkOp op, func::FuncOp kernel)
      : op(op), kernel(kernel), builder(op), loc(op.getLoc()),
        context(op.getContext()) {}

  LogicalResult lower();

private:
  Value indexConst(int64_t value);
  Value i32Const(int64_t value);
  Value placeholder(Type type);
  Value extract(Value tensor, Value row, int64_t column);
  Value transposeTile(Value inputTile, Value outputTile, Value dst);
  Value copyTile(Value tile, Type tileType, int64_t column, Value dst);
  void storeTile(Value tile, Value view, Value row, int64_t column, Value dst);
  void emitSection(llvm::function_ref<void()> emit);
  Bank allocate(RankedTensorType type, TopkPayload payload);
  Value reserve(const Bank &bank);
  Value waitAndAttach(const Bank &bank);
  void push(const Bank &bank);
  void pop(const Bank &bank);

  void emitInitialSort(Value row, const Bank &valuesOut,
                       const std::optional<Bank> &indicesOut);
  void emitPhase(ArrayRef<TileUpdate> updates, const Bank &src, const Bank &dst,
                 const std::optional<Bank> &indexSrc,
                 const std::optional<Bank> &indexDst, bool rebuild,
                 int64_t iteration, bool &ascending);
  void emitTransposeExtract(Value row, const Bank &src, Value output,
                            bool moveUint16);
  void emitFusedExtract(Value row, const Bank &keys, const Bank &stagingValues,
                        const Bank &stagingIndices, Value valuesOut,
                        Value indicesOut);

  TopkOp op;
  func::FuncOp kernel;
  OpBuilder builder;
  Location loc;
  MLIRContext *context;
  TopkOrder order = TopkOrder::Descending;
  bool largest = true;
  bool fused = false;
  bool switchDir = false;
  int64_t k = 0;
  int64_t width = 0;
  int64_t outputWidth = 0;
  int64_t logk = 0;
  int64_t endPhase = 0;
  Value values;
  Value indices;
  Value dst0;
  Value dst1;
  Value dst2;
  Value dst3;
  Value startPhase;
  Value endPhaseValue;
  Value kValue;
  Value logkValue;
  Type valueTile;
  Type indexTile;
};

Value TopkLowering::indexConst(int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

Value TopkLowering::i32Const(int64_t value) {
  return arith::ConstantOp::create(builder, loc,
                                   builder.getI32IntegerAttr(value))
      .getResult();
}

Value TopkLowering::placeholder(Type type) {
  return UnrealizedConversionCastOp::create(builder, loc, type, ValueRange{})
      .getResult(0);
}

Value TopkLowering::extract(Value tensor, Value row, int64_t column) {
  return tensor::ExtractOp::create(builder, loc, tensor,
                                   ValueRange{row, indexConst(column)})
      .getResult();
}

Value TopkLowering::transposeTile(Value inputTile, Value outputTile,
                                  Value dst) {
  // The result describes the DST content, which keeps the input element type.
  // `output` names the buffer the section packs into; in fused mode that is
  // the u32 key buffer, and annotation reads it to build transpose_wh_init.
  return TileTransposeOp::create(builder, loc, inputTile.getType(), inputTile,
                                 outputTile, dst)
      .getResult();
}

Value TopkLowering::copyTile(Value tile, Type tileType, int64_t column,
                             Value dst) {
  // src_indices are the CB coordinates. The extract that produced `tile`
  // is not consulted when the copy is linearized.
  return CopyTileOp::create(
             builder, loc, TypeRange{DSTRegisterType::get(context), tileType},
             tile, ValueRange{indexConst(0), indexConst(column)}, dst)
      .getDstTile();
}

void TopkLowering::storeTile(Value tile, Value view, Value row, int64_t column,
                             Value dst) {
  TileStoreOp::create(builder, loc, tile, view,
                      ValueRange{row, indexConst(column)}, dst,
                      DFBTileStoreKind::Producer,
                      /*row_prefix=*/UnitAttr());
}

void TopkLowering::emitSection(llvm::function_ref<void()> emit) {
  auto section = DstSectionOp::create(builder, loc);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(section.getBody().front().getTerminator());
    emit();
  }
  builder.setInsertionPointAfter(section);
}

Bank TopkLowering::allocate(RankedTensorType type, TopkPayload payload) {
  OpBuilder::InsertionGuard guard(builder);
  BindCBOp bind = createCompilerAllocatedDFB(type, loc, kernel, builder);
  if (payload != TopkPayload::Plain) {
    bind->setAttr(kTopkPayloadAttrName, TopkPayloadAttr::get(context, payload));
    bind->setAttr(kTopkOrderAttrName, TopkOrderAttr::get(context, order));
  }
  return Bank{bind.getResult(), type};
}

Value TopkLowering::reserve(const Bank &bank) {
  auto reserved =
      CBReserveOp::create(builder, loc, bank.type, bank.handle, IntegerAttr());
  builder.setInsertionPointAfter(reserved);
  return reserved.getResult();
}

Value TopkLowering::waitAndAttach(const Bank &bank) {
  AttachCBOp attached =
      createDFBWaitAndAttach(bank.handle, bank.type, loc, builder);
  builder.setInsertionPointAfter(attached);
  return attached.getResult();
}

void TopkLowering::push(const Bank &bank) {
  auto pushed = CBPushOp::create(builder, loc, bank.handle, IntegerAttr());
  builder.setInsertionPointAfter(pushed);
}

void TopkLowering::pop(const Bank &bank) {
  auto popped = CBPopOp::create(builder, loc, bank.handle, IntegerAttr());
  builder.setInsertionPointAfter(popped);
}

void TopkLowering::emitInitialSort(Value row, const Bank &valuesOut,
                                   const std::optional<Bank> &indicesOut) {
  Value valuesView = reserve(valuesOut);
  Value indicesView;
  if (indicesOut) {
    indicesView = reserve(*indicesOut);
  }
  bool ascending = !largest;
  for (int64_t column = 0; column < width; column += 2) {
    int64_t direction = ascending ? 1 : 0;
    emitSection([&] {
      Value left =
          transposeTile(extract(values, row, column),
                        extract(valuesView, indexConst(0), column), dst0);
      Value right =
          transposeTile(extract(values, row, column + 1),
                        extract(valuesView, indexConst(0), column + 1), dst1);
      Value indexLeft = extract(indices, row, column);
      Value indexRight = extract(indices, row, column + 1);
      Value indexLeftOut =
          indicesOut ? extract(indicesView, indexConst(0), column) : indexLeft;
      Value indexRightOut =
          indicesOut ? extract(indicesView, indexConst(0), column + 1)
                     : indexRight;
      transposeTile(indexLeft, indexLeftOut, dst2);
      transposeTile(indexRight, indexRightOut, dst3);
      if (fused) {
        TileTopkFuseOp::create(builder, loc, dst0, order);
      }
      TileTopkLocalSortOp::create(
          builder, loc, dst0, i32Const(direction), endPhaseValue, startPhase,
          /*end_step=*/Value(), /*start_step=*/Value(), order,
          /*fp32_dest_acc_en=*/BoolAttr(), /*stable_sort=*/false,
          /*fused=*/fused, /*rank_stamped=*/false, /*tag_bits=*/16);
      Type packedType = fused ? valuesOut.type.getElementType() : valueTile;
      storeTile(fused ? placeholder(packedType) : left, valuesView,
                indexConst(0), column, dst0);
      storeTile(fused ? placeholder(packedType) : right, valuesView,
                indexConst(0), column + 1, dst1);
      if (!fused) {
        storeTile(placeholder(indexTile), indicesView, indexConst(0), column,
                  dst2);
        storeTile(placeholder(indexTile), indicesView, indexConst(0),
                  column + 1, dst3);
      }
    });
    if (switchDir) {
      ascending = !ascending;
    }
  }
  push(valuesOut);
  if (indicesOut) {
    push(*indicesOut);
  }
}

void TopkLowering::emitPhase(ArrayRef<TileUpdate> updates, const Bank &src,
                             const Bank &dst,
                             const std::optional<Bank> &indexSrc,
                             const std::optional<Bank> &indexDst, bool rebuild,
                             int64_t iteration, bool &ascending) {
  Value srcTensor = waitAndAttach(src);
  Value indexSrcTensor;
  if (indexSrc) {
    indexSrcTensor = waitAndAttach(*indexSrc);
  }
  Value dstView = reserve(dst);
  Value indexDstView;
  if (indexDst) {
    indexDstView = reserve(*indexDst);
  }

  SmallVector<bool> written(width, false);
  Type tileType = src.type.getElementType();
  for (TileUpdate update : updates) {
    written[update.left] = true;
    if (!update.skipSecond) {
      written[update.right] = true;
    }
    int64_t direction = ascending ? 1 : 0;
    emitSection([&] {
      Value left = copyTile(extract(srcTensor, indexConst(0), update.left),
                            tileType, update.left, dst0);
      Value right;
      if (!update.skipSecond) {
        right = copyTile(extract(srcTensor, indexConst(0), update.right),
                         tileType, update.right, dst1);
      }
      if (!fused) {
        copyTile(extract(indexSrcTensor, indexConst(0), update.left), indexTile,
                 update.left, dst2);
        if (!update.skipSecond) {
          copyTile(extract(indexSrcTensor, indexConst(0), update.right),
                   indexTile, update.right, dst3);
        }
      }
      if (rebuild) {
        TileTopkRebuildOp::create(
            builder, loc, dst0, i32Const(direction), i32Const(iteration),
            kValue, logkValue, i32Const(update.skipSecond ? 1 : 0), order,
            /*fp32_dest_acc_en=*/BoolAttr(), /*stable_sort=*/false,
            /*fused=*/fused, /*rank_stamped=*/false, /*tag_bits=*/16);
      } else {
        TileTopkMergeOp::create(
            builder, loc, dst0, i32Const(iteration), kValue, order,
            /*fp32_dest_acc_en=*/BoolAttr(), /*stable_sort=*/false,
            /*fused=*/fused, /*rank_stamped=*/false, /*tag_bits=*/16,
            /*direction=*/!largest);
      }
      storeTile(left, dstView, indexConst(0), update.left, dst0);
      if (!update.skipSecond) {
        storeTile(right, dstView, indexConst(0), update.right, dst1);
      }
      if (!fused) {
        storeTile(placeholder(indexTile), indexDstView, indexConst(0),
                  update.left, dst2);
        if (!update.skipSecond) {
          storeTile(placeholder(indexTile), indexDstView, indexConst(0),
                    update.right, dst3);
        }
      }
    });
    if (rebuild && switchDir) {
      ascending = !ascending;
    }
  }

  for (int64_t column = 0; column < width; ++column) {
    if (written[column]) {
      continue;
    }
    emitSection([&] {
      Value tile = copyTile(extract(srcTensor, indexConst(0), column), tileType,
                            column, dst0);
      storeTile(tile, dstView, indexConst(0), column, dst0);
      if (!fused) {
        Value indexTileValue =
            copyTile(extract(indexSrcTensor, indexConst(0), column), indexTile,
                     column, dst2);
        storeTile(indexTileValue, indexDstView, indexConst(0), column, dst2);
      }
    });
  }

  pop(src);
  push(dst);
  if (indexSrc) {
    pop(*indexSrc);
    push(*indexDst);
  }
}

void TopkLowering::emitTransposeExtract(Value row, const Bank &src,
                                        Value output, bool moveUint16) {
  Value srcTensor = waitAndAttach(src);
  for (int64_t column = 0; column < outputWidth; ++column) {
    emitSection([&] {
      Value tile = transposeTile(extract(srcTensor, indexConst(0), column),
                                 extract(output, row, column), dst0);
      if (moveUint16) {
        TileTopkUint16MoveDestTileToPackHalfOp::create(builder, loc, dst0,
                                                       BoolAttr());
      }
      storeTile(tile, output, row, column, dst0);
    });
  }
  pop(src);
}

void TopkLowering::emitFusedExtract(Value row, const Bank &keys,
                                    const Bank &stagingValues,
                                    const Bank &stagingIndices, Value valuesOut,
                                    Value indicesOut) {
  Value keysTensor = waitAndAttach(keys);
  Value valuesView = reserve(stagingValues);
  Value indicesView = reserve(stagingIndices);
  Type keyTile = keys.type.getElementType();
  for (int64_t column = 0; column < outputWidth; ++column) {
    emitSection([&] {
      copyTile(extract(keysTensor, indexConst(0), column), keyTile, column,
               dst0);
      TileTopkDefuseOp::create(builder, loc, dst0, i32Const(1), order);
      storeTile(placeholder(valueTile), valuesView, indexConst(0), column,
                dst0);
      storeTile(placeholder(indexTile), indicesView, indexConst(0), column,
                dst2);
    });
  }
  push(stagingValues);
  push(stagingIndices);
  pop(keys);

  emitTransposeExtract(row, stagingValues, valuesOut, /*moveUint16=*/false);
  emitTransposeExtract(row, stagingIndices, indicesOut, /*moveUint16=*/true);
}

LogicalResult TopkLowering::lower() {
  values = op.getValues();
  indices = op.getIndices();
  if (!getAttachedCB(values)) {
    return op.emitOpError("values must be attached to a dataflow buffer");
  }
  if (!getAttachedCB(indices)) {
    return op.emitOpError(
        "index tensor is the identity indices the data-movement kernel "
        "publishes; the compiler does not generate that reader yet");
  }
  if (!op.getResultValues().hasOneUse() || !op.getResultIndices().hasOneUse()) {
    return op.emitOpError(
        "each result must be stored exactly once into a reserved dataflow "
        "buffer");
  }
  auto valuesStore = dyn_cast<StoreOp>(*op.getResultValues().user_begin());
  auto indicesStore = dyn_cast<StoreOp>(*op.getResultIndices().user_begin());
  auto valuesReserve =
      valuesStore ? findCBReserveForView(valuesStore.getView()) : CBReserveOp();
  auto indicesReserve = indicesStore
                            ? findCBReserveForView(indicesStore.getView())
                            : CBReserveOp();
  if (!valuesStore || valuesStore.getTensor() != op.getResultValues() ||
      !indicesStore || indicesStore.getTensor() != op.getResultIndices() ||
      !valuesReserve || !indicesReserve ||
      valuesStore.getView().getType() != op.getResultValues().getType() ||
      indicesStore.getView().getType() != op.getResultIndices().getType()) {
    return op.emitOpError(
        "each result must be stored into a reserved dataflow buffer of the "
        "result type");
  }
  if (valuesStore->getBlock() != indicesStore->getBlock()) {
    return op.emitOpError(
        "value and index results must be stored in the same block");
  }
  Operation *firstStore = valuesStore->isBeforeInBlock(indicesStore)
                              ? valuesStore.getOperation()
                              : indicesStore.getOperation();
  DominanceInfo dominance(kernel);
  if (!dominance.properlyDominates(valuesReserve.getOperation(), firstStore) ||
      !dominance.properlyDominates(indicesReserve.getOperation(), firstStore)) {
    return op.emitOpError(
        "result buffer reserves must dominate both result stores");
  }
  // A multiply of a result is materialized after this operation, so that
  // reserve does not dominate the operation. Emit at the first store, where
  // both reserves are available.
  builder.setInsertionPoint(firstStore);

  auto valuesType = cast<RankedTensorType>(values.getType());
  valueTile = valuesType.getElementType();
  indexTile = cast<RankedTensorType>(indices.getType()).getElementType();
  int64_t height = valuesType.getShape()[0];
  width = valuesType.getShape()[1];
  // The merge network selects whole tiles. A request below one tile runs the
  // 32-wide network; the sorted result tile holds the first k columns.
  k = std::max<int64_t>(op.getK(), 32);
  largest = op.getLargest();
  fused = op.getStable();
  order = largest ? TopkOrder::Descending : TopkOrder::Ascending;
  switchDir = k == 64;
  outputWidth = (k + 31) / 32;
  logk = llvm::Log2_64(static_cast<uint64_t>(k));
  endPhase = logk - 1;
  int64_t logWidth = llvm::Log2_64(static_cast<uint64_t>(width));
  int64_t tilesPerSeq = outputWidth;

  Type packedTile =
      fused
          ? ttcore::TileType::get(context,
                                  cast<ttcore::TileType>(valueTile).getShape(),
                                  ttcore::DataType::UInt32)
          : valueTile;
  auto scratchType = RankedTensorType::get({1, width}, packedTile);
  TopkPayload payload = fused ? TopkPayload::FusedKeys : TopkPayload::Plain;
  std::array<Bank, 2> valueBanks = {allocate(scratchType, payload),
                                    allocate(scratchType, payload)};
  std::optional<Bank> indexBanks[2];
  if (!fused) {
    auto indexScratch =
        RankedTensorType::get({1, width}, cast<Type>(indexTile));
    indexBanks[0] = allocate(indexScratch, TopkPayload::Plain);
    indexBanks[1] = allocate(indexScratch, TopkPayload::Plain);
  }

  dst0 = indexConst(0);
  builder.setInsertionPointAfter(dst0.getDefiningOp());
  dst1 = indexConst(1);
  builder.setInsertionPointAfter(dst1.getDefiningOp());
  dst2 = indexConst(2);
  builder.setInsertionPointAfter(dst2.getDefiningOp());
  dst3 = indexConst(3);
  builder.setInsertionPointAfter(dst3.getDefiningOp());
  startPhase = i32Const(0);
  builder.setInsertionPointAfter(startPhase.getDefiningOp());
  endPhaseValue = i32Const(endPhase);
  builder.setInsertionPointAfter(endPhaseValue.getDefiningOp());
  kValue = i32Const(k);
  builder.setInsertionPointAfter(kValue.getDefiningOp());
  logkValue = i32Const(logk);
  builder.setInsertionPointAfter(logkValue.getDefiningOp());

  Value zero = indexConst(0);
  builder.setInsertionPointAfter(zero.getDefiningOp());
  Value heightValue = indexConst(height);
  builder.setInsertionPointAfter(heightValue.getDefiningOp());
  Value step = indexConst(1);
  builder.setInsertionPointAfter(step.getDefiningOp());
  auto rowLoop = scf::ForOp::create(builder, loc, zero, heightValue, step);
  builder.setInsertionPointToStart(rowLoop.getBody());
  Value row = rowLoop.getInductionVar();

  emitInitialSort(row, valueBanks[0],
                  fused ? std::nullopt : std::optional<Bank>(indexBanks[0]));

  int64_t numSequences = (width * 32) / k;
  int64_t sequencesPerPair = std::max<int64_t>(64 / k, 2);
  int current = 0;
  for (int64_t iteration = 0; iteration < logWidth; ++iteration) {
    int next = 1 - current;
    SmallVector<TileUpdate> merges;
    int64_t distance = ((1LL << iteration) * k) >> 5;
    for (int64_t sequence = 0; sequence < numSequences;
         sequence += sequencesPerPair) {
      for (int64_t tile = 0; tile < tilesPerSeq; ++tile) {
        int64_t left = ((sequence * (1LL << iteration) * k) >> 5) + tile;
        int64_t right = left + distance;
        if (left == right) {
          right = left + 1;
        }
        if (left >= width || right >= width) {
          break;
        }
        merges.push_back(TileUpdate{left, right, false});
      }
    }
    bool unusedAscending = !largest;
    emitPhase(merges, valueBanks[current], valueBanks[next],
              fused ? std::nullopt : std::optional<Bank>(indexBanks[current]),
              fused ? std::nullopt : std::optional<Bank>(indexBanks[next]),
              /*rebuild=*/false, iteration, unusedAscending);
    current = next;

    numSequences >>= 1;
    int64_t targetTiles = (numSequences == 1 && tilesPerSeq == 1) ? 1 : 2;
    sequencesPerPair = sequencesPerPair == 2 ? 2 : sequencesPerPair >> 1;
    next = 1 - current;
    SmallVector<TileUpdate> rebuilds;
    int64_t selected[2] = {};
    int selectedCount = 0;
    for (int64_t sequence = 0; sequence < numSequences;
         sequence += (sequencesPerPair >> 1)) {
      for (int64_t tile = 0; tile < tilesPerSeq; ++tile) {
        int64_t left = ((sequence * (1LL << (iteration + 1)) * k) >> 5) + tile;
        if (left >= width) {
          break;
        }
        selected[selectedCount++] = left;
        if (selectedCount == targetTiles) {
          bool skipSecond = targetTiles == 1;
          rebuilds.push_back(TileUpdate{
              selected[0], skipSecond ? selected[0] : selected[1], skipSecond});
          selectedCount = 0;
        }
      }
    }
    bool ascending = !largest;
    emitPhase(rebuilds, valueBanks[current], valueBanks[next],
              fused ? std::nullopt : std::optional<Bank>(indexBanks[current]),
              fused ? std::nullopt : std::optional<Bank>(indexBanks[next]),
              /*rebuild=*/true, iteration, ascending);
    current = next;
  }

  std::optional<Bank> stagingValues;
  std::optional<Bank> stagingIndices;
  if (fused) {
    stagingValues = allocate(RankedTensorType::get({1, outputWidth}, valueTile),
                             TopkPayload::Plain);
    stagingIndices = allocate(
        RankedTensorType::get({1, outputWidth}, indexTile), TopkPayload::Plain);
  }

  if (fused) {
    emitFusedExtract(row, valueBanks[current], *stagingValues, *stagingIndices,
                     valuesStore.getView(), indicesStore.getView());
  } else {
    emitTransposeExtract(row, valueBanks[current], valuesStore.getView(),
                         /*moveUint16=*/false);
    emitTransposeExtract(row, *indexBanks[current], indicesStore.getView(),
                         /*moveUint16=*/false);
  }

  valuesStore.erase();
  indicesStore.erase();
  op.erase();
  return success();
}

struct TTLLowerTopkPass : impl::TTLLowerTopkBase<TTLLowerTopkPass> {
  void runOnOperation() override {
    func::FuncOp kernel = getOperation();
    SmallVector<TopkOp> ops;
    kernel.walk([&](TopkOp topk) { ops.push_back(topk); });
    for (TopkOp topk : ops) {
      if (failed(TopkLowering(topk, kernel).lower())) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
