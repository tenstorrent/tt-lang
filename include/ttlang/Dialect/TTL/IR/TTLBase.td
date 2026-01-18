// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLBASE_TD
#define TTLANG_DIALECT_TTL_IR_TTLBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// TTL dialect definition
//===----------------------------------------------------------------------===//

def TTL_Dialect : Dialect {
  let name = "ttl";
  let summary = "TT-Lang tensor-level IR";
  let description = [{
    Tensor-level operations for the TT-Lang DSL with SSA semantics for
    circular buffers, pipes, and synchronization.
  }];
  let cppNamespace = "::mlir::tt::ttl";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];

  let dependentDialects = [
    "::mlir::tt::ttcore::TTCoreDialect"
  ];
}

//===----------------------------------------------------------------------===//
// TTL base type and operation definitions
//===----------------------------------------------------------------------===//

class TTL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class TTL_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TTL_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class TTL_Op<string mnemonic, list<Trait> traits = []>
    : Op<TTL_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TTL traits
//===----------------------------------------------------------------------===//

class TTL_Trait<string name, list<Trait> traits = []>
    : NativeOpTrait<name, traits> {
  let cppNamespace = "::mlir::tt::ttl";
}

def TTL_ComputeOpTrait : TTL_Trait<"TTLComputeOpTrait", []>;
def TTL_DataMovementOpTrait : TTL_Trait<"TTLDataMovementOpTrait", []>;
def TTL_SynchronizationOpTrait : TTL_Trait<"TTLSynchronizationOpTrait", []>;
def TTL_CircularBufferOpTrait : TTL_Trait<"TTLCircularBufferOpTrait", []>;
def TTL_TileComputeOpTrait : TTL_Trait<"TTLTileComputeOpTrait", []>;
def TTL_UnaryElementwiseOpTrait : TTL_Trait<"TTLUnaryElementwiseOpTrait", []>;
def TTL_BinaryElementwiseOpTrait : TTL_Trait<"TTLBinaryElementwiseOpTrait", []>;
def TTL_TileUnaryOpTrait : TTL_Trait<"TTLTileUnaryOpTrait", []>;
def TTL_TileBinaryOpTrait : TTL_Trait<"TTLTileBinaryOpTrait", []>;

#endif // TTLANG_DIALECT_TTL_IR_TTLBASE_TD
