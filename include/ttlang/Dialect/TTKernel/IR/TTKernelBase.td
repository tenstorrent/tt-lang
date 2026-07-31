// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTKERNEL_TTKERNELDIALECT_TD
#define TTLANG_DIALECT_TTKERNEL_TTKERNELDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTKernel dialect definition.
//===----------------------------------------------------------------------===//

def TTKernel_Dialect : Dialect {
    let name = "ttkernel";
    let summary = "Tenstorrent kernel-level operations.";
    let description = [{
        This dialect models Tenstorrent kernel-level operations and types used
        by tt-lang lowering and C++ emission.
    }];
    let cppNamespace = "::mlir::tt::ttkernel";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::memref::MemRefDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::ttcore::TTCoreDialect"
    ];
}

//===----------------------------------------------------------------------===//
// TTKernel Traits.
//===----------------------------------------------------------------------===//

class TTKernel_Trait<string name, list<Trait> traits = []> : NativeOpTrait<name, traits> {
  let cppNamespace = "::mlir::tt::ttkernel";
}

def TTKernel_FPUOpTrait : TTKernel_Trait<"TTKernelFPUOpTrait", []>;

def TTKernel_SFPUOpTrait : TTKernel_Trait<"TTKernelSFPUOpTrait", []>;

def TTKernel_InitOpTrait : TTKernel_Trait<"TTKernelInitOpTrait", []>;

def TTKernel_UnaryOpTrait : TTKernel_Trait<"TTKernelUnaryOpTrait", []>;

def TTKernel_BinaryOpTrait : TTKernel_Trait<"TTKernelBinaryOpTrait", []>;

def TTKernel_TernaryOpTrait : TTKernel_Trait<"TTKernelTernaryOpTrait", []>;

def TTKernel_DeviceZoneOpTrait : TTKernel_Trait<"TTKernelDeviceZoneOpTrait", []>;

def TTKernel_TridNocOpTrait : TTKernel_Trait<"TTKernelTridNocOpTrait", []>;

def TTKernel_NocOpTrait : TTKernel_Trait<"TTKernelNocOpTrait", []>;

def TTKernel_NocReadCommandOpTrait
    : TTKernel_Trait<"TTKernelNocReadCommandOpTrait", []>;

def TTKernel_NocWriteCommandOpTrait
    : TTKernel_Trait<"TTKernelNocWriteCommandOpTrait", []>;

def TTKernel_NocAtomicCommandOpTrait
    : TTKernel_Trait<"TTKernelNocAtomicCommandOpTrait", []>;

def TTKernel_NocCommandStatePreservingOpTrait
    : TTKernel_Trait<"TTKernelNocCommandStatePreservingOpTrait", []>;

def TTKernel_NocCommandStateDependentOpTrait
    : TTKernel_Trait<"TTKernelNocCommandStateDependentOpTrait",
                     [TTKernel_NocOpTrait]>;

def TTKernel_NocCommandStateUnknownOpTrait
    : TTKernel_Trait<"TTKernelNocCommandStateUnknownOpTrait", []>;

def TTKernel_LayoutOpTrait : TTKernel_Trait<"TTKernelLayoutOpTrait", []>;

//===----------------------------------------------------------------------===//
// Base TTKernel operation definitions.
//===----------------------------------------------------------------------===//

class TTKernel_Op<string mnemonic, list<Trait> traits = []> :
        Op<TTKernel_Dialect, mnemonic, traits>;

class TTKernel_FPUOp<string mnemonic, list<Trait> traits = []> :
        TTKernel_Op<mnemonic, [TTKernel_FPUOpTrait] # traits>;

class TTKernel_SFPUOp<string mnemonic, list<Trait> traits = []> :
        TTKernel_Op<mnemonic, [TTKernel_SFPUOpTrait] # traits>;

class TTKernel_InitOp<string mnemonic, list<Trait> traits = []> :
        TTKernel_Op<mnemonic, [TTKernel_InitOpTrait] # traits>;

class TTKernel_NocOp<string mnemonic, list<Trait> traits = []> :
        TTKernel_Op<mnemonic, [TTKernel_NocOpTrait] # traits>;

class TTKernel_NocReadCommandOp<string mnemonic, list<Trait> traits = []> :
        TTKernel_NocOp<mnemonic, [TTKernel_NocReadCommandOpTrait] # traits>;

class TTKernel_NocWriteCommandOp<string mnemonic, list<Trait> traits = []> :
        TTKernel_NocOp<mnemonic, [TTKernel_NocWriteCommandOpTrait] # traits>;

class TTKernel_NocAtomicCommandOp<string mnemonic, list<Trait> traits = []> :
        TTKernel_NocOp<mnemonic, [TTKernel_NocAtomicCommandOpTrait] # traits>;

#endif
