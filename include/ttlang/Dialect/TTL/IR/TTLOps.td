// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTL operation definitions
//===----------------------------------------------------------------------===//

def TTL_CreateCBOp : TTL_Op<"create_cb", [Pure]> {
  let summary = "Create a circular buffer value";
  let arguments = (ins
    I64ArrayAttr:$shape,
    TypeAttr:$element_type,
    I64Attr:$buffer_factor,
    OptionalAttr<I32Attr>:$buffer_index,
    OptionalAttr<I64Attr>:$page_size
  );
  let results = (outs TTL_CircularBuffer:$result);
  let assemblyFormat = "`(` `)` attr-dict `:` type($result)";
  let hasVerifier = 0;
}

def TTL_CopyOp : TTL_Op<"copy", [Pure]> {
  let summary = "Asynchronous copy between tensor and circular buffer";
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst
  );
  let results = (outs TTL_TransferHandle:$xf);
  let assemblyFormat = "$src `,` $dst attr-dict `:` functional-type(operands, results)";
}

def TTL_WaitOp : TTL_Op<"wait", [Pure]> {
  let summary = "Wait on a transfer handle";
  let arguments = (ins TTL_TransferHandle:$xf);
  let assemblyFormat = "$xf attr-dict";
}

def TTL_ReturnOp : TTL_Op<"return", [Pure, HasParent<"DmKernelOp">,
                                     Terminator, ReturnLike]> {
  let summary = "Return from a TTL kernel";
  let description = [{
    Terminates a TTL kernel function. Currently only supports void returns.
  }];
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def TTL_DmKernelOp : TTL_Op<"kernel.dm",
    [AutomaticAllocationScope, FunctionOpInterface, IsolatedFromAbove,
     Symbol, CallableOpInterface, OpAsmOpInterface]> {
  let summary = "Data movement kernel entry";
  let description = [{
    Defines a data-movement kernel. Arguments must be TTNN tensors or supported
    scalar element types. Verifier enforces operand types.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // FunctionOpInterface methods.
    ::mlir::Region *getCallableRegion() {
      return this->isExternal() ? nullptr : &getBody();
    }
    ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ::mlir::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
    unsigned getNumResults() { return getResultTypes().size(); }
  }];
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_TD
