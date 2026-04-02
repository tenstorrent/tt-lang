// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// TTL enum definitions
//===----------------------------------------------------------------------===//

// Note: TTL uses TTKernel's ThreadType enum (#ttkernel.thread<noc>) for
// kernel thread annotations on functions.

def TTL_TransferKind : I32EnumAttr<
    "TransferKind", "TTL transfer kind",
    [
      I32EnumAttrCase<"read", 0>,
      I32EnumAttrCase<"write", 1>,
    ]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

def TTL_BcastType_Col    : I32EnumAttrCase<"Col",    1, "col">;
def TTL_BcastType_Row    : I32EnumAttrCase<"Row",    2, "row">;
def TTL_BcastType_Scalar : I32EnumAttrCase<"Scalar", 3, "scalar">;

def TTL_BcastType : I32EnumAttr<"BcastType", "TTL broadcast type",
    [TTL_BcastType_Col, TTL_BcastType_Row, TTL_BcastType_Scalar]> {
  let cppNamespace = "::mlir::tt::ttl";
}

//===----------------------------------------------------------------------===//
// Reduce enums
//===----------------------------------------------------------------------===//

def TTL_ReduceType_Sum : I32EnumAttrCase<"Sum", 0, "sum">;
def TTL_ReduceType_Max : I32EnumAttrCase<"Max", 1, "max">;

def TTL_ReduceType : I32EnumAttr<"ReduceType", "TTL reduction function type",
    [TTL_ReduceType_Sum, TTL_ReduceType_Max]> {
  let cppNamespace = "::mlir::tt::ttl";
}

// Buffer type for tensor memory placement.
// Values match TTNN BufferType enum for compatibility.
def TTL_BufferType_DRAM       : I32EnumAttrCase<"DRAM",       0, "dram">;
def TTL_BufferType_L1         : I32EnumAttrCase<"L1",         1, "l1">;
def TTL_BufferType_SystemMemory : I32EnumAttrCase<"SystemMemory", 2, "system_memory">;
def TTL_BufferType_L1Small    : I32EnumAttrCase<"L1Small",    3, "l1_small">;

def TTL_BufferType : I32EnumAttr<"BufferType", "TTL buffer type",
    [TTL_BufferType_DRAM, TTL_BufferType_L1,
     TTL_BufferType_SystemMemory, TTL_BufferType_L1Small]> {
  let cppNamespace = "::mlir::tt::ttl";
}

// Memory layout for how tensors are distributed across cores.
// Values match TTNN TensorMemoryLayout enum for compatibility.
def TTL_TensorMemoryLayout_Interleaved  : I32EnumAttrCase<"Interleaved",  0, "interleaved">;
def TTL_TensorMemoryLayout_SingleBank   : I32EnumAttrCase<"SingleBank",   1, "single_bank">;
def TTL_TensorMemoryLayout_HeightSharded : I32EnumAttrCase<"HeightSharded", 2, "height_sharded">;
def TTL_TensorMemoryLayout_WidthSharded  : I32EnumAttrCase<"WidthSharded",  3, "width_sharded">;
def TTL_TensorMemoryLayout_BlockSharded  : I32EnumAttrCase<"BlockSharded",  4, "block_sharded">;

def TTL_TensorMemoryLayout : I32EnumAttr<
    "TensorMemoryLayout", "TTL tensor memory layout",
    [TTL_TensorMemoryLayout_Interleaved, TTL_TensorMemoryLayout_SingleBank,
     TTL_TensorMemoryLayout_HeightSharded, TTL_TensorMemoryLayout_WidthSharded,
     TTL_TensorMemoryLayout_BlockSharded]> {
  let cppNamespace = "::mlir::tt::ttl";
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD
