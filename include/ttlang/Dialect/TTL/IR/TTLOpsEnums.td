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

def TTL_LogicalKernelKind_Compute :
    I32EnumAttrCase<"Compute", 0, "compute">;
def TTL_LogicalKernelKind_DataMovement :
    I32EnumAttrCase<"DataMovement", 1, "data_movement">;

def TTL_LogicalKernelKind : I32EnumAttr<
    "LogicalKernelKind", "Target-independent logical kernel kind",
    [TTL_LogicalKernelKind_Compute, TTL_LogicalKernelKind_DataMovement]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

def TTL_TransferKind : I32EnumAttr<
    "TransferKind", "TTL transfer kind",
    [
      I32EnumAttrCase<"read", 0>,
      I32EnumAttrCase<"write", 1>,
    ]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

def TTL_PipeTransferKind_PointToPoint :
    I32EnumAttrCase<"PointToPoint", 0, "point_to_point">;
def TTL_PipeTransferKind_Collective :
    I32EnumAttrCase<"Collective", 1, "collective">;

def TTL_PipeTransferKind : I32EnumAttr<
    "PipeTransferKind", "TTL pipe transfer kind",
    [TTL_PipeTransferKind_PointToPoint, TTL_PipeTransferKind_Collective]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

def TTL_ExternalTemplateArgKind_SignedInteger :
    I32EnumAttrCase<"SignedInteger", 0, "signed_integer">;
def TTL_ExternalTemplateArgKind_Boolean :
    I32EnumAttrCase<"Boolean", 1, "boolean">;
def TTL_ExternalTemplateArgKind_UnsignedInteger :
    I32EnumAttrCase<"UnsignedInteger", 2, "unsigned_integer">;
def TTL_ExternalTemplateArgKind_DFBIndex :
    I32EnumAttrCase<"DFBIndex", 3, "dfb_index">;
def TTL_ExternalTemplateArgKind_DFBDescriptor :
    I32EnumAttrCase<"DFBDescriptor", 4, "dfb_descriptor">;

def TTL_ExternalTemplateArgKind : I32EnumAttr<
    "ExternalTemplateArgKind", "External function template argument kind",
    [TTL_ExternalTemplateArgKind_SignedInteger,
     TTL_ExternalTemplateArgKind_Boolean,
     TTL_ExternalTemplateArgKind_UnsignedInteger,
     TTL_ExternalTemplateArgKind_DFBIndex,
     TTL_ExternalTemplateArgKind_DFBDescriptor]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

def TTL_DFBProtocolEffectKind_Reserve :
    I32EnumAttrCase<"Reserve", 0, "reserve">;
def TTL_DFBProtocolEffectKind_Push :
    I32EnumAttrCase<"Push", 1, "push">;
def TTL_DFBProtocolEffectKind_Wait :
    I32EnumAttrCase<"Wait", 2, "wait">;
def TTL_DFBProtocolEffectKind_Pop :
    I32EnumAttrCase<"Pop", 3, "pop">;

def TTL_DFBProtocolEffectKind : I32EnumAttr<
    "DFBProtocolEffectKind", "Dataflow buffer protocol effect kind",
    [TTL_DFBProtocolEffectKind_Reserve, TTL_DFBProtocolEffectKind_Push,
     TTL_DFBProtocolEffectKind_Wait, TTL_DFBProtocolEffectKind_Pop]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

def TTL_DFBNonTransactionalAccessKind_Inspect :
    I32EnumAttrCase<"Inspect", 0, "inspect">;

def TTL_DFBNonTransactionalAccessKind : I32EnumAttr<
    "DFBNonTransactionalAccessKind",
    "Dataflow buffer non-transactional access kind",
    [TTL_DFBNonTransactionalAccessKind_Inspect]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

def TTL_DFBTileStoreKind_Producer :
    I32EnumAttrCase<"Producer", 0, "producer">;
def TTL_DFBTileStoreKind_ConsumerReplacement :
    I32EnumAttrCase<"ConsumerReplacement", 1, "consumer_replacement">;

def TTL_DFBTileStoreKind : I32EnumAttr<
    "DFBTileStoreKind", "Dataflow buffer tile-store ownership",
    [TTL_DFBTileStoreKind_Producer,
     TTL_DFBTileStoreKind_ConsumerReplacement]> {
  let cppNamespace = "::mlir::tt::ttl";
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

def TTL_AccumulationInitialMode_Overwrite
    : I32EnumAttrCase<"Overwrite", 0, "overwrite">;
def TTL_AccumulationInitialMode_AccumulateExisting
    : I32EnumAttrCase<"AccumulateExisting", 1, "accumulate_existing">;
def TTL_AccumulationInitialMode_Init : I32EnumAttrCase<"Init", 2, "init">;

def TTL_AccumulationInitialMode : I32EnumAttr<"AccumulationInitialMode",
    "TTL accumulation initial-value mode",
    [TTL_AccumulationInitialMode_Overwrite,
     TTL_AccumulationInitialMode_AccumulateExisting,
     TTL_AccumulationInitialMode_Init]> {
        let cppNamespace = "::mlir::tt::ttl";
}

//===----------------------------------------------------------------------===//
// Exp input clamping
//===----------------------------------------------------------------------===//

// Controls whether the (approximate) exp clamps very negative inputs before
// the SFPU evaluation. Underlying values match tt-metal's ckernel::InputClamping
// and the TTKernel InputClamping enum so the lowering can forward by value:
//   None (0): no clamp -- faster, but inputs below ~-88.5 produce incorrect
//             outputs (guaranteed negative; consider packer ReLU).
//   ClampToNegative (1): clamp very negative inputs -- safer, slightly slower.
def TTL_InputClamping_None            : I32EnumAttrCase<"None",            0, "none">;
def TTL_InputClamping_ClampToNegative : I32EnumAttrCase<"ClampToNegative", 1, "clamp_to_negative">;

def TTL_InputClamping : I32EnumAttr<"InputClamping", "TTL exp input clamping mode",
    [TTL_InputClamping_None, TTL_InputClamping_ClampToNegative]> {
  let cppNamespace = "::mlir::tt::ttl";
}

def TTL_AccumulationCombiner_Add : I32EnumAttrCase<"Add", 0, "add">;
// TODO(#646): Add non-additive combiners once their DST-resident legality and
// lowering behavior are specified.

def TTL_AccumulationCombiner : I32EnumAttr<"AccumulationCombiner",
    "TTL accumulation combiner", [TTL_AccumulationCombiner_Add]> {
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

def TTL_TileExecutionStrategy_FPU :
    I32EnumAttrCase<"FPU", 0, "fpu">;
def TTL_TileExecutionStrategy_SFPU :
    I32EnumAttrCase<"SFPU", 1, "sfpu">;

def TTL_TileExecutionStrategy : I32EnumAttr<
    "TileExecutionStrategy", "tile execution strategy",
    [TTL_TileExecutionStrategy_FPU, TTL_TileExecutionStrategy_SFPU]> {
  let cppNamespace = "::mlir::tt::ttl";
  let genSpecializedAttr = 0;
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSENUMS_TD
