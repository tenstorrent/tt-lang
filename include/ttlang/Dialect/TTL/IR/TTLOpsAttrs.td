// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsEnums.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// TTL attribute definitions
//===----------------------------------------------------------------------===//

def TTL_LogicalKernelAttr : TTL_Attr<"LogicalKernel", "logical_kernel"> {
  let summary = "Target-independent identity of one operation kernel";
  let description = [{
    Records the logical kind and operation-local identity of a kernel function.
    A canonical kind selector omits its identity. An explicit operation handle
    records its identity and owning operation. Compiler-owned affinity kernels
    use `role` instead of an operation.
  }];
  let parameters = (ins
    "LogicalKernelKind":$kind,
    OptionalParameter<"StringAttr">:$identity,
    OptionalParameter<"StringAttr">:$operation,
    OptionalParameter<"StringAttr">:$role
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_DispatchConditionAttr
    : TTL_Attr<"DispatchCondition", "dispatch_condition"> {
  let summary = "Module-local dispatch-stable condition identity";
  let description = [{
    Identifies one immutable condition declaration within a module. Equal
    attributes reference the same condition and evaluate to the same truth
    value for one dispatch and launch coordinate. Distinct conditions within a
    module require distinct ordinals. Zero is false and nonzero is true. The
    value may differ between launch coordinates or dispatches.

    The ordinal is module-local and has no runtime meaning. The scalar type
    is the external result carrier and must be signless i32 or i64.
  }];
  let parameters = (ins
    "int64_t":$ordinal,
    "Type":$scalarType
  );
  let assemblyFormat = "`<` $ordinal `,` $scalarType `>`";
  let genVerifyDecl = 1;
}

def TTL_DFBAllocationGroupAttr
    : TTL_Attr<"DFBAllocationGroup", "dfb_allocation_group"> {
  let summary = "Module-local verified dataflow buffer allocation identity";
  let description = [{
    Identifies logical dataflow buffers that must receive one physical DFB
    index after the compiler proves compatible static configuration, storage,
    non-overlapping lifecycles, and reusable protocol state. Equal attributes
    identify one group. Distinct attributes impose no allocation relationship.

    `$ordinal` is module-local and has no runtime meaning. The attribute does
    not synchronize execution, reset DFB state, or by itself permit an
    interference edge to be removed.
  }];
  let parameters = (ins "int64_t":$ordinal);
  let assemblyFormat = "`<` $ordinal `>`";
  let genVerifyDecl = 1;
}

def TTL_SynchronizedDFBResetAttr
    : TTL_Attr<"SynchronizedDFBReset", "synchronized_dfb_reset"> {
  let summary = "Worker-local synchronized dataflow buffer reset identity";
  let description = [{
    Identifies one dynamic reset instance and the complete set of logical
    kernels that participate on each active worker. Every participant reaches
    the same instance before any participant returns. The reset establishes
    canonical read-pointer, write-pointer, occupancy, and initialization state
    for each target declared by the operation.

    The ordinal is module-local and has no runtime meaning. Equal attributes
    reference the same reset instance. Distinct ordinals identify independent
    reset instances. Participant order is canonical and otherwise semantically
    insignificant. The participant set contains one compute kernel and both
    data movement kernels that own DFB interface state. Each participant
    executes the declaration at most once per dispatch and launch node.
  }];
  let parameters = (ins
    "int64_t":$ordinal,
    ArrayRefParameter<"::mlir::tt::ttl::LogicalKernelAttr">:$participants
  );
  let assemblyFormat = "`<` $ordinal `,` `participants` `[` $participants `]` `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    static SynchronizedDFBResetAttr getCheckedInstance(
        ::mlir::Location location, ::mlir::MLIRContext *context,
        int64_t ordinal,
        ::llvm::ArrayRef<LogicalKernelAttr> participants);
  }];
}

def TTL_SliceAttr : TTL_Attr<"Slice", "slice"> {
  let summary = "Slice specification for core range";
  let parameters = (ins
    "int64_t":$start,
    "int64_t":$stop,
    "int64_t":$step
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
  let description = [{
    Encodes a Python slice(start, stop, step) for pipe receiver sets.
    Half-open interval: [start, stop).
  }];
}

def TTL_TensorBackingAttr
    : TTL_Attr<"TensorBacking", "tensor_backing"> {
  let summary = "Tensor storage used by a dataflow buffer";
  let description = [{
    Identifies a byte range in one operation tensor argument that supplies the
    storage for a dataflow buffer. The range is relative to each launch node's
    tensor shard. It must match the complete dataflow buffer capacity and must
    remain valid for the complete program execution.
  }];
  let parameters = (ins
    "int64_t":$tensorIndex,
    "int64_t":$byteOffset,
    "int64_t":$byteSize
  );
  let assemblyFormat = "`<` `tensor_index` `=` $tensorIndex `,`"
                       " `byte_offset` `=` $byteOffset `,`"
                       " `byte_size` `=` $byteSize `>`";
  let genVerifyDecl = 1;
}

def TTL_PipeTransferKindAttr : TTL_Attr<"PipeTransferKind",
                                        "pipe_transfer_kind"> {
  let summary = "Pipe transfer kind";
  let description = [{
    Encodes whether a pipe transfer is point-to-point or collective. The kind
    describes the logical transfer contract before hardware lowering chooses
    concrete synchronization and data movement resources.
  }];
  let parameters = (ins
    "PipeTransferKind":$value
  );
  let assemblyFormat = "`<` $value `>`";
}

def TTL_TileExecutionStrategyAttr
    : TTL_Attr<"TileExecutionStrategy", "tile_execution_strategy"> {
  let summary = "Selected tile execution strategy";
  let description = [{
    Identifies the execution strategy that defines operand routing and DST
    behavior for a tile operation with execution-strategy alternatives.
  }];
  let parameters = (ins "TileExecutionStrategy":$value);
  let assemblyFormat = "`<` $value `>`";
}

def TTL_ExternalTemplateArgAttr
    : TTL_Attr<"ExternalTemplateArg", "external_template_arg"> {
  let summary = "Static argument to an external C++ template";
  let description = [{
    Records one ordered external template argument without representing static
    metadata as an SSA value. For DFB index and descriptor arguments, `value`
    indexes the opaque call's `template_dfb_operands` segment. Other kinds
    store the complete 32-bit literal payload.
  }];
  let parameters = (ins
    "ExternalTemplateArgKind":$kind,
    "int64_t":$value
  );
  let assemblyFormat = "`<` $kind `,` $value `>`";
  let genVerifyDecl = 1;
}

def TTL_DFBProtocolEffectAttr
    : TTL_Attr<"DFBProtocolEffect", "dfb_protocol_effect"> {
  let summary = "One synchronous dataflow buffer protocol effect";
  let description = [{
    Records one action completed by an opaque external call before it returns.
    `dependency_index` identifies an element of the value sequence returned by
    the owning operation's `getDFBDependencyOperands()` interface method.
    `num_tiles` is the positive, statically known transaction size and cannot
    exceed the selected dependency's capacity. Array order is execution order,
    including repeated transactions.
  }];
  let parameters = (ins
    "DFBProtocolEffectKind":$kind,
    "int64_t":$dependencyIndex,
    "int64_t":$numTiles
  );
  let assemblyFormat = "`<` $kind `,` $dependencyIndex `,` $numTiles `>`";
  let genVerifyDecl = 1;
}

def TTL_DFBProtocolEffectArrayAttr
    : TypedArrayAttrBase<TTL_DFBProtocolEffectAttr,
                         "ordered dataflow buffer protocol effects">;

def TTL_DFBNonTransactionalAccessAttr
    : TTL_Attr<"DFBNonTransactionalAccess", "dfb_non_transactional_access"> {
  let summary = "One synchronous non-transactional dataflow buffer access";
  let description = [{
    Describes an external call's use of one ordered DFB dependency without a
    reserve, push, wait, or pop transaction. `dependency_index` selects the
    call's ordered DFB dependency list. `inspect` states that the callee may
    read the DFB descriptor or contents but does not publish, consume, or leave
    the DFB changed when it returns. The call must complete all associated work
    synchronously.
  }];
  let parameters = (ins
    "DFBNonTransactionalAccessKind":$kind,
    "int64_t":$dependencyIndex
  );
  let assemblyFormat = "`<` $kind `,` $dependencyIndex `>`";
  let genVerifyDecl = 1;
}

def TTL_DFBNonTransactionalAccessArrayAttr
    : TypedArrayAttrBase<TTL_DFBNonTransactionalAccessAttr,
                         "ordered non-transactional dataflow buffer accesses">;

def TTL_LayoutAttr : TTL_Attr<"Layout", "layout"> {
  let summary = "Tensor layout encoding for grid distribution and memory placement";
  let description = [{
    Encodes how a tensor is distributed across the device grid and placed in
    memory. Used as the encoding attribute on RankedTensorType.

    This replaces the dependency on TTNNLayoutAttr from the TTNN dialect,
    providing the minimal layout information needed by the tt-lang compiler:
    - Logical tensor shape
    - Tile element type
    - Buffer type (L1, DRAM, etc.)
    - Grid shape for distribution
    - Memory layout (interleaved, sharded, etc.)
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "BufferType":$bufferType,
    ArrayRefParameter<"int64_t">:$grid,
    "TensorMemoryLayout":$memoryLayout
  );

  let assemblyFormat = "`<` `shape` `=` `[` $shape `]` `,`"
                       " `element_type` `=` $elementType `,`"
                       " `buffer` `=` $bufferType `,`"
                       " `grid` `=` `[` $grid `]` `,`"
                       " `memory` `=` $memoryLayout `>`";

  let genVerifyDecl = 1;
}

def TTL_PipeRecordAttr : TTL_Attr<"PipeRecord", "pipe_record"> {
  let summary = "Static PipeNet pipe record";
  let description = [{
    Encodes one static PipeNet record for PipeNet foreach ops. The owning
    `#ttl.pipenet_records` attribute contains the PipeNet id; each record
    contains source and destination coordinates plus a flag selecting
    point-to-point or collective semantics.

    `isCollective` is a frontend-supplied flag, not derived from coordinates:
    `slice(i, i + 1)` is a singleton collective record with one destination
    node, but it must preserve the PipeNet-level collective contract.
  }];

  let parameters = (ins
    "int64_t":$srcX,
    "int64_t":$srcY,
    "int64_t":$dstStartX,
    "int64_t":$dstStartY,
    "int64_t":$dstEndX,
    "int64_t":$dstEndY,
    DefaultValuedParameter<"bool", "false">:$isCollective
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_PipeNetRecordsAttr : TTL_Attr<"PipeNetRecords", "pipenet_records"> {
  let summary = "Identifies a PipeNet and contains its static pipe records";
  let description = [{
    Identifies a PipeNet by id and contains the ordered list of
    `#ttl.pipe_record` attributes that make up the net. PipeNet foreach ops use
    this attribute to represent callback iteration without creating one
    source-level operation per record.

    `pipes` is in user PipeNet construction order. A launch node that matches
    multiple records executes them in this order. Duplicate records are
    distinct entries and execute as separate callback invocations.
  }];

  let parameters = (ins
    "int64_t":$pipeNetId,
    OptionalParameter<"::mlir::StringAttr">:$pipeNetName,
    ArrayRefParameter<"::mlir::tt::ttl::PipeRecordAttr">:$pipes
  );

  let assemblyFormat = "`<` `net` $pipeNetId (`name` $pipeNetName^)? "
                       "`pipes` `[` $pipes `]` `>`";
  let genVerifyDecl = 1;
}

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD
