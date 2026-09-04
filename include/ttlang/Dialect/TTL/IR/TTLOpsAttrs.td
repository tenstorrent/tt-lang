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
    Identifies one synchronized reset declaration and the complete set of
    logical kernels that participate on each active worker. Every participant
    reaches each corresponding dynamic instance before any participant
    returns. A declaration in an immutable sequential structured loop denotes
    one reset instance per iteration; participant loops must define the same
    nested iteration sequence. The reset establishes canonical read-pointer,
    write-pointer, occupancy, and initialization state for each target declared
    by the operation.

    The ordinal is module-local and has no runtime meaning. Equal attributes
    reference the same reset instance or statically repeated instance sequence.
    Distinct ordinals identify independent reset declarations. Participant
    order is canonical and otherwise semantically insignificant. The
    participant set contains one compute kernel and both data movement kernels
    that own DFB interface state. Each participant executes the declaration at
    most once at each point in the enclosing nested iteration sequence and
    launch node.
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

def TTL_DFBReconfigurationAttr
    : TTL_Attr<"DFBReconfiguration", "dfb_reconfiguration"> {
  let summary = "Worker-local synchronized DFB descriptor update";
  let description = [{
    Identifies one synchronized update to worker-local DFB descriptors and the
    complete set of logical kernels that participate on each active worker.
    Each reconfiguration call may execute at most once per dispatch and launch
    node, or once in every iteration of nested sequential loops with
    compile-time-known trip counts. Repeated execution requires at least two
    reconfiguration calls. Corresponding calls use equivalent loop nests and
    appear in the same order in every participant. Every declaration in one
    module uses the same participant set.
    At each call, the runtime waits for prior DFB-interface work to complete,
    installs the next descriptors, and then allows following DFB-interface work
    to begin. Independent math and SFPU work may overlap the descriptor update.

    This attribute identifies where synchronization and compiler-generated
    descriptor installation occur; it does not contain descriptor values.
    Source programs retain logical DFB identities across reconfiguration calls;
    physical indices and configurations are compiler-owned. Reconfiguration
    normally preserves a DFB's index, storage, and interface state while it has
    unread data or an incomplete protocol. Setting `discardDfbState` allows the
    lifecycle to end when its final access precedes the corresponding
    `ttl.dfb_reconfiguration` operation. If the index is reused, runtime resets
    its interface before the next DFB access.

    The ordinal is module-local and has no runtime meaning. Equal attributes
    reference the same descriptor update. Distinct ordinals identify distinct
    updates. When a reconfiguration sequence repeats, the final call restores
    every initial descriptor required by the next loop iteration.
    Participant order is canonical and otherwise semantically insignificant.
  }];
  let parameters = (ins
    "int64_t":$ordinal,
    ArrayRefParameter<"::mlir::tt::ttl::LogicalKernelAttr">:$participants,
    DefaultValuedParameter<"bool", "false">:$discardDfbState
  );
  let assemblyFormat = "`<` $ordinal `,` `participants` `[` $participants `]` (`,` `discard_dfb_state` `=` $discardDfbState^)? `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    static DFBReconfigurationAttr getCheckedInstance(
        ::mlir::Location location, ::mlir::MLIRContext *context,
        int64_t ordinal,
        ::llvm::ArrayRef<LogicalKernelAttr> participants,
        bool discardDfbState = false);
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

def TTL_DeviceDomainComponentAttr :
    TTL_Attr<"DeviceDomainComponent", "device_domain_component"> {
  let summary = "Named rectangular component of a device domain";
  let description = [{
    Defines one named rectangular index space in a logical device domain. The
    extent contains one positive value per dimension.
  }];
  let parameters = (ins
    "StringAttr":$name,
    "DenseI64ArrayAttr":$extent
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_DeviceDomainAttr : TTL_Attr<"DeviceDomain", "device_domain"> {
  let summary = "Logical index set of devices";
  let description = [{
    Defines the logical devices over which a transfer graph is expressed. A
    single component represents a regular domain. Multiple components form a
    product domain. Components describe only index-set semantics; physical
    topology and routing are not part of this attribute.

    Example:

    ```mlir
    #ttl.device_domain<components =
      <name = "board", extent = [2]>,
      <name = "device", extent = [4]>
    >
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"DeviceDomainComponentAttr">:$components
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_DeviceRefAttr : TTL_Attr<"DeviceRef", "device_ref"> {
  let summary = "Member of a logical device domain";
  let description = [{
    Identifies one logical device using one coordinate array per domain
    component. Coordinates use the component order declared by the associated
    device domain.
  }];
  let parameters = (ins
    ArrayRefParameter<"DenseI64ArrayAttr">:$coordinates
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_DeviceRangeAttr : TTL_Attr<"DeviceRange", "device_range"> {
  let summary = "Half-open range in a logical device domain";
  let description = [{
    Defines a rectangular range `[lo, hi)` using two device references from
    the same domain.
  }];
  let parameters = (ins
    "DeviceRefAttr":$lo,
    "DeviceRefAttr":$hi
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_TransferEdgeAttr : TTL_Attr<"TransferEdge", "transfer_edge"> {
  let summary = "Directed edge between logical devices";
  let description = [{
    Defines a transfer from one source device to either one destination device
    or a destination range. Exactly one destination form is present.
  }];
  let parameters = (ins
    "DeviceRefAttr":$source,
    OptionalParameter<"DeviceRefAttr">:$destination,
    OptionalParameter<"DeviceRangeAttr">:$destinationRange
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_DeviceTransferAttr : TTL_Attr<"DeviceTransfer", "device_transfer"> {
  let summary = "Device edge bound to a node-level pipe";
  let description = [{
    Binds one logical device-domain edge to a node-level pipe. The attribute
    records device semantics required by target lowering without describing a
    physical topology, route, link, or hop count.
  }];
  let parameters = (ins
    "DeviceDomainAttr":$domain,
    "TransferEdgeAttr":$edge
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
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

def TTL_FabricManagerEffectAttr
    : TTL_Attr<"FabricManagerEffect", "fabric_manager_effect"> {
  let summary = "External fabric manager lifetime effect";
  let description = [{
    Associates one external call with an operation-local fabric manager claim.
    Acquire ownership begins at call entry and remains live after a successful
    return. Release ownership ends when the call returns. A scoped effect owns
    the manager for the complete call.
  }];
  let parameters = (ins
    "StringAttr":$claim,
    "FabricManagerEffectKind":$kind
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def TTL_FabricManagerIntervalAttr
    : TTL_Attr<"FabricManagerInterval", "fabric_manager_interval"> {
  let summary = "Fabric connection-manager ownership interval";
  let description = [{
    Identifies one generated or external manager ownership interval in a
    logical kernel. Generated intervals reference the function's fabric route
    indices. External intervals reference an operation-local claim. The
    optional launch-node list records the exact nodes that execute an external
    interval. An absent list uses the enclosing kernel descriptor; a present
    empty list means no node executes the interval. The interference list
    contains interval identities whose lifetimes may overlap. Scoped external
    intervals in distinct structured-control regions are sequential when they
    execute on the same single launch node because each call releases its
    manager before returning.
  }];
  let parameters = (ins
    "StringAttr":$identity,
    "FabricManagerIntervalKind":$kind,
    OptionalParameter<"StringAttr">:$claim,
    "DenseI64ArrayAttr":$routeIndices,
    ArrayRefParameter<"StringAttr">:$interferingIntervals,
    OptionalParameter<"DenseI64ArrayAttr">:$launchNodes
  );
  let assemblyFormat = "`<` struct(params) `>`";
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
    - Tile or scalar element type, which distinguishes tiled from row-major
      storage
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
    point-to-point or collective semantics. `deviceTransfer` identifies the
    logical-device edge for graph PipeNets and is absent for local PipeNets.

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
    DefaultValuedParameter<"bool", "false">:$isCollective,
    OptionalParameter<"DeviceTransferAttr">:$deviceTransfer
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
