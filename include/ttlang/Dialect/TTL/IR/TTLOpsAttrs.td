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
