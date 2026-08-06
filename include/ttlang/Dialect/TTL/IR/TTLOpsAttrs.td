// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLAttrInterfaces.td"
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

def TTL_AxisNeighborTransferAttr :
    TTL_Attr<"AxisNeighborTransfer", "axis_neighbor_transfer",
             [TTL_StructuredTransferAttrInterface]> {
  let summary = "Transfer relation between axis neighbors";
  let description = [{
    Describes transfers from each device to the device at the specified
    positive offset along one logical domain axis. `wrap` includes transfers
    that cross the axis boundary.
  }];
  let parameters = (ins
    "StringAttr":$component,
    "int64_t":$axis,
    "int64_t":$offset,
    "BoolAttr":$wrap
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_StencilTransferAttr :
    TTL_Attr<"StencilTransfer", "stencil_transfer",
             [TTL_StructuredTransferAttrInterface]> {
  let summary = "Transfer relation defined by neighbor offsets";
  let description = [{
    Describes the union of directed translations in `offsets` within one
    logical domain component. Each offset has the component rank and may
    contain positive and negative values. `wrap` applies each translation
    modulo the component extent; otherwise, transfers outside the component
    are omitted.
  }];
  let parameters = (ins
    "StringAttr":$component,
    ArrayRefParameter<"DenseI64ArrayAttr">:$offsets,
    "BoolAttr":$wrap
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_GatherTransferAttr :
    TTL_Attr<"GatherTransfer", "gather_transfer",
             [TTL_StructuredTransferAttrInterface]> {
  let summary = "Transfer relation from every device to one root";
  let description = [{
    Describes transfers within each product-domain slice from every non-root
    device in `component` to `root`. The root contains only the selected
    component's coordinate.
  }];
  let parameters = (ins
    "StringAttr":$component,
    "DeviceRefAttr":$root
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_ScatterTransferAttr :
    TTL_Attr<"ScatterTransfer", "scatter_transfer",
             [TTL_StructuredTransferAttrInterface]> {
  let summary = "Transfer relation from one source to every other device";
  let description = [{
    Describes transfers within each product-domain slice from `source` to every
    other device in `component`. The source contains only the selected
    component's coordinate.
  }];
  let parameters = (ins
    "StringAttr":$component,
    "DeviceRefAttr":$source
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_TransferGraphAttr : TTL_Attr<"TransferGraph", "transfer_graph"> {
  let summary = "Transfer relation over a logical device domain";
  let description = [{
    Describes device-level communication independently from node endpoints and
    dataflow buffer operands. The relation is either an explicit non-empty edge
    list or one structured transfer descriptor that denotes at least one edge.
  }];
  let parameters = (ins
    "DeviceDomainAttr":$domain,
    OptionalArrayRefParameter<"TransferEdgeAttr">:$edges,
    OptionalParameter<"StructuredTransferAttrInterface">:$structured
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
    multiple records executes them in this order.
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
