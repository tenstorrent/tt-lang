// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD
#define TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD

include "ttlang/Dialect/TTL/IR/TTLBase.td"
include "ttlang/Dialect/TTL/IR/TTLOpsEnums.td"

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

def TTL_DeviceDomainAttr : TTL_Attr<"DeviceDomain", "device_domain"> {
  let summary = "Flat or hierarchical logical device domain";
  let description = [{
    Defines the logical device domain over which transfer graphs are expressed.

    Levels are stored as dictionaries inside the aggregate attribute so the
    domain remains one canonical IR object. Each level dictionary contains:
    - `name`: string
    - `extent`: `DenseI64ArrayAttr`
    - `topology`: string (e.g., `fabric_1d` or `fabric_ring`)
    - `cluster_axis`: integer
    - `periodic`: bool
    - `mesh_id`: integer

    Storage class: compile-time metadata only. The attribute stores O(L + R)
    data for L levels and total rank R. It never stores a dense domain table,
    adjacency table, role table, PipeNet table, or device-visible state.
  }];
  let parameters = (ins
    "ArrayAttr":$levels
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTL_TransferGraphAttr : TTL_Attr<"TransferGraph", "transfer_graph"> {
  let summary = "Device-level transfer graph over a device domain";
  let description = [{
    Describes device-level communication independently from core endpoints and
    DFB operands. Core/DFB binding happens at the PipeNet or transfer site.

    Edges and structured descriptors are stored as dictionaries inside the
    aggregate attribute so helper concepts do not become independent IR
    objects. Explicit edge dictionaries contain `source` plus exactly one of
    `destination` or `destination_range`. Device coordinates are arrays of
    per-level `DenseI64ArrayAttr` values in domain level order.

    Storage class: compile-time metadata only. Explicit graphs store O(E)
    user edges. Structured graphs store O(1) descriptors. This attribute does
    not allocate runtime host state or device-visible communication memory;
    later lowering must allocate only local live sender/receiver state and
    queue-depth-dependent state.
  }];
  let parameters = (ins
    "DeviceDomainAttr":$domain,
    "ArrayAttr":$edges,
    OptionalParameter<"DictionaryAttr">:$structured
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

#endif // TTLANG_DIALECT_TTL_IR_TTLOPSATTRS_TD
