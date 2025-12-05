# Distributed tensor type proposal
Boyana Norris, Nov 20, 2005

During my recent [investigation](https://docs.google.com/document/d/1j_SxzQsv74LH5FemZC9IbkRcR-mAV8rfLDcDDgE2A4E/edit?usp=sharing) on using builtin MLIR memory management instead of custom allocator/deallocators (which are complex to develop and maintain), I ran into a dead end because of the irreconcilable semantics differences between builtin memrefs and tt-mlir distributed memrefs (with sharding attributes). 

I would like to propose an alternative architectural approach – create a custom tensor type specifically designed for TT-MLIR's distributed memory model and simplify resulting memrefs.

**`DistributedTensorType`**: A custom tensor type implementing `bufferization::TensorLikeType` for high-level IR (TTIR). During bufferization this tensor type lowers to **regular `MemRefType` shards** (e.g., `memref<32x32xf32, #ttcore.memory_space<l1>>`)—one per core. No special distributed memref type or complex layout attributes are required. After bufferization the IR contains only standard memrefs, exactly what TTKernel/TTMetal already expect.

### Current Architecture: MemRef + Layout Attributes

TT-MLIR currently uses MLIR's standard `MemRefType` with custom layout attributes:

```mlir
memref<4x4x1x1x!ttcore.tile<32x32, bf16>, 
       #ttcore.shard<2048x2048, 1>,        // Layout attribute
       #ttcore.memory_space<l1>>            // Memory space attribute
```

**Limitations:**
- `MemRefType` inherits from `BaseMemRefType`, which implements `ShapedType::Trait` and `PtrLikeTypeInterface::Trait`
- These traits assume single-buffer, contiguous memory semantics
- Operations like `memref.extract_strided_metadata` expect all memrefs to be decomposable into `(base_ptr, offset, strides)`
- Layout attributes can only modify how the memref is interpreted, not its fundamental type semantics

### Introduce a DistributedTensorType

The `DistributedTensorType` is a custom tensor type for high-level IR (TTIR) that implements `bufferization::TensorLikeType`. During one-shot bufferization each distributed tensor expands into a set of plain per-core `MemRefType` shards with only `#ttcore.memory_space` annotations—no `#ttcore.shard` layouts survive. This keeps distributed semantics at the tensor level, lets D2M/TTMetal continue operating on memrefs, and finally allows OwnershipBasedBufferDeallocation to run on standard buffers.

Important properties:
- **Explicit distribution metadata:** grid shape, shard shape, element type, and strategy are encoded as type parameters instead of ad-hoc layout attributes.
- **Tensor-first optimizations:** tiling, fusion, and other functional transforms stay on tensors; buffer-only passes work on the derived shard memrefs.
- **Selective interface implementation:** the type provides `ShapedType`/`TensorLikeType` APIs but deliberately avoids `BaseMemRefType`, blocking accidental use of memref-only utilities before bufferization.
- **Plain-memref bufferization:** `getBufferType()` returns `MemRefType::get(shardShape, elementType, /*layout=*/nullptr, memorySpace)`, so downstream passes see regular buffers.
- **Grouping metadata:** bufferization attaches a transient `ttcore.shard_group<tensor_id, shard_idx>` attribute to each shard memref so stream insertion, DMA planning, and layout passes can recover the logical tensor. A dedicated “strip shard grouping” pass removes this metadata right before running OwnershipBasedBufferDeallocation, ensuring the ownership pass only sees vanilla memrefs.

#### Type definition (illustrative)

```tablegen
def TTCore_DistributedTensorType : TTCore_Type<"DistributedTensor", "dist_tensor",
    [ShapedTypeInterface, ValueSemantics,
     DeclareTypeInterfaceMethods<Bufferization_TensorLikeTypeInterface>]> {
  let summary = "Distributed tensor type for multi-core execution";
  
  let description = [{
    Represents a tensor distributed across a grid of cores. The tensor is
    logically a single N-dimensional array, but physically stored as shards
    across multiple cores' local memories.
    
    During bufferization, this type is expanded into **plain memref shards**. Each
    core gets a normal `memref<shard_shape x element_type, #ttcore.memory_space<l1>>`
    value. No distributed layout attributes remain after bufferization—the IR is
    just a set of per-core memrefs that TTKernel/TTMetal already expect.
  }];
  
  let parameters = (ins
    "ArrayRef<int64_t>":$gridShape,      // Distribution across cores (e.g., 8x8)
    "ArrayRef<int64_t>":$shardShape,     // Per-core local shape (e.g., 32x32)
    "Type":$elementType,                  // Element type (e.g., f32)
    "DistributionStrategyAttr":$strategy // Sharded, interleaved, etc.
  );
  
  let extraClassDeclaration = [{
    // ShapedType interface
    Type getElementType() const { return getElementType(); }
    bool hasRank() const { return true; }
    ArrayRef<int64_t> getShape() const {
      // Full logical shape is grid_shape + shard_shape
      // For grid=8x8, shard=32x32 -> shape is [8, 8, 32, 32]
      SmallVector<int64_t> fullShape;
      fullShape.append(getGridShape().begin(), getGridShape().end());
      fullShape.append(getShardShape().begin(), getShardShape().end());
      return fullShape;
    }
    
    // TensorLikeType interface methods
    FailureOr<bufferization::BufferLikeType> getBufferType(
        const bufferization::BufferizationOptions &options,
        llvm::function_ref<InFlightDiagnostic()> emitError) const {
      // Each shard eventually becomes a normal memref of the shard shape in L1.
      return MemRefType::get(getShardShape(), getElementType(),
                             /*layout=*/nullptr,
                             ttcore::MemorySpaceAttr::get(
                                 getContext(), ttcore::MemorySpace::DeviceL1));
    }
    
    LogicalResult verifyCompatibleBufferType(
        bufferization::BufferLikeType bufferType,
        llvm::function_ref<InFlightDiagnostic()> emitError) const {
      auto memrefType = dyn_cast<MemRefType>(bufferType);
      if (!memrefType)
        return emitError() << "expected shard memref type";
      return success(memrefType.getShape() == getShardShape() &&
                     memrefType.getElementType() == getElementType());
    }
  }];
}
```

#### Why tensors?

Tensors capture immutable value semantics and enable higher-level optimizations (tiling, fusion, algebraic simplification) without prematurely committing to buffer layouts. Analyses and transforms are easier on tensors because the pure value semantics lets one reason algebraically about dataflow without tracking aliasing, mutability, or explicit memory layout the way one must with memrefs.

Memrefs are better suited for explicit memory planning (DMA coordinates, circular buffers). Keeping distribution metadata on tensors lets us optimize functionally, then hand well-structured shard memrefs to the D2M/TTMetal passes that already expect buffers.

In MLIR the idiomatic pattern is to keep high-level transforms on tensors (or other value-semantic types), then run one-shot bufferization to materialize concrete memref values, and only after that invoke the storage- and device-specific passes (affine lowering, DMA planning, stream insertion, deallocation). That separation lets tensor optimizations stay canonical while downstream passes rely on the standard buffer interfaces without duplicating tensor logic.

#### Bufferization flow

1. **Tensor stage (TTIR):** ops consume/produce `DistributedTensorType` values.
2. **One-shot bufferization:** each tensor value is expanded into per-core shard memrefs tagged with `ttcore.shard_group`.
3. **D2M/TTMetal passes:** allocation planning, stream insertion, DMA lowering, etc. operate on the tagged memrefs. Most of the listed passes will continue to work with little or no change once we hand them plain shard memrefs. The one outlier is `D2MGenericLowerDMAs`, which currently depends on `ttcore::DeviceLayoutInterface` for both grid and shard metadata and would need substantial re-plumbing to read that information from the proposed shard-group metadata instead of from the memref layout.
4. **Metadata strip:** once no pass needs shard topology, drop the grouping attribute so only plain memrefs remain.
5. **Ownership-based deallocation:** run MLIR’s stock pass to insert `bufferization.dealloc` on the vanilla shard memrefs.

For brevity, the rest of this document uses a shorthand handle
`!ttcore.shard_group<memref<...>>` to denote "the set of per-core memref shards
produced during bufferization, along with the metadata that ties them back to the
original distributed tensor." This is not a new buffer type; it is simply a
wrapper around already-materialized memrefs.

### Impact on Existing Pipelines

The `DistributedTensorType` approach would affect TT-MLIR's compilation pipelines at specific stages:

#### Frontend Pipeline (TTIR → D2M)
- **TTIR operations** would consume/produce `DistributedTensorType` instead of `RankedTensorType`
- **TTIRToD2M conversion** would create `d2m.generic` operations with distributed tensor operands
- **Grid selection and layout lowering** would operate on distributed tensors, setting distribution strategy attributes
- **No (significant) changes** to device registration, decomposition, or canonicalization passes
- `D2MGenericTileComputeLoops` could (should) be refactored to operate on tensors.

#### Middleend Pipeline (D2M optimization and bufferization)
- **Elementwise fusion** would fuse operations on distributed tensors
- **One-shot bufferization** would expand each `DistributedTensorType` into per-core memref shards with `ttcore.shard_group` metadata
- **Allocation/stream insertion** would operate on shard-group memrefs (no change to allocation logic, just different metadata)
- **All subsequent D2M passes** (tiling, register access, linearization, DMA lowering, loop generation, region extraction) would see plain memrefs with grouping metadata
- **New pass required**: "strip shard grouping" pass before deallocation to remove metadata
- **Deallocation**: Standard `OwnershipBasedBufferDeallocationPass` replaces custom pass

#### Backend Pipeline (D2M → TTKernel → TTMetal → EmitC)
- **No changes**: All passes operate on plain memrefs after bufferization
- D2M→TTKernel, TTKernel control flow, D2M→TTMetal, and EmitC lowering remain unchanged

#### Summary
The distributed tensor type exists **only in the frontend** (TTIR and early D2M). After bufferization, the rest of the pipeline sees standard memrefs, so the vast majority of passes require no modification. The main implementation effort is in:
1. Updating TTIR and TTIRToD2M to use the new type
2. Implementing `BufferizableOpInterface` for distributed tensor operations
3. Writing the shard-group metadata strip pass
4. Updating tests

### Advantages of DistributedTensorType Approach

#### 1. Explicit Distributed Semantics

The `DistributedTensorType` makes distributed memory a first-class concept:
- Grid dimensions are explicit type parameters, not hidden in layout attributes
- Operations can query whether a value is distributed via `isa<DistributedTensorType>`
- Type system enforces correct handling of distributed memory at compile time
- Value semantics (tensor) remain separate from reference semantics (standard memref shards)

#### 2. Selective Trait Implementation

Because this is a tensor type, we only implement the traits that make sense for tensors:
- `ShapedType`/`ValueSemantics` for rank, shape, and element-type queries.
- `bufferization::TensorLikeType` so one-shot bufferization knows how to lower it to memrefs.
- Any TT-specific “distributed memory” interface we invent for querying grid metadata.

We deliberately *avoid* memref-only traits and interfaces; the whole point is to keep distributed semantics in the tensor world until bufferization materializes plain memrefs.

#### 3. Operation Compatibility Control

Distributed semantics exist only at the tensor level. Once bufferization runs, the IR contains regular memrefs that all downstream dialects already understand. This keeps the rest of MLIR unaware of TT-MLIR's distributed model and avoids having to extend every memref-based operation.

#### 4. Deallocation Once Everything Is a Plain MemRef

Because the long-term plan is to bufferize distributed tensors down to ordinary shard memrefs (with only memory-space annotations), we do **not** need a custom per-core deallocation strategy. After bufferization and the TT-specific passes that still rely on shard metadata have completed, the IR contains only plain memrefs, so MLIR’s stock `OwnershipBasedBufferDeallocationPass` can run unmodified and insert the right `bufferization.dealloc` operations. No additional per-core `d2m.dealloc` pass is required.

#### 5. Better Error Messages

Type mismatches produce clearer errors:

```
error: 'bufferization.dealloc' op operand #0 must be standard memref,
but got 'memref<32x32xf32, #ttcore.shard<1024x1024, 1>, #ttcore.memory_space<l1>>'
annotated with a shard-group attribute
```

vs. the current:

```
error: 'memref.extract_strided_metadata' op operand #0 must be strided memref 
of any type values, but got 'memref<4x4x1x1x!ttcore.tile<32x32, bf16>, 
#ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>'
```

#### 6. Reuse of Existing Bufferization Patterns

Because the new type plugs directly into `bufferization::TensorLikeType`, we can keep MLIR’s standard one-shot bufferization pipeline. Operations implement `BufferizableOpInterface` once, and the infrastructure performs alias analysis, ownership propagation, and tensor→memref conversion without bespoke TT-specific passes.

### Challenges and Trade-offs

#### 1. Migration Effort

**Challenge:** Converting the entire stack (TTIR → D2M → TTKernel → TTMetal) to reference `DistributedTensorType` touches almost every file that currently hardcodes `MemRefType` or `RankedTensorType`.

**What must change:**
- **Type declarations:** Add the new type in `TTCoreOpsTypes.td`, implement printers/parsers, update ODS builders, and regenerate all `.inc` files.
- **Interfaces:** Teach every op participating in bufferization to implement `BufferizableOpInterface` for the new type; this spans TTIR math ops, D2M generics, DMA ops, and copy ops.
- **Pass pipelines:** Update conversion passes (`ttir-to-ttmetal`, `d2m-generic-generate-loops`, etc.) to recognize the tensor type at their entry points and emit plain memrefs by the time they exit.
- **Tests:** Rewrite more than 100 `.mlir` tests to use the new syntax, add canonicalization coverage, and adjust FileCheck expectations.

**Mitigation:** Stage the migration behind a feature flag. Keep both representations alive temporarily (current memref-with-attrs and new tensor type) and add conversion adapters so that each pipeline stage can opt in independently while CI continues to cover the legacy path.

#### 2. MLIR Infrastructure Assumptions

**Challenge:** Most **builtin MLIR utilities and passes** assume every buffer value implements `BaseMemRefType`. Examples:
- Operations such as `memref.dim`, `memref.subview`, and `memref.extract_strided_metadata` require a true memref with strided metadata.
- Bufferization and canonicalization helpers (`createFoldMemRefAliasOpsPass`, `createLowerAffinePass`, `mlir::bufferization::analysis` helpers) all call into these memref-specific utilities.
- TT-MLIR’s own middle-end passes reuse those helpers (e.g., `D2MGenericLinearizeMemref`, `D2MGenericLowerDMAs`) and therefore also expect plain memrefs by the time they run.

**Mitigation:** Keep the distributed semantics at the tensor level only. As soon as `DistributedTensorType` bufferizes to per-core memrefs, the existing builtin utilities continue to work unmodified. No parallel versions are needed; we simply delay the use of those utilities until after tensor bufferization is complete.

#### 3. Interoperability

**Challenge:** Builtin MLIR dialects (`linalg`, `scf`, `arith`, `tensor`) only accept builtin tensor or memref types. Feeding `DistributedTensorType` directly into them would violate verifier assumptions and break canonicalization patterns.

**Examples:**
- `linalg.generic` expects operands/results to implement `ShapedType`; it has no notion of grid metadata, so distributed tensors must be lowered to per-core tensors or memrefs first.
- `scf.for` region arguments are regular SSA values; region cloning utilities assume values have copy semantics, which distributed tensors lack without explicit adapters.
- Folded ops like `tensor.cast` or `bufferization.to_memref` cannot emit distributed types because their type converters understand only builtin kinds.

**Mitigation:** Introduce explicit “enter/exit distributed world” operations (to be defined, e.g., hypothetical `ttcore.materialize_shard` / `ttcore.pack_shards`). Standard dialects continue to operate on plain tensors/memrefs, and we wrap their use sites with these adapters wherever distributed values cross dialect boundaries.

```mlir
// Convert a shard-group handle to the local memref needed by linalg
%local_view = ttcore.extract_shard %a_shards[%core_x, %core_y]
  : !ttcore.shard_group<memref<..., #ttcore.memory_space<l1>>>
    -> memref<1x1x!ttcore.tile<32x32, f32>>

linalg.generic ins(%local_view : memref<...>) { ... }
```

#### 4. Tooling and Debugging

**Challenge:** MLIR’s built-in visualizers, print/debug utilities, and IR diff tools assume types adhere to the builtin type hierarchy. A new tensor type needs end-to-end support (AsmPrinter, AsmParser, Attr parser, diagnostics) or else developer workflows degrade quickly.

**Examples:**
- `mlir-opt --verify-diagnostics` will refuse to round-trip IR without precise printers/parsers for the distributed type parameters (grid shape, shard shape, strategy attr).
- IDE tooling (Cursor, VSCode MLIR plugins) depends on the textual form to “peek definition”; missing pretty printers result in unreadable dumps.
- Debug helpers in `ttmlir-opt --print-ir-after-all` currently search for `memref<... #ttcore.shard ...>` substrings; those heuristics would fail once the layout lives inside a custom type.

**Mitigation:** Ship printer/parser support together with the type, extend `mlir::AsmPrinter::printOptionalAttrDictWithKeyword` helpers to ensure strategy attrs render consistently, and enhance the in-tree IR pretty-printers (e.g., `TTMetalDumpPass`) to surface shard metadata. Supplement with documentation/screenshots so developers know how to interpret the new syntax.

### Comparison with Current Approach

| Aspect | Current (MemRef + Attrs) | Custom Tensor Type |
|--------|-------------------------|-------------|
| **Type Safety** | Weak (layout in attributes) | Strong (distribution in type) |
| **MLIR Compatibility** | High (standard type) | Low (custom infrastructure) |
| **Error Detection** | Runtime/verification | Compile-time (type system) |
| **Implementation Effort** | Low (reuse existing) | High (new infrastructure) |
| **Semantic Clarity** | Implicit (hidden in attrs) | Explicit (in type) |
| **Deallocation** | Requires workarounds | Natural type-specific logic |
| **Migration Cost** | None (current state) | Very high |

### Precedents in MLIR Ecosystem

#### Anything we can use directly?

A few MLIR efforts already flirt with distributed tensor semantics, but none map directly to TT-MLIR’s needs:

- **`shard` dialect tensors** – Upstream MLIR has a `shard.shard` op plus sharding attributes that annotate plain tensor types with device-grid metadata. That’s the closest analogue to what we’re proposing, but it stops at the tensor level: there’s no lowering story to per-core memrefs or TT-style DMA queues, which is why we’d still need our own type even if we adopt some ideas from the dialect [https://mlir.llvm.org/docs/Dialects/Shard/](https://mlir.llvm.org/docs/Dialects/Shard/).

- **`nvgpu` / `gpu` dialect tiling types** – These dialects model warp- or block-level fragments (`nvgpu.mma.sync`, `gpu.subgroup_mma`) and sometimes wrap tensors with layout metadata, but they assume SIMT GPUs with shared memory, not TT’s multi-core mesh with explicit L1 shards. They’re useful inspiration for tiling interfaces, yet the runtime assumptions are incompatible.

- **IREE and other downstream projects** – Some out-of-tree compilers (IREE, TPU backends) experiment with “distributed” tensor annotations to drive SPMD lowering, but they rely on custom attributes over builtin tensor types. Again, they don’t address per-core bufferization or the OwnershipBasedBufferDeallocation gap we’re solving here, so importing them directly would only replicate our current attr-heavy approach.

Our choices essentially boil down to: (1) continue with `MemRef + #ttcore.shard`, (2) try to fully adopt the upstream `shard` dialect, or (3) introduce a TT-specific `DistributedTensorType` that explicitly lowers to plain shard memrefs. Option (3) is the only one that simultaneously keeps high-level tensor semantics, plugs into One-Shot Bufferization, and lets us re-enable OwnershipBasedBufferDeallocation once the IR is down to standard memrefs without incurring a possibly problematic external dependency (`shard`).

Similar (but not directly usable) custom memory types exist in other MLIR dialects:

1. **SPIR-V Dialect**: Custom pointer types with different storage classes
2. **GPU Dialect**: Separate types for device vs. host memory
3. Author's prior compiler writing experiences also suggest that investing the effort in designing a custom tensor-level type leads to huge complexity and effort savings in the rest of the compilation pipeline.


#### Gotchas for Distributed Tensors

**1. Limited `BufferizableOpInterface` Coverage**

Our ops (e.g., `ttcore.extract_shard`, `d2m.generic`) must teach bufferization how shards relate. Without custom aliasing methods, One-Shot Bufferization can't prove that shard SSA values don't alias across cores. The solution is to implement `BufferizableOpInterface` for the ops that create/extract shards.

**2. Buffer Allocation Strategy**

Standard bufferization assumes single allocation per tensor. Distributed tensors need per-core allocation:

```mlir
// Conceptual: One logical tensor, multiple physical buffers
%dist_tensor = bufferization.alloc_tensor() 
  : ttcore.dist_tensor<grid=8x8, shard=32x32xf32>

// Lowered to D2M: 64 separate allocations (8x8 grid)
d2m.generic {grid = #ttcore.grid<8x8>, ...} {
^bb0(%core_x: index, %core_y: index):
  %local_buf = memref.alloc() : memref<32x32xf32, #ttcore.memory_space<l1>>
  // ... use local_buf ...
}
```

**Solution**: The bufferization pass tags every shard memref with a `ttcore.shard_group` attribute that stores the parent distributed value and the shard coordinates. Passes that build DMAs or streams read this attribute to recover the logical tensor, and OwnershipBasedBufferDeallocation uses the same tag to insert one `bufferization.dealloc` per shard of the group.

**3. Copy Semantics**

Standard `bufferization.copy` doesn't understand distributed layouts:

```mlir
// Problem: How to copy distributed tensor?
bufferization.copy %src, %dst : 
  ttcore.dist_tensor<grid=8x8, shard=32x32xf32>
```

**Solution**: Lower to distributed copy operation:

```mlir
d2m.generic {grid = #ttcore.grid<8x8>, ...}
    ins(%src_shards : !ttcore.shard_group<memref<32x32xf32, #ttcore.memory_space<l1>>>) 
    outs(%dst_shards : !ttcore.shard_group<memref<32x32xf32, #ttcore.memory_space<l1>>>) {
^bb0(%src_shard: memref<32x32xf32>, %dst_shard: memref<32x32xf32>):
  memref.copy %src_shard, %dst_shard
  d2m.yield
}
```

**4. Deallocation Timing**

Ownership-based deallocation only runs after the “strip shard grouping” step, which itself is scheduled after all grid-level synchronization has been lowered into the IR (e.g., via `d2m.barrier`, launch-grid semantics, or TTMetal runtime intrinsics). As a result, every `bufferization.dealloc` that the upstream pass emits occurs after the necessary synchronization points, and we no longer need custom `d2m.dealloc` operations or manual barrier insertion in this phase.

**5. Interoperability with Standard Dialects**

Standard dialects (Linalg, SCF, affine) don't understand distributed types:

```mlir
// Problem: Linalg expects standard tensors
linalg.matmul ins(%A, %B : ttcore.dist_tensor<...>, ttcore.dist_tensor<...>)
              outs(%C : ttcore.dist_tensor<...>)
// Error: linalg.matmul doesn't accept DistributedTensorType
```

**Solution**: Explicit conversion at dialect boundaries:

```mlir
// Option 1: Lower distributed op to d2m.generic working on shard groups
%C = ttir.matmul %A, %B : ttcore.dist_tensor<...> -> ttcore.dist_tensor<...>
// Lowers to:
d2m.generic {grid = ...}
    ins(%A_shards, %B_shards :
         !ttcore.shard_group<memref<..., #ttcore.memory_space<l1>>>)
    outs(%C_shards :
         !ttcore.shard_group<memref<..., #ttcore.memory_space<l1>>>) {
  // Per-core linalg.matmul on local shards
  linalg.matmul ins(%A_local, %B_local : memref<...>) outs(%C_local : memref<...>)
}

// Option 2: Materialize as standard tensor (expensive!)
%A_g = ttcore.gather %A : ttcore.dist_tensor<...> -> tensor<...>
%C_g = linalg.matmul ins(%A_g, %B_g : tensor<...>)
%C = ttcore.scatter %C_g : tensor<...> -> ttcore.dist_tensor<...>
```

#### Comparison: Status Quo vs. DistributedTensorType

| Aspect | MemRef + Layout Attrs (Today) | DistributedTensorType → Plain MemRefs |
|--------|--------------------------------|---------------------------------------|
| **Type Safety** | Layout hidden in attrs; easy to misuse | Distribution encoded in the type system |
| **Bufferization** | Requires per-pass ad-hoc logic | One-Shot Bufferization handles tensors uniformly |
| **Alias Analysis** | Manual reasoning per pass | Reuses MLIR infrastructure once converted to memrefs |
| **MLIR Idiomaticity** | Diverges from tensor→memref pattern | Matches standard tensor semantics |
| **Deallocation** | Ownership pass fails on shard layouts | Ownership pass succeeds after tensor lowering |
| **Migration Cost** | None (already deployed) | High (new type + pass updates + tests) |

#### Summary

**For TT-MLIR's use case**, the **DistributedTensorType → plain memref shards** approach is desirable because:

1. **Matches existing architecture**: TT-MLIR already separates tensor-level TTIR from buffer-level D2M; the new type simply makes the tensor side explicit.
2. **Leverages One-Shot Bufferization**: Reuses mature MLIR infrastructure (alias analysis, buffer placement) by plugging into `TensorLikeType`.
3. **Clearer semantics**: Tensors remain functional/immutable; all mutable state is confined to standard memrefs after bufferization.
4. **Ownership-based deallocation becomes viable**: Once shards are plain memrefs, we can re-enable `OwnershipBasedBufferDeallocationPass` without custom infrastructure.

**Implementation path**:
1. Create `DistributedTensorType` with the `TensorLikeType` interface.
2. Implement `getBufferType()` to materialize shard memrefs (using existing memory-space attributes).
3. Update TTIR/D2M ops to accept/produce the new tensor type and teach bufferization how to lower them to shards.
4. Limit the use of passes that require `#ttcore.shard` metadata to the pre-bufferization portion of the pipeline.
5. Invoke One-Shot Bufferization followed immediately by Ownership-Based Deallocation once only plain memrefs remain.
