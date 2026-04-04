# DSL Expression to MLIR Translation

## Overview

Block expressions replace eager elementwise emission with lazy expression
nodes (`ttl.block_expr.*` ops) that are materialized into `ttl.compute`
ops at store sites. This eliminates dominance problems caused by eager
emission ordering, makes fusion structural rather than analytical, and
aligns the MLIR representation with the TTLang spec's deferred
`BlockExpr` semantics.

## Problem

Three related problems motivated this design.

### Fill type resolution

The `fill(value)` operation creates a tensor initialized to a constant.
Its result type (shape, dtype) depends on the output DFB it will
eventually be stored to, but eager emission requires a concrete type at
the Python evaluation site -- before any reserve exists:

```python
y = ttl.math.fill(0)           # what type? no reserve yet
for _ in range(KT):
    y += a_blk @ b_blk
with y_dfb.reserve() as y_blk:
    y_blk.store(y)             # type is known HERE
```

Lazy expressions defer type resolution: `fill` emits a
`ttl.block_expr.fill` that carries the value but not the final type.
The materializer resolves the type at the store site, where the output
DFB's shape and dtype are concrete.

### Compute placement and dominance

The eager pipeline emits `ttl.add` when Python `+` evaluates, before
the output DFB reserve may exist:

```python
result = av + bv                    # eagerly emits ttl.add HERE
with out_dfb.reserve() as o:       # cb_reserve comes AFTER the add
    o.store(result)
```

The resulting `ttl.compute` (created by `ConvertTTLToCompute`) is
positioned before the reserve. The `tile_store` inside the compute body
references the reserve view, which does not dominate the compute. The
current code works around this by moving the compute op forward
(`computeOp->moveBefore(op)`).

Lazy expressions eliminate this: the compute is created at the store
position, which is always after the reserve by construction (Python
`with` block guarantees this).

### Analytical fusion complexity

Fusion in the eager pipeline is analytical: `traceFusionToRoots()` walks
backward from a sink op through elementwise ops, collecting connected
ops into a fusion trace, then `buildFusedCompute()` creates a single
`ttl.compute` from the trace. This logic must handle fusion boundaries
(matmul as a leaf, broadcast requiring DFB-attached input), diamond
detection (deduplication via `SmallSetVector`), and broadcast dimension
compatibility. With lazy expressions, the block expression DAG rooted
at the store IS the fusion graph, eliminating the need for separate
tracing logic.

## Solution: Lazy Block Expressions

### Expression Ops

All expression-producing operations emit `ttl.block_expr.*` ops. No
eager elementwise ops are emitted by the Python DSL:

- Arithmetic operators (`+`, `-`, `*`, `/`) emit
  `ttl.block_expr.add`, `.sub`, `.mul`, `.div`
- Unary math functions (`exp`, `abs`, `sqrt`, etc.) emit
  `ttl.block_expr.exp`, `.abs`, `.sqrt`, etc.
- `ttl.math.broadcast` emits `ttl.block_expr.bcast`
- `ttl.math.fill` emits `ttl.block_expr.fill`
- Matrix multiply (`@`) emits `ttl.block_expr.matmul`

Block expression ops carry the `Pure` trait (no side effects, no hardware
resources consumed). They exist only before the materialization pass.
After materialization, only `ttl.compute` with tile ops remains.

The eager tensor-level elementwise ops (`ttl.add`, `ttl.exp`, etc.)
still exist in the dialect definition for backward compatibility with
existing MLIR lit tests, but are not emitted by the DSL frontend.

### Materialization

The `materialize-block-exprs` pass handles all `ttl.store` ops in a
function:

For stores whose tensor operand is a block_expr result:

1. Trace the store's tensor operand backward through block_expr ops to
   find DFB-attached root inputs.
2. Collect all block_expr ops in topological order.
3. Create `ttl.compute` at the store position, after all reserves.
4. Emit tile ops in the compute body matching the expression DAG, plus
   `ttl.tile_store` for each store.
5. Erase the block_expr ops and the tensor-level store.

For passthrough stores (DFB-attached input, no block_expr ops):

1. Create a `ttl.compute` with just a `tile_store` (no tile compute ops).

No reordering is needed in either case -- the compute is at the store
position, which is after all reserves by construction.

### Fusion

Fusion is structural: the block expression DAG rooted at the store IS
the fusion graph. The materializer's backward trace through
`block_expr.*` ops determines what fuses. The trace stops at values not
produced by block_expr ops (DFB-attached values from `attach_cb`,
`cb_wait`).

The matmul+add fold is directly visible in the expression DAG:

```mlir
%mm  = ttl.block_expr.matmul %a, %b
%sum = ttl.block_expr.add %acc, %mm
ttl.store %sum, %view
```

The materializer detects that the add's operand is a matmul with a
single user and emits a 3-operand `ttl.tile_matmul_block(a, b, acc)`
instead of separate matmul and add tile ops.

### Multi-Store Grouping

When multiple stores consume the exact same SSA value, they are grouped
into a single compute with one set of tile ops and N `tile_store`
emissions:

```python
result = av + bv
o1.store(result)    # same value
o2.store(result)    # same value
```

Produces one compute with two `tile_store` ops rather than two
independent computes that duplicate the add.

Grouping is restricted to stores consuming the identical SSA value.
Stores whose expression DAGs share sub-expressions but consume different
final values remain in separate computes. This constraint arises because
`init_sfpu` configures the hardware PACK unit for a single output
dataflow buffer; packing different DST register values to different
output buffers within the same sync region requires PACK reconfiguration
that the current init infrastructure does not support.

### Impact on Signposts and DPrints

Block expression ops carry the `Pure` trait, so they have no fixed
ordering relative to side-effecting ops (signposts, dprints). When the
materializer erases block_expr ops and creates a compute body, any
signpost or dprint ops that were interleaved with the block_expr ops in
the source IR would be orphaned -- left in the parent block outside the
compute, wrapping the outer loop structure rather than the inner tile
operations where they belong.

The materializer addresses this by collecting interleaved side-effect ops
before materialization and replaying them at the corresponding positions
inside the compute body. Collection walks the block in three sweeps:

- Leading: signpost/dprint ops immediately before the first block_expr
  op (walked backward until a non-side-effect op is hit).
- Interleaved: signpost/dprint ops between consecutive block_expr ops.
  Each is associated with the next block_expr op in topological order,
  so it is emitted before that op's tile counterpart.
- Trailing: signpost/dprint ops after the last block_expr op, up to the
  first `cb_push`/`cb_pop`. For multi-store groups, trailing ops are
  partitioned per store: begin signposts before a store and end signposts
  after it are kept together, so each store's profiling scope wraps its
  own `tile_store`.

## Pipeline

```
[Python DSL emits ttl.block_expr.add, ttl.block_expr.exp, etc.]
materialize-block-exprs    <- block_expr stores + passthrough stores
set-compute-kernel-config
assign-dst
subblock-compute-for-dst   (if --ttl-maximize-dst)
insert-tile-regs-sync
lower-to-loops
...
```

`materialize-block-exprs` handles all store materialization:
block_expr-based stores produce compute ops with fused tile ops,
passthrough stores produce compute ops with just a `tile_store`.
`ConvertTTLToCompute` is no longer needed in the pipeline.

All passes after compute construction (`assign-dst`,
`insert-tile-regs-sync`, `lower-to-loops`, `convert-ttl-to-ttkernel`)
operate on the same `ttl.compute` IR regardless of which materialization
produced it.

## Files

| File | Role |
|------|------|
| `include/ttlang/Dialect/TTL/IR/TTLOps.td` | Block expression + tile op definitions (via multiclass) |
| `lib/Dialect/TTL/Transforms/TTLMaterializeBlockExprs.cpp` | Materialization pass (block_expr + passthrough) |
| `python/ttl/operators.py` | Python operators emit block_expr ops |
| `python/ttl/ttl_api.py` | Pipeline configuration |
| `python/gen_elementwise.py` | Generates block_expr op bindings from `TTLElementwiseOps.def` |
