# DST Register Allocation

## Overview

DST (destination) registers are hardware registers used for tile computations. This pass (`TTLAssignDST.cpp`) assigns DST indices to tile operations based on their execution category.

## Core Principle: Operation Classification

Each tile operation falls into one of the following categories based on
its execution engine and DST register usage:

**SFPU binary operations** (e.g., `mul_binary_tile`, `add_binary_tile`)
read both operands from DST and write to a fresh DST slot:
- Each input gets its own DST slot (loaded via `copy_tile`)
- Output gets a fresh DST slot
- `inputs_footprint = number of tile inputs`

**FPU binary operations** (e.g., `add_tiles`, `sub_tiles`, `mul_tiles`)
read both operands from circular buffers via the SrcA/SrcB unpackers and
write the result directly to DST:
- Inputs stay in CBs and do not occupy DST slots
- Output gets a fresh DST slot
- `inputs_footprint = 0`

**`binary_dest_reuse_tiles`** reads one operand from DST (an
intermediate result from a prior operation) and unpacks the other from a
CB. The result overwrites the DST operand in-place:
- DST operand: already allocated (no new slot)
- CB operand: stays in CB (no DST slot)
- Output overwrites the DST operand's slot
- `inputs_footprint = 0` (DST input is reused, CB input is not in DST)

**Unary operations** (e.g., `exp_tile`, `abs_tile`, `neg_tile`) operate
in-place on a DST register:
- Output reuses the input's DST slot
- No additional DST allocation needed
- `inputs_footprint = 0` for standalone unary ops

**Broadcast** (`tile_bcast`) reads a single operand from a CB and
writes the broadcast result to a fresh DST slot:
- Input stays in CB (like `copy_tile`, uses `TTLCBInputTileOpTrait`)
- Output gets a fresh DST slot (not in-place)
- `inputs_footprint = 0`
- **Init constraint**: `unary_bcast_init` performs a full init
  (configures UNPACK + MATH + PACK) and must appear before any other
  init operations in the block.

**Reduce** (`tile_reduce`) reads an input tile and a scaler tile from
CBs, and accumulates the reduction result into a fresh DST slot:
- Both inputs stay in CBs (`TTLCBInputTileOpTrait`)
- Output gets a fresh DST slot (not in-place)
- `inputs_footprint = 0`
- **Accumulation**: Multiple `reduce_tile` calls to the same DST index
  accumulate the reduction across tiles (e.g., summing a row of tiles).
  The DST slot must remain live for the entire reduction loop.
  `tile_regs_acquire` implicitly zeros DST before accumulation begins;
  each `reduce_tile` adds to the existing DST value:
  ```cpp
  // From tt-metal: tests/tt_metal/tt_metal/test_kernels/compute/reduce_w.cpp
  acquire_dst();                              // zeros DST
  for (uint32_t wt = 0; wt < Wt; ++wt) {
      cb_wait_front(c_0, onetile);
      reduce_tile(c_0, c_2, 0, 0, 0);        // accumulates into DST[0]
      cb_pop_front(c_0, onetile);
  }
  pack_tile(0, c_16);                         // pack accumulated result
  release_dst();
  ```
- **Init constraint**: `reduce_init` performs a full init (UNPACK +
  MATH + PACK) and sets a packer edge mask. `reduce_uninit` must be
  called after the last `reduce_tile` to clear this mask before any
  non-reduce operation.

**Matmul** (`tile_matmul`) reads A and B tiles from CBs and
accumulates the matrix product into a fresh DST slot:
- Both inputs stay in CBs (`TTLCBInputTileOpTrait`)
- Output gets a fresh DST slot (not in-place)
- `inputs_footprint = 0`
- **Accumulation**: Multiple `matmul_tiles` calls to the same DST index
  accumulate C += A * B across K-dimension tiles. The DST slot must
  remain live across the entire K loop. `tile_regs_acquire` implicitly
  zeros DST before the loop; all K iterations accumulate before
  `tile_regs_commit`:
  ```cpp
  // From tt-metal: programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp
  tile_regs_acquire();                        // zeros DST
  for (uint32_t kt = 0; kt < Kt; kt++) {
      cb_wait_front(cb_in0, 1);
      cb_wait_front(cb_in1, 1);
      matmul_tiles(cb_in0, cb_in1, 0, 0, 0); // accumulates into DST[0]
      cb_pop_front(cb_in0, 1);
      cb_pop_front(cb_in1, 1);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_tile(0, cb_out);                       // pack accumulated result
  tile_regs_release();
  ```
  When K is split into multiple blocks that exceed DST capacity, partial
  results are spilled to an intermediate CB and reloaded with
  `copy_tile` in subsequent blocks (see
  `bmm_large_block_zm.cpp` in tt-metal for this pattern).
- **Init**: `mm_init_short` configures UNPACK + MATH (not PACK). Can
  be called multiple times per kernel to re-enter matmul mode after
  other operations.

**Transpose (CB → DST)** (`transpose_wh_tile`) reads a tile from a CB,
transposes width and height dimensions (B[w,h] = A[h,w]), and writes
the result to a fresh DST slot:
- Input stays in CB (`TTLCBInputTileOpTrait`)
- Output gets a fresh DST slot (not in-place)
- `inputs_footprint = 0`
- **Init constraint**: `transpose_wh_init(icb, ocb)` performs a full
  init (configures UNPACK + MATH + PACK). `transpose_wh_init_short(icb)`
  reconfigures UNPACK + MATH only (for switching from another op).
  `transpose_wh_uninit(icb, ocb)` must be called after the last
  transpose before re-initializing for another operation.
- **Int32 path**: For int32 data, the hardware uses a different code
  path internally: `copy_tile_to_dst` with `UnpackToDestEn` followed by
  `transpose_dest` (an in-place DST transpose). For non-int32 data, the
  transpose is performed during unpack via the XY transpose flag.

**Transpose (in-place DST)** (`transpose_wh_dest`) transposes a tile
already in a DST register in-place (B[w,h] = A[h,w]):
- Output reuses the input's DST slot
- No additional DST allocation needed
- `inputs_footprint = 0` for standalone transpose
- **Init**: `transpose_wh_dest_init_short()` takes no arguments
  (configures MATH only).
- **Note**: Present in tt-metal (`compute_kernel_api/transpose_wh_dest.h`)
  but not yet modeled in the TTKernel dialect. Will need TTKernel ops
  (`transpose_wh_dest`, `transpose_wh_dest_init_short`) and a TTL op
  before it can be used in the compiler pipeline.

The following table summarizes DST slot requirements per category:

| Category | Input Source | DST Input Slots | DST Output Slots | In-Place | Accumulates |
|----------|-------------|-----------------|------------------|----------|-------------|
| SFPU binary | DST (both) | 2+ | 1 (fresh) | No | No |
| FPU binary | CB (both) | 0 | 1 (fresh) | No | No |
| `dest_reuse` | 1 DST + 1 CB | 0 (reused) | 1 (overwrites input) | Yes | No |
| Unary | DST | 0 | 0 (overwrites input) | Yes | No |
| Broadcast | CB | 0 | 1 (fresh) | No | No |
| Reduce | CB (input + scaler) | 0 | 1 (fresh) | No | Yes (reduction dim) |
| Matmul | CB (A + B) | 0 | 1 (fresh) | No | Yes (K dim) |
| Transpose (CB) | CB | 0 | 1 (fresh) | No | No |
| Transpose (DST) | DST | 0 | 0 (overwrites input) | Yes | No |

### Operation Category Traits

Each category is identified by a composition of four orthogonal traits
defined in TableGen. One already exists: `TTLCBInputTileOpTrait` marks
operations that read tile inputs from CB rather than DST. Three
additional traits complete the system:

- **`TTLDSTInputsTrait`**: At least one operand is consumed from DST.
- **`TTLInPlaceOpTrait`**: Result overwrites the DST input (input and
  output share the same DST slot).
- **`TTLAccumulatingOpTrait`**: Result accumulates across multiple
  invocations to the same DST index (the DST slot must remain live
  across an accumulation loop).

Each operation category is uniquely identified by its trait combination:

| Category | `DSTInputs` | `CBInput` | `InPlace` | `Accumulating` |
|----------|:-----------:|:---------:|:---------:|:--------------:|
| SFPU binary | x | | | |
| FPU binary | | x | | |
| `dest_reuse` | x | x | x | |
| Unary | x | | x | |
| Broadcast | | x | | |
| Reduce | | x | | x |
| Matmul | | x | | x |
| Transpose (CB) | | x | | |
| Transpose (DST) | x | | x | |

The allocator queries these traits compositionally:
- `hasTrait<TTLCBInputTileOpTrait>()` -> operand stays in CB (no DST
  slot needed for that operand)
- `hasTrait<TTLDSTInputsTrait>()` -> operand needs a DST slot
- `hasTrait<TTLInPlaceOpTrait>()` -> merge input/output intervals
  (Phase 2)
- `hasTrait<TTLAccumulatingOpTrait>()` -> DST slot must remain live
  across the accumulation loop

## Lifetime Interval Visualization

The following diagram illustrates how lifetime intervals work for an
SFPU binary followed by a unary operation. Both block arguments are
loaded to DST via `copy_tile` before the SFPU binary consumes them:

```
MLIR:
  %in0 = block_arg
  %in1 = block_arg
  %0 = mul(%in0, %in1)    // SFPU binary (both operands from DST)
  %1 = abs(%0)             // Unary (in-place)
  yield %1

Operation Timeline:    0      1      2      3
                       |      |      |      |
Lifetime Intervals:
  %in0  [0────1]       █████
  %in1  [0────1]       █████
  %0    [1─────────3]         ████████████    ← Merged with %1
  %1    [2─────────3]              █████████  ← (unary in-place)

DST Assignment:
  Inputs/Intermediates: %in0->DST[0], %in1->DST[1]
  Outputs:              %0->DST[2], %1->DST[2] (shared, in-place)
```

**Key observations**:
- Block arguments (`%in0`, `%in1`) start at operation 0
- Binary op result (`%0`) starts at its definition (op 1)
- Unary op result (`%1`) merges with its input (`%0`) - both use same DST
- Input intervals are short (expire after first use)
- Output interval extends to yield operation

## DST Allocation Algorithm

The algorithm has five phases: optional operation scheduling to minimize register pressure (Phase 0), IR normalization via copy insertion (Phase 1), live interval construction with in-place merging (Phase 2), linear scan allocation for inputs and intermediates (Phase 3), and linear scan allocation for outputs (Phase 4).

**References**:
- Christian Wimmer and Michael Franz. 2010. Linear scan register allocation on SSA form. In Proceedings of CGO '10. https://doi.org/10.1145/1772954.1772979
- P. S. Rawat et al. 2019. Associative instruction reordering to alleviate register pressure. In Proceedings of SC '18. https://doi.org/10.1109/SC.2018.00049

### Phase 0: Operation Scheduling for Register Pressure (Optional)

Reorder independent operations within the `ttl.compute` block to minimize register pressure and reduce the number of copies needed in Phase 1.

**Goals**:
- Schedule full-init operations (broadcast, reduce) first to avoid
  clobbering prior init configurations (see Init Ordering below).
- Schedule in-place consumers (unary, `dest_reuse`) last among a
  value's consumers to eliminate `copy_dest_values` operations.

**Algorithm** (adapted from LARS [Rawat et al. SC'18]):

```
# Build SSA dependency graph
deps = build_dependency_graph(operations)
ready = [op for op in operations if has_no_unscheduled_deps(op, deps)]
scheduled = []
live_values = set()  # Values currently live in DST registers

while ready:
  best_op = None
  best_cost_tuple = (-inf, -inf, -inf, -inf, -inf, inf, -inf)

  for op in ready:
    # Init priority: full-init ops (bcast, reduce) must be scheduled
    # before any other init type. See "Init Ordering" below.
    init_priority = 1 if is_full_init(op) else 0

    # Compute cost tuple (lexicographic comparison)
    release_pot = count_last_uses(op, remaining_ops)  # Frees DST registers
    fire_pot = count_newly_ready_ops(op, deps)        # Enables more ops
    primary_aff = sum_reuse_with_live(op, live_values) # Reuse with live values
    secondary_aff = indirect_reuse_score(op, live_values)
    non_live_aff_penalty = -count_reuse_with_nonlive(op, remaining_ops)

    # Special heuristic: penalize in-place ops on multi-consumer values
    # (schedule in-place ops last to avoid copies)
    if has_trait(op, TTLInPlaceOpTrait) and op.dst_operand.num_consumers > 1:
      non_live_aff_penalty -= 100  # Large penalty

    cost_tuple = (init_priority, release_pot, fire_pot, primary_aff,
                  secondary_aff, non_live_aff_penalty,
                  critical_path_priority(op))

    if cost_tuple > best_cost_tuple:
      best_cost_tuple = cost_tuple
      best_op = op

  # Schedule best operation
  scheduled.append(best_op)

  # Update live values
  for operand in best_op.operands:
    if is_last_use(operand, best_op, remaining_ops):
      live_values.remove(operand)
  live_values.add(best_op.result)

  # Update ready list
  ready = update_ready_list(deps, scheduled)

return scheduled  # New operation order
```

**Cost Tuple Metrics** (lexicographically sorted, highest priority first):

1. **Init Priority**: 1 for full-init operations (broadcast, reduce), 0
   otherwise. This is the highest-priority term: full-init ops are
   always scheduled before non-full-init ops when both are ready. See
   Init Ordering below.

2. **Release Potential (Rpot)**: Number of values at their last use in this operation. Higher is better (frees DST registers).

3. **Fire Potential (Fpot)**: Number of dependent operations that become schedulable after executing this operation. Higher is better (enables more scheduling flexibility).

4. **Primary Affinity (Paff)**: Sum of reuse strength with already-live values. Operations that use values already in DST registers have high affinity. Higher is better (locality).

5. **Secondary Affinity (Saff)**: Indirect reuse - operations sharing common inputs with already-scheduled operations. Higher is better.

6. **Non-Live Affinity Penalty (-Nnpaff)**: Negative of the number of unscheduled operations this op shares inputs with. More negative is worse (will extend live ranges).
   - **Special case**: If this operation has `TTLInPlaceOpTrait` and operates on a multi-consumer value, add a large penalty to schedule it last.

7. **Critical Path Priority (Pl)**: Distance from this operation to the yield or store (end of block). Operations on the critical path receive higher priority.

**Init Ordering**: Broadcast (`unary_bcast_init`) and reduce
(`reduce_init`) perform full inits that configure UNPACK + MATH + PACK.
A subsequent short init (e.g., `exp_tile_init`) reconfigures only MATH,
leaving the PACK configuration intact. If a full-init operation were
scheduled after a short-init operation, the full init would clobber the
prior PACK configuration, requiring a re-init. Scheduling full-init
operations first avoids this: the full init sets up PACK once, and
subsequent short inits override only MATH/UNPACK as needed.

If multiple full-init operations appear in the same block (e.g., both
broadcast and reduce), each will clobber the other's PACK
configuration. The scheduler orders them by data dependencies; if they
are independent, their relative order is determined by the remaining
cost tuple metrics (Rpot, Fpot, etc.). The lowering pass must insert
the appropriate re-init between them.

**Example**:
```mlir
%0 = add_tiles(%in0, %in1) {fpu}  // FPU binary -> DST
%1 = tile_exp(%in2)                 // Separate SFPU chain
%2 = add_binary_tile(%0, %1)        // SFPU binary consumer (NOT in-place)
%3 = tile_abs(%0)                   // Unary consumer (in-place)

# Both add_binary_tile and abs are ready after their dependencies resolve.
# Neither is a full-init op, so init_priority = 0 for both.
# Cost tuples (init_priority, Rpot, Fpot, Paff, Saff, -Nnpaff, Pl):
abs:             <0, 0, 0, 1, 0, -100, ...>  # Large penalty for in-place on multi-consumer
add_binary_tile: <0, 0, 0, 1, 0, -1, ...>    # No penalty (SFPU binary, not in-place)

# add_binary_tile has higher cost tuple -> scheduled first
# Result: abs becomes last consumer -> no copy needed!
```

**When to Apply**: This phase is optional but recommended when the `ttl.compute` block contains:
- Values with multiple consumers (especially mix of in-place and non-in-place)
- Many independent operations (flexibility to reorder)
- High register pressure (close to capacity limits)

**Implementation Approach**:

The SSA dependency graph is constructed by walking the def-use chains
of operations within the `ttl.compute` block body. Each operation
becomes a node; an edge from A to B exists when B uses a result of A.
Block arguments (the `^bb0(...)` parameters) have no defining operation
and impose no dependencies — operations that consume only block
arguments are immediately ready.

MLIR provides the necessary infrastructure for this:

- **Def-use iteration**: `Value::getUsers()` returns all operations
  consuming a value; `Value::getDefiningOp()` returns the producing
  operation (or `nullptr` for block arguments). Together these are
  sufficient to build the full dependency DAG for a single block. See
  `mlir/test/lib/IR/TestPrintDefUse.cpp` for a complete walk example.

- **Topological ordering**: `mlir/Analysis/TopologicalSortUtils.h`
  provides `computeTopologicalSorting`, which maintains a set of
  unscheduled operations and repeatedly selects ready operations whose
  operands are all defined by already-scheduled operations or block
  arguments. The `isOperandReady` callback enables custom readiness
  logic (e.g., treating CB-only values as always ready).

- **Slice analysis**: `mlir/Analysis/SliceAnalysis.h` provides
  `getForwardSlice` (walks uses from a definition) and
  `getBackwardSlice` (walks definitions from a use). These are useful
  for computing the `critical_path_priority` metric: a backward slice
  from the yield gives the critical-path length for each operation.

The greedy list scheduler follows the pattern used by LLVM's
`ScheduleDAGInstrs` (`llvm/lib/CodeGen/ScheduleDAGInstrs.cpp`):

1. **Build the DAG**: Walk operations in the block. For each operation,
   iterate over its operands via `op->getOperands()`. For each operand,
   call `getDefiningOp()` to find the producer. If the producer is
   non-null and lives in the same block, add an edge producer -> op. If
   null (block argument) or defined outside the block, no edge is
   needed.

2. **Initialize the ready list**: Operations with no in-block
   predecessors (all operands are block arguments or constants) are
   initially ready. This is analogous to LLVM's
   `SUnit::isTopReady()` (`NumPredsLeft == 0`).

3. **Greedy selection loop**: Pop the best operation from the ready
   list using the cost tuple. After scheduling an operation, decrement
   the unscheduled-predecessor count for each successor. When a
   successor's count reaches zero, add it to the ready list. This is
   the same `ReleaseSuccessors` pattern used in
   `llvm/lib/CodeGen/PostRASchedulerList.cpp`.

4. **Update live values**: After scheduling, add the operation's
   results to the live set and remove any operands at their last use.
   These live-value counts feed the cost tuple metrics (Rpot, Paff).

Unlike LLVM's machine-level scheduler, this operates at the MLIR
operation level within a single basic block, so there are no anti- or
output-dependencies (SSA guarantees single definitions), no memory
aliasing concerns (tile operations are pure), and — initially — no
latency modeling (all operations are treated as unit-cost). The
dependency graph reduces to a pure data-flow DAG derived from SSA
def-use chains. A future extension should incorporate per-operation
latencies derived from microbenchmark data into the cost tuple,
enabling latency-aware scheduling analogous to LLVM's
`computeOperandLatency` in `ScheduleDAGInstrs`.

### Phase 1: Insert Copy Operations

Normalize the IR by inserting explicit copy operations where values have multiple consumers.

```
for each tile value v:
  if v has multiple consumers (users.size() > 1):
    # Sort consumers by block order (operation position)
    consumers = sorted(v.users, key=lambda op: op.position_in_block)

    # Check if ANY consumer overwrites its DST input in-place.
    has_inplace_consumer = any(
      has_trait(c, TTLInPlaceOpTrait) for c in consumers
    )

    if has_inplace_consumer:
      # CRITICAL: In-place ops overwrite their DST input!
      # Strategy: Keep the LAST consumer using original value,
      # insert copies for all earlier consumers
      for i in range(len(consumers) - 1):
        insert before consumers[i]:
          v_copy_i = ttl.copy_dest_values(v, new_dst_reg)  # src -> dst (TTL convention)
        replace consumers[i]'s use of v with v_copy_i
    # else: All consumers lack TTLInPlaceOpTrait -> no copies needed
```

**Rationale**:
- Operations with `TTLInPlaceOpTrait` (unary, `binary_dest_reuse_tiles`) overwrite their DST input
- Operations without `TTLInPlaceOpTrait` (SFPU binary, FPU binary) write to a fresh output DST or do not use DST inputs at all
- If a value has multiple consumers and **any** consumer has `TTLInPlaceOpTrait`, copies are needed for earlier consumers
- If **all** consumers lack the trait, no copies needed

By copying for all consumers except the last one (when in-place consumers exist), the pass guarantees:
1. The last consumer can safely use (and potentially overwrite) the original value
2. All earlier consumers work on independent copies
3. Non-in-place multi-consumer values avoid unnecessary copies

**Note on Implementation**: Phase 1 inserts **DST-to-DST copy operations**, which are distinct from the CB-to-DST copies used to load inputs:

**API Difference**:
- **CB-to-DST copy** (for loading inputs from circular buffers):
  - `copy_tile(cb_id, tile_index, dst_index)` -> unpacks from CB to DST
- **DST-to-DST copy** (inserted in Phase 1 for multi-consumer values):
  - **TTL level**: `copy_dest_values(src_index, dst_index)` -> copies from src to dst
  - **Note**: TTL uses natural src->dst order, which is then lowered to tt-metal's `copy_dest_values(dst, src)` (reversed)

**Lowering Path for DST-to-DST Copies**:
- **TTL level**: `ttl.copy_dest_values(src, dst)`
- **TTKernel level**: `ttkernel.copy_dest_values(dst, src)` (arguments reversed from TTL)
- **tt-metal level**: `llk_math_eltwise_binary_sfpu_copy_dest_values(dst, src)` (SFPU binary operation)

**Note on Argument Order Convention**:
Throughout this document, pseudocode and MLIR examples use **TTL convention** for `copy_dest_values(source, destination)` - natural left-to-right data flow. When lowered to tt-metal, arguments are reversed to `copy_dest_values(destination, source)`. All generated C++ code examples follow tt-metal convention with destination first.

**Note on Initialization**: The lowering to tt-metal requires `copy_dest_values_init()` before first use. This is a lowering detail handled by the backend - not shown in examples for brevity.

**Example transformation** (see Example 4):
```mlir
%0 = mul_tiles(%in0, %in1)  {fpu}   // FPU binary -> DST
%0_copy_0 = copy_dest_values(%0)     // Copy for first consumer
%1 = tile_abs(%0_copy_0)             // Uses copy (safe to overwrite)
%2 = tile_exp(%0)                    // Last consumer uses original
```

### Phase 2: Build Live Intervals with In-Place Merging

Build live intervals for each tile value. Following Wimmer & Franz
[CGO'10], we process operations in linear order (forward pass). For
in-place operations (those with `TTLInPlaceOpTrait`), merge the input
and output intervals since they must share the same DST register
(hardware constraint).

This phase also identifies **CB-only values**: block arguments whose
only consumers read them from a CB rather than from DST. These values
never enter DST and do not receive live intervals.

```
intervals = {}
merged_sets = {}  # Union-find structure to track merged values
cb_only_values = set()  # Values that stay in CBs, never enter DST

# Number operations in linear order
for i, op in enumerate(operations):
  op.index = i

# Identify CB-only block arguments.
# A block argument is CB-only if ALL of its consumers read it from a CB
# rather than from DST. The check uses two traits:
#   - TTLCBInputTileOpTrait: all operands read from CB (FPU binary,
#     copy_tile, broadcast, reduce, matmul)
#   - TTLDSTInputsTrait + TTLCBInputTileOpTrait: mixed-input op
#     (dest_reuse) -- only the CB operand stays in CB
# For CB-input ops, the block arg stays in CB; the operation's *result*
# is what enters DST (and gets its own interval).
for each block_arg:
  all_cb = True
  for each consumer of block_arg:
    if has_trait(consumer, TTLCBInputTileOpTrait):
      if has_trait(consumer, TTLDSTInputsTrait):
        # Mixed-input op (dest_reuse): only the CB operand stays in CB
        if block_arg != consumer.cb_operand:
          all_cb = False
          break
      # else: all operands read from CB (FPU binary, copy_tile, bcast,
      #        reduce, matmul) -- block_arg stays in CB
    else:
      all_cb = False
      break
  if all_cb:
    cb_only_values.add(block_arg)

# Process operations in forward order to build intervals
for each operation op:
  # For each input operand, extend its interval to include this use
  for each input_val in op.inputs:
    if input_val in cb_only_values:
      continue  # CB-only value -- no DST interval

    if input_val not in intervals:
      # Block argument that needs DST -- starts at block entry (index 0)
      intervals[input_val] = Interval(0, op.index)
    else:
      # Extend existing interval to cover this use
      intervals[input_val].end = max(intervals[input_val].end, op.index)

  # For each output operand, create new interval starting at this definition
  for each output_val in op.results:
    intervals[output_val] = Interval(op.index, op.index)

# Extend intervals for values used in yield
for each yielded_val in yield_operands:
  intervals[yielded_val].end = max(intervals[yielded_val].end, yield_op.index)

# Merge intervals for in-place ops (those with TTLInPlaceOpTrait).
# In-place ops overwrite the DST input, so input and output MUST share
# the same DST register. This merge is unconditional because it
# reflects a hardware constraint: the operation physically overwrites
# the input slot.
#
# Note: Operations without TTLInPlaceOpTrait write to a FRESH output
# DST slot and do NOT merge with their inputs.
for each op where has_trait(op, TTLInPlaceOpTrait):
  input_val = op.dst_operand  # the DST input (not CB operand for dest_reuse)
  output_val = op.result

  # Unconditional merge: hardware requires same DST register
  merged_interval = Interval(
    start = min(intervals[input_val].start, intervals[output_val].start),
    end = max(intervals[input_val].end, intervals[output_val].end)
  )
  intervals[input_val] = merged_interval
  intervals[output_val] = merged_interval

  # Track merged values using union-find
  union(merged_sets, input_val, output_val)
```

**Summary**: Merged intervals represent values that MUST share the
same DST register. The union-find structure (`merged_sets`) tracks
equivalence classes of merged values. During allocation (Phases 3 & 4),
when we assign a register to one value in a merged set, we assign it to
ALL values in that set.

**CB-Only Values**: Block arguments identified as CB-only are excluded
from DST allocation entirely. This is the primary mechanism by which
CB-reading operations reduce `inputs_and_intermediates_footprint`. For
example, in `exp(a + b)`, both `a` and `b` are CB-only because their
sole consumer is an FPU `add_tiles` -- reducing the footprint from 2 to
0 compared to the SFPU-only path. Similarly, in `exp(a) * b` (Example
6), `a` is CB-only because its consumer is `copy_tile` (which reads
from CB and produces a new DST value), and `b` is CB-only because its
consumer is `dest_reuse` CB operand. Broadcast, reduce, and matmul work
the same way: the block arg stays in CB, and only the operation's result
enters DST. All CB-reading operations (copy_tile, tile_bcast,
tile_reduce, tile_matmul) carry `TTLCBInputTileOpTrait` in the
implementation, so the CB-only check is trait-based.

### Phase 3: Linear Scan Allocation for Inputs/Intermediates

Run linear scan register allocation for all values **except** those yielded to output block arguments. This phase allocates the input region DST[0..k-1] which will be reused across loop iterations.

```
active = []  # Currently live intervals (sorted by end position)
free_regs = [0, 1, 2, ..., capacity-1]
dst_assignment = {}
processed_merged_sets = set()  # Track which merged sets we've allocated

for interval in sorted(intervals, key=lambda i: i.start):
  # Skip outputs - they'll be allocated in Phase 4
  if interval.value is yielded to output:
    continue

  # Skip if this value's merged set has already been processed
  merged_set_id = find(merged_sets, interval.value)
  if merged_set_id in processed_merged_sets:
    continue  # Already allocated with its merged partners

  # Expire old intervals (following Wimmer & Franz algorithm)
  for active_interval in sorted(active, key=lambda i: i.end):
    if active_interval.end < interval.start:
      free_regs.append(dst_assignment[active_interval.value])
      active.remove(active_interval)

  # Allocate register
  if len(free_regs) > 0:
    reg = free_regs.pop(0)

    # Assign to ALL values in the merged set (unary chains share DST)
    all_merged_vals = get_all_values_in_merged_set(merged_sets, interval.value)
    for merged_val in all_merged_vals:
      dst_assignment[merged_val] = reg

    active.append(interval)
    processed_merged_sets.add(merged_set_id)
  else:
    # Spill (error for now, future work)
    error("insufficient DST registers")

inputs_and_intermediates_footprint = max(dst_assignment.values()) + 1
```

**Handling Merged Intervals**: When we encounter an interval that's part of a merged set (e.g., unary chain), we:
1. Check if any value in the set was already processed -> skip if yes
2. Allocate a single DST register for the ENTIRE merged set
3. Assign that register to ALL values in the set simultaneously
4. Mark the merged set as processed to avoid double allocation

Example: For `%0 = abs(%in); %1 = exp(%0)` merged into one interval, we allocate one DST register and assign it to both `%0` and `%1`.

### Phase 4: Linear Scan Allocation for Outputs

Run linear scan allocation again for values yielded to output block arguments, starting DST indices after the input/intermediate region. This phase allocates the output region DST[k..n] where outputs will be stored (one per unrolled iteration).

```
base_out_dst_index = inputs_and_intermediates_footprint
active = []
free_regs = [base_out_dst_index, base_out_dst_index+1, ..., capacity-1]
processed_merged_sets = set()  # Separate tracking for output region

for interval in sorted(intervals, key=lambda i: i.start):
  # Only process values yielded to output
  if interval.value is NOT yielded to output:
    continue

  # Skip if already allocated (part of merged set with intermediates)
  if interval.value in dst_assignment:
    continue  # Already assigned in Phase 3 as intermediate

  # Skip if this value's merged set has already been processed
  merged_set_id = find(merged_sets, interval.value)
  if merged_set_id in processed_merged_sets:
    continue

  # Expire old intervals (same as Phase 3)
  for active_interval in sorted(active, key=lambda i: i.end):
    if active_interval.end < interval.start:
      free_regs.append(dst_assignment[active_interval.value])
      active.remove(active_interval)

  # Allocate register in output region
  if len(free_regs) > 0:
    reg = free_regs.pop(0)

    # Assign to ALL values in the merged set
    all_merged_vals = get_all_values_in_merged_set(merged_sets, interval.value)
    for merged_val in all_merged_vals:
      # Only assign if not already assigned in Phase 3
      if merged_val not in dst_assignment:
        dst_assignment[merged_val] = reg

    active.append(interval)
    processed_merged_sets.add(merged_set_id)
  else:
    error("insufficient DST registers for outputs")
```

**Handling Cross-Region Merged Intervals**: Some merged sets span both regions. Example: `%0 = mul(...); %1 = abs(%0); yield %1`. Here `%0` and `%1` are merged, but only `%1` is yielded.

- **Phase 3** skips both (merged with yielded value)
- **Phase 4** processes `%1`, assigns it DST[k], and also assigns `%0` to DST[k]
- Result: Both intermediate `%0` and output `%1` share DST[k] in output region

**Result**: DST layout with separate regions:
- `DST[0..k-1]`: Inputs and intermediates (reused across loop iterations)
- `DST[k..n]`: Outputs (allocated sequentially, one per unrolled iteration)

## Worked Examples

### Example 1: Unary Operation

**Input MLIR**:
```mlir
^bb0(%in: tile, %out: tile):
  %0 = tile_abs(%in)
  linalg.yield %0
```

**Phase 1**: No multi-consumer values -> No copies

**Phase 2 (Build Intervals)**:
```
Intervals:
  %in: [0, 1]
  %0:  [1, 2]

Merge for unary (abs):
  merged_interval = [min(0,1), max(1,2)] = [0, 2]
  %in: [0, 2]
  %0:  [0, 2]
  mark_as_merged(%in, %0)
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %in [0,2]: (skip, will be handled via merge in Phase 4)
Process %0 [0,2]: (yielded to output, skip)

inputs_and_intermediates_footprint = 0 (no assignments)
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 0
Process %0 [0,2]: allocate DST[0]
  %0 is merged with %in, so:
  dst_assignment[%0] = DST[0]
  dst_assignment[%in] = DST[0]  (merged)
```

**Final Assignment**:
```
Inputs/Intermediates: (none)

Outputs (DST[0]):
  %in -> DST[0]
  %0  -> DST[0]  (in-place)
```

**Generated Code**:
```cpp
tile_regs_acquire();
copy_tile(CB0, 0, DST[0]);
abs_tile(DST[0]);               // In-place
tile_regs_commit();
tile_regs_wait();
pack_tile(DST[0], CB_out, 0);
tile_regs_release();
```

---

### Example 2: Unary Chain (3 ops)

**Input MLIR**:
```mlir
^bb0(%in: tile, %out: tile):
  %0 = tile_abs(%in)
  %1 = tile_exp(%0)
  %2 = tile_relu(%1)
  linalg.yield %2
```

**Lifetime Interval Visualization (after unary merging)**:
```
Operation Timeline:    0      1      2      3      4
                       |      |      |      |      |
After merging (all unary ops share one DST):
  %in       [0────────────────────4]    █████████████████████  ┐
  %0        [1────────────────────4]         ████████████████  │ All merged
  %1        [2────────────────────4]              ███████████  │ into single
  %2        [3────────────────────4]                   ██████  ┘ DST register

DST Assignment:
  Inputs/Intermediates: (none)
  Outputs:              %in,%0,%1,%2 -> DST[0] (all share, in-place chain)
```

**Key**: Sequential unary chain - all operations share one DST register, each operating in-place.

**Phase 1**: No multi-consumer values -> No copies

**Phase 2 (Build Intervals)**:
```
Intervals before merging:
  %in: [0, 1]
  %0:  [1, 2]
  %1:  [2, 3]
  %2:  [3, 4]

Merge for abs (%in, %0):
  %in: [0, 2]
  %0:  [0, 2]

Merge for exp (%0, %1):
  %0: [0, 3]  (extends existing)
  %1: [0, 3]

Merge for relu (%1, %2):
  %1: [0, 4]  (extends existing)
  %2: [0, 4]

All values merged into single interval [0, 4]:
  %in, %0, %1, %2 all marked as merged
```

**Phase 3**: All values are merged with yielded output -> skip all

**Phase 4**:
```
base_out_dst_index = 0

Process %2 [0,4]: allocate DST[0]
  Merged with %in, %0, %1:
  dst_assignment[%in] = DST[0]
  dst_assignment[%0]  = DST[0]
  dst_assignment[%1]  = DST[0]
  dst_assignment[%2]  = DST[0]
```

**Final Assignment**:
```
Inputs/Intermediates: (none)

Outputs (DST[0]):
  %in -> DST[0]
  %0  -> DST[0]  (in-place)
  %1  -> DST[0]  (in-place)
  %2  -> DST[0]  (in-place)
```

**Generated Code**:
```cpp
tile_regs_acquire();
copy_tile(CB0, 0, DST[0]);
abs_tile(DST[0]);
exp_tile(DST[0]);
relu_tile(DST[0]);
tile_regs_commit();
tile_regs_wait();
pack_tile(DST[0], CB_out, 0);
tile_regs_release();
```

---

### Example 3: FPU Binary + SFPU Chain with Unrolling (1x4, f32)

This example demonstrates FPU-aware allocation for `exp(a + b)`. The
FPU `add_tiles` reads both operands from CBs, so `%in0` and `%in1`
never enter DST. Compare with Example 5, where both operands are DST
intermediates requiring SFPU binary and 2 DST input slots.

**Input MLIR**:
```mlir
^bb0(%in0: tile, %in1: tile, %out: tile):
  %0 = tile_add(%in0, %in1)  {execution_target = "fpu"}
  %1 = tile_exp(%0)
  linalg.yield %1
```

**Output shape**: 1x4 = 4 tiles, f32 (capacity = 4)

**Phase 1 (Copy Insertion)**: No multi-consumer values -> no copies.

**Phase 2 (Build Intervals)**:
```
CB-only analysis:
  %in0: only consumer is FPU binary tile_add -> CB-only
  %in1: only consumer is FPU binary tile_add -> CB-only
  cb_only_values = {%in0, %in1}

Intervals (excluding CB-only values):
  %0: [1, 2]  (def at tile_add, used by tile_exp)
  %1: [2, 3]  (def at tile_exp, used by yield)

  Note: %in0 and %in1 have NO intervals -- they stay in CBs.

Merge for exp (%0, %1):
  %0: [1, 3]  (merged)
  %1: [1, 3]  (merged)
  mark_as_merged(%0, %1)
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %0 [1,3]: (skip - merged with %1 which is yielded to output)
Process %1 [1,3]: (yielded to output, skip)

inputs_and_intermediates_footprint = 0
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 0

Process %1 [1,3]: allocate DST[0]
  Merged with %0 -> dst_assignment[%0] = DST[0], dst_assignment[%1] = DST[0]
```

**Final Assignment**:
```
CB-only (no DST): %in0, %in1

Inputs/Intermediates: (none)

Outputs (DST[0]):
  %0 -> DST[0]
  %1 -> DST[0]  (in-place with %0)
```

**Unroll Factor Calculation**:
```
inputs_and_intermediates_footprint = 0  (FPU binary needs no DST inputs)
numOutputs = 1 (%out)
available_for_outputs = capacity - 0 = 4
unrollFactor = min(4 / 1, 4) = 4  (fully unrolled!)
```

**Comparison with SFPU-only path**: If `tile_add` were SFPU binary
instead of FPU, `%in0` and `%in1` would need DST slots via `copy_tile`,
giving `inputs_and_intermediates_footprint = 2` and
`available_for_outputs = 4 - 2 = 2`. The FPU path doubles the unroll
factor for this chain.

**Generated Code (fully unrolled, f32, 4 tiles)**:
```cpp
// No copy_tile -- FPU reads directly from CBs
add_tiles_init(cb_in0, cb_in1);
tile_regs_acquire();

// All 4 FPU adds
add_tiles(cb_in0, cb_in1, 0, 0, 0);  // CB -> FPU -> DST[0]
add_tiles(cb_in0, cb_in1, 1, 1, 1);  // CB -> FPU -> DST[1]
add_tiles(cb_in0, cb_in1, 2, 2, 2);  // CB -> FPU -> DST[2]
add_tiles(cb_in0, cb_in1, 3, 3, 3);  // CB -> FPU -> DST[3]

// All 4 SFPU exps (in-place on DST)
exp_tile_init();
exp_tile(0);
exp_tile(1);
exp_tile(2);
exp_tile(3);

tile_regs_commit();
tile_regs_wait();
pack_tile_block(0, cb_out, 4);       // DST[0..3] -> CB
tile_regs_release();
```

**Summary**: With `inputs_and_intermediates_footprint = 0`, all DST
registers are available for outputs. The FPU binary operation requires
no DST input slots, and the SFPU exp consumes the result in-place via
merging.

---

### Example 4: FPU Binary with Multi-Consumer SFPU (Copy Insertion)

This example demonstrates copy insertion in an FPU-aware context:
`abs(a * b)` and `exp(a * b)`, where `mul_tiles` is FPU binary (both
operands from CBs) and both consumers are SFPU unary (in-place on DST).

**Input MLIR**:
```mlir
^bb0(%in0: tile, %in1: tile, %out0: tile, %out1: tile):
  %0 = mul_tiles(%in0, %in1)  {execution_target = "fpu"}
  %1 = tile_abs(%0)
  %2 = tile_exp(%0)
  linalg.yield %1, %2
```

**Phase 1 (Copy Insertion)**:
```
%0 has 2 consumers (abs at position 0, exp at position 1)
Both are in-place -> copy for all except last:

Result:
  %0 = mul_tiles(%in0, %in1)  {execution_target = "fpu"}
  %0_copy_0 = copy_dest_values(%0)   // Copy for first consumer (abs)
  %1 = tile_abs(%0_copy_0)            // Uses copy
  %2 = tile_exp(%0)                   // Last consumer uses original
```

**Phase 2 (Build Intervals)**:
```
CB-only analysis:
  %in0: only consumer is FPU binary mul_tiles -> CB-only
  %in1: only consumer is FPU binary mul_tiles -> CB-only
  cb_only_values = {%in0, %in1}

Intervals (excluding CB-only values):
  %0:        [1, 3]  (def at mul_tiles, used by exp at position 3)
  %0_copy_0: [2, 2]  (def at copy, immediately used by abs)
  %1:        [2, 3]  (def at abs, used by yield)
  %2:        [3, 4]  (def at exp, used by yield)

Merge for abs (%0_copy_0, %1):
  %0_copy_0: [2, 3]
  %1:        [2, 3]

Merge for exp (%0, %2):
  %0: [1, 4]
  %2: [1, 4]
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %0 [1,4]: (skip - merged with %2, yielded)
Process %0_copy_0 [2,3]: (skip - merged with %1, yielded)

inputs_and_intermediates_footprint = 0
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 0

Process %0 [1,4]: allocate DST[0]
  Merged with %2 -> dst_assignment[%0] = DST[0], dst_assignment[%2] = DST[0]

Process %1 [2,3]: allocate DST[1]
  Merged with %0_copy_0 -> dst_assignment[%0_copy_0] = DST[1], dst_assignment[%1] = DST[1]
```

**Final Assignment**:
```
CB-only (no DST): %in0, %in1

Inputs/Intermediates: (none)

Outputs (DST[0-1]):
  %0        -> DST[0]
  %2        -> DST[0]  (in-place with %0)
  %0_copy_0 -> DST[1]
  %1        -> DST[1]  (in-place with copy)
```

**Generated Code**:
```cpp
mul_tiles_init(cb_in0, cb_in1);
tile_regs_acquire();
mul_tiles(cb_in0, cb_in1, 0, 0, 0);      // CB -> FPU -> DST[0]
copy_dest_values(DST[0], DST[1]);        // Copy for abs (first consumer)
abs_tile_init();
abs_tile(DST[1]);                        // In-place on copy
exp_tile_init();
exp_tile(DST[0]);                        // In-place on original (last consumer)
tile_regs_commit();
tile_regs_wait();
pack_tile(DST[1], cb_out0, 0);          // Pack abs result
pack_tile(DST[0], cb_out1, 0);          // Pack exp result
tile_regs_release();
```

**Summary**: Because `mul_tiles` is FPU binary, both inputs stay in
CBs and `inputs_and_intermediates_footprint = 0`. The same computation
with SFPU binary mul would require both inputs loaded to DST via
`copy_tile`, giving `inputs_and_intermediates_footprint = 2` and
requiring 4 total DST slots instead of 2.

---

### Example 5: SFPU Binary — Both Operands from DST

This example demonstrates the case requiring SFPU binary:
`exp(a) + exp(b)`, where `+` must be SFPU binary because both operands
are DST intermediates (results of SFPU exp). When both operands are
already in DST from prior computations, SFPU binary is the only option.
Compare with Example 3, where the operands come from CBs and FPU binary
is used instead.

**Input MLIR**:
```mlir
^bb0(%in_a: tile, %in_b: tile, %out: tile):
  %0 = tile_exp(%in_a)
  %1 = tile_exp(%in_b)
  %2 = add_binary_tile(%0, %1)   // SFPU binary: both operands in DST
  linalg.yield %2
```

**Output shape**: 1x4 = 4 tiles, f32 (capacity = 4)

**Phase 1**: No multi-consumer values -> No copies

**Phase 2 (Build Intervals)**:
```
CB-only analysis:
  %in_a: consumer is tile_exp (SFPU unary) -> NOT CB-only
  %in_b: consumer is tile_exp (SFPU unary) -> NOT CB-only
  cb_only_values = {}

Intervals:
  %in_a: [0, 0]  (block arg, used at op 0)
  %in_b: [0, 1]  (block arg, used at op 1)
  %0:    [0, 2]  (def at op 0, used at op 2)
  %1:    [1, 2]  (def at op 1, used at op 2)
  %2:    [2, 3]  (def at op 2, used at yield)

Merge for exp (%in_a, %0):
  %in_a: [0, 2]
  %0:    [0, 2]
  mark_as_merged(%in_a, %0)

Merge for exp (%in_b, %1):
  %in_b: [0, 2]
  %1:    [0, 2]
  mark_as_merged(%in_b, %1)

No merge for add_binary_tile (SFPU binary writes to fresh DST slot)
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %in_a/%0 [0,2]: NOT yielded -> allocate DST[0]
  Merged set {%in_a, %0} -> DST[0]
Process %in_b/%1 [0,2]: NOT yielded -> allocate DST[1]
  Merged set {%in_b, %1} -> DST[1]
Process %2 [2,3]: (yielded, skip)

inputs_and_intermediates_footprint = 2
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 2

Process %2 [2,3]: allocate DST[2]
  At start=2: expire %in_a/%0, %in_b/%1
  Allocate DST[2]
```

**Final Assignment**:
```
Inputs/Intermediates (DST[0-1]):
  %in_a -> DST[0]
  %0    -> DST[0]  (exp in-place)
  %in_b -> DST[1]
  %1    -> DST[1]  (exp in-place)

Outputs (DST[2]):
  %2 -> DST[2]
```

**Unroll Factor Calculation**:
```
inputs_and_intermediates_footprint = 2  (both inputs need DST for SFPU)
numOutputs = 1 (%out)
available_for_outputs = capacity - 2 = 4 - 2 = 2
unrollFactor = min(2 / 1, 4) = 2
```

**Generated Code**:
```cpp
copy_tile_to_dst_init_short(cb_a);
tile_regs_acquire();
copy_tile(cb_a, 0, DST[0]);
copy_tile(cb_b, 0, DST[1]);
exp_tile_init();
exp_tile(DST[0]);
exp_tile(DST[1]);
add_binary_tile_init();
add_binary_tile(DST[0], DST[1], DST[2]);
tile_regs_commit();
tile_regs_wait();
pack_tile(DST[2], cb_out, 0);
tile_regs_release();
```

**Summary**: Both inputs require DST slots because the SFPU unary
(exp) operates in-place on DST. The SFPU binary add then consumes both
DST intermediates and writes to a fresh output slot.
`inputs_and_intermediates_footprint = 2`, limiting the unroll factor to
2 (vs. 4 in Example 3 where FPU binary avoids DST input slots entirely).

---

### Example 6: `binary_dest_reuse_tiles` Chain with Unrolling (1x4, f32)

This example demonstrates allocation for `exp(a) * b`, where the SFPU
chain result feeds a `binary_dest_reuse_tiles` multiplication with a CB
operand. The `dest_reuse` overwrites the DST intermediate in-place.

Both block arguments are CB-only: `%in_a` because its consumer is
`copy_tile` (which reads from CB), and `%in_b` because its consumer is
`dest_reuse` CB operand. The `copy_tile` result `%0` is the value that
enters DST — each unrolled iteration targets a different output DST
slot directly.

**Input MLIR**:
```mlir
^bb0(%in_a: tile, %in_b: tile, %out: tile):
  %0 = copy_tile(%in_a)                                        // CB -> DST
  %1 = tile_exp(%0)                                             // SFPU in-place
  %2 = tile_mul(%1, %in_b) {execution_target = "dest_reuse"}   // DST * CB -> DST
  linalg.yield %2
```

**Output shape**: 1x4 = 4 tiles, f32 (capacity = 4)

**Phase 1 (Copy Insertion)**: No multi-consumer values -> no copies.

**Phase 2 (Build Intervals)**:
```
CB-only analysis:
  %in_a: consumer is copy_tile (reads from CB) -> CB-only
  %in_b: consumer is dest_reuse CB operand -> CB-only
  cb_only_values = {%in_a, %in_b}

Intervals (excluding CB-only values):
  %0: [1, 2]  (def at copy_tile, used by tile_exp)
  %1: [2, 3]  (def at tile_exp, used by tile_mul dest_reuse)
  %2: [3, 4]  (def at tile_mul, used by yield)

  Note: %in_a and %in_b have NO intervals -- they stay in CBs.
  The copy_tile result %0 is the first value that enters DST.

Merge for exp (%0, %1):
  %0: [1, 3]
  %1: [1, 3]
  mark_as_merged(%0, %1)

Merge for dest_reuse (%1, %2):
  %1: [1, 4]  (extends existing merged interval)
  %2: [1, 4]
  mark_as_merged(%1, %2)

All three merge into one set: {%0, %1, %2} with interval [1, 4]
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %0 [1,4]: (skip - merged with %2 which is yielded)
Process %1 [1,4]: (skip - same merged set)
Process %2 [1,4]: (yielded, skip)

inputs_and_intermediates_footprint = 0
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 0

Process %2 [1,4]: allocate DST[0]
  Merged with %0, %1:
  dst_assignment[%0] = DST[0]
  dst_assignment[%1] = DST[0]
  dst_assignment[%2] = DST[0]
```

**Final Assignment**:
```
CB-only (no DST): %in_a, %in_b

Inputs/Intermediates: (none)

Outputs (DST[0]):
  %0 -> DST[0]  (copy_tile result)
  %1 -> DST[0]  (exp in-place)
  %2 -> DST[0]  (dest_reuse in-place)
```

**Unroll Factor Calculation**:
```
inputs_and_intermediates_footprint = 0  (all inputs CB-only)
numOutputs = 1 (%out)
available_for_outputs = capacity - 0 = 4
unrollFactor = min(4 / 1, 4) = 4  (fully unrolled!)
```

**Generated Code (fully unrolled, f32, 4 tiles)**:
```cpp
copy_tile_to_dst_init_short(cb_a);
tile_regs_acquire();

// Iteration 0: copy -> exp -> dest_reuse (all on DST[0])
copy_tile(cb_a, 0, 0);                                          // CB_a -> DST[0]
exp_tile_init();
exp_tile(0);                                                      // DST[0] in-place
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_b, 0, 0);     // DST[0] * CB_b -> DST[0]

// Iteration 1 (all on DST[1])
copy_tile_to_dst_init_short(cb_a);
copy_tile(cb_a, 1, 1);
exp_tile_init();
exp_tile(1);
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_b, 1, 1);

// Iteration 2 (all on DST[2])
copy_tile_to_dst_init_short(cb_a);
copy_tile(cb_a, 2, 2);
exp_tile_init();
exp_tile(2);
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_b, 2, 2);

// Iteration 3 (all on DST[3])
copy_tile_to_dst_init_short(cb_a);
copy_tile(cb_a, 3, 3);
exp_tile_init();
exp_tile(3);
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_b, 3, 3);

tile_regs_commit();
tile_regs_wait();
pack_tile_block(0, cb_out, 4);                                   // DST[0..3] -> CB
tile_regs_release();
```

**Summary**: Both inputs are CB-only: `copy_tile` reads `%in_a`
from CB (the block arg never enters DST), and `dest_reuse` reads
`%in_b` from CB. The entire chain `{%0, %1, %2}` shares a single DST
slot, and `inputs_and_intermediates_footprint = 0`. Each unrolled
iteration targets a different output DST slot — `copy_tile` writes
directly to DST[i], and the subsequent in-place chain stays there.

---

### Example 7: Mixed Chain — FPU + Unary + dest_reuse with Unrolling (1x4, f32)

This example demonstrates the combination of all three non-SFPU-binary
categories in a single chain: `abs(a + b) * c`. The FPU `add_tiles`
reads from CBs, the SFPU `abs` operates in-place on DST, and
`binary_dest_reuse_tiles` overwrites the DST intermediate with the
product against a CB operand. All three inputs are CB-only.

**Input MLIR**:
```mlir
^bb0(%in_a: tile, %in_b: tile, %in_c: tile, %out: tile):
  %0 = add_tiles(%in_a, %in_b)    {execution_target = "fpu"}          // FPU binary
  %1 = tile_abs(%0)                                                     // SFPU unary in-place
  %2 = mul_tiles(%1, %in_c)       {execution_target = "dest_reuse"}   // dest_reuse
  linalg.yield %2
```

**Output shape**: 1x4 = 4 tiles, f32 (capacity = 4)

**Phase 1 (Copy Insertion)**: No multi-consumer values -> no copies.

**Phase 2 (Build Intervals)**:
```
CB-only analysis:
  %in_a: only consumer is FPU binary add_tiles -> CB-only
  %in_b: only consumer is FPU binary add_tiles -> CB-only
  %in_c: only consumer is dest_reuse CB operand -> CB-only
  cb_only_values = {%in_a, %in_b, %in_c}

Intervals (excluding CB-only values):
  %0: [0, 1]  (def at add_tiles, used by tile_abs)
  %1: [1, 2]  (def at tile_abs, used by mul_tiles dest_reuse)
  %2: [2, 3]  (def at mul_tiles, used by yield)

Merge for abs (%0, %1):
  %0: [0, 2]
  %1: [0, 2]
  mark_as_merged(%0, %1)

Merge for dest_reuse (%1, %2):
  %1: [0, 3]  (extends existing merged interval)
  %2: [0, 3]
  mark_as_merged(%1, %2)

All three merge into one set: {%0, %1, %2} with interval [0, 3]
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %0/%1/%2 [0,3]: (skip - merged with %2 which is yielded)

inputs_and_intermediates_footprint = 0
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 0

Process %2 [0,3]: allocate DST[0]
  Merged with %0, %1:
  dst_assignment[%0] = DST[0]
  dst_assignment[%1] = DST[0]
  dst_assignment[%2] = DST[0]
```

**Final Assignment**:
```
CB-only (no DST): %in_a, %in_b, %in_c

Inputs/Intermediates: (none)

Outputs (DST[0]):
  %0 -> DST[0]  (FPU binary result)
  %1 -> DST[0]  (abs in-place)
  %2 -> DST[0]  (dest_reuse in-place)
```

**Unroll Factor Calculation**:
```
inputs_and_intermediates_footprint = 0  (all inputs CB-only)
numOutputs = 1 (%out)
available_for_outputs = capacity - 0 = 4
unrollFactor = min(4 / 1, 4) = 4  (fully unrolled!)
```

**Generated Code (fully unrolled, f32, 4 tiles)**:
```cpp
add_tiles_init(cb_a, cb_b);
tile_regs_acquire();

// Iteration 0: FPU add -> abs -> dest_reuse mul
add_tiles(cb_a, cb_b, 0, 0, 0);                                // CB -> FPU -> DST[0]
abs_tile_init();
abs_tile(0);                                                     // DST[0] in-place
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_c, 0, 0);    // DST[0] * CB_c -> DST[0]

// Iteration 1
add_tiles_init(cb_a, cb_b);
add_tiles(cb_a, cb_b, 1, 1, 1);                                // CB -> FPU -> DST[1]
abs_tile_init();
abs_tile(1);
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_c, 1, 1);

// Iteration 2
add_tiles_init(cb_a, cb_b);
add_tiles(cb_a, cb_b, 2, 2, 2);                                // CB -> FPU -> DST[2]
abs_tile_init();
abs_tile(2);
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_c, 2, 2);

// Iteration 3
add_tiles_init(cb_a, cb_b);
add_tiles(cb_a, cb_b, 3, 3, 3);                                // CB -> FPU -> DST[3]
abs_tile_init();
abs_tile(3);
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_c, 3, 3);

tile_regs_commit();
tile_regs_wait();
pack_tile_block(0, cb_out, 4);                                  // DST[0..3] -> CB
tile_regs_release();
```

**Summary**: All three inputs are CB-only: `%in_a` and `%in_b` via
FPU binary, `%in_c` via dest_reuse CB operand. The entire computation
chain `{%0, %1, %2}` merges into a single DST slot.
`inputs_and_intermediates_footprint = 0` means all 4 DST registers
(f32 capacity) are available for outputs, enabling full unrolling.
This example combines all three non-SFPU-binary categories (FPU binary,
unary, dest_reuse) in a single chain, demonstrating maximal DST
utilization.

---

## DST Capacity

Examples 1 and 2 use **capacity = 8** (bf16 with double-buffering). Examples 3-7 use **capacity = 4** (f32 with double-buffering, `fp32_dest_acc_en=true`) to demonstrate FPU-aware allocation under tighter constraints.

**Actual Hardware Capacity** (from TT-Metal API documentation):
- **Physical DST size**: 16 tiles
- **With double-buffering**: 8 tiles (effective capacity during compute)
- **Data type dependency**:
  - f16/bf16: 16 tiles physical (8 with double-buffering)
  - f32: 8 tiles physical (4 with double-buffering when `fp32_dest_acc_en=true`)

The DST allocation pass should query the actual capacity from device configuration rather than using hardcoded values. The capacity depends on:
1. Data type (f16 vs f32)
2. Device configuration (`fp32_dest_acc_en`, `fullSyncEn`)
3. Double-buffering mode

**TODO (Issue #150)**: Compute capacity dynamically from datatype and device config.

## Pipeline Integration

The DST allocation pass runs in this order:

1. `ttl-assign-dst`: Assigns DST indices, adds `ttl.unroll_factor` attribute
2. `ttl-subblock-compute-for-dst`: Partitions `ttl.compute` into DST-sized subblocks via TilingInterface
3. `ttl-insert-tile-regs-sync`: Inserts DST lifecycle ops (acquire/commit/wait/release)
4. `ttl-lower-to-loops`: Converts `ttl.compute` to `scf.for` loops

See `DST_Utilization.md` for the full pipeline and target pipeline with
integrated unrolling, operation grouping, and subblock-level sync.

## Future Work

* **Implement Phase 0 (Operation Scheduling)**: The algorithm for register-pressure-aware scheduling is described in Phase 0 above. Implementing this as a pre-pass before DST allocation can reduce `copy_dest_values` operations for blocks with mixed in-place and non-in-place consumers (see the inline example in Phase 0). Key challenge: determining when to apply Phase 0 (cost/benefit analysis for small compute blocks).

* **Pack Multiple Contiguous Tiles**: Use `pack_tile_block` to pack multiple contiguous tiles in a single call. Requires analysis to determine when output tiles are contiguous in DST (e.g., `DST[2,3,4,5]` for a 2x2 block with row-major layout). Currently each tile is packed individually.

* **Register Spilling via Compute Splitting**: When `inputs_and_intermediates_footprint + required_outputs > capacity`, split the `ttl.compute` operation into multiple smaller compute operations that each fit within DST capacity.

  **Approach**: Instead of traditional register spilling (store to L1, reload later), split the computation:
  ```mlir
  // Original (exceeds DST capacity):
  ttl.compute {
    %0 = op1(...)
    %1 = op2(%0, ...)
    %2 = op3(%1, ...)  // Too many live values!
    %3 = op4(%2, ...)
    yield %3
  }

  // Split into two compute operations:
  ttl.compute {
    %0 = op1(...)
    %1 = op2(%0, ...)
    yield %1            // Intermediate result
  }
  // %1 stored to L1 via normal flow, DST registers released

  ttl.compute {
    %2 = op3(%1, ...)  // Reload from L1
    %3 = op4(%2, ...)
    yield %3
  }
  ```

  **Benefits over traditional spilling**:
  - Leverages existing L1 allocation and DMA infrastructure
  - Clear separation of concerns (each compute op has independent DST allocation)
  - Easier to reason about register pressure within each compute block
  - Natural integration with circular buffer management

  **Splitting Strategy**: Use interval splitting at "spill points" where live range pressure is highest. The Wimmer & Franz paper (Section 7) discusses interval splitting for linear scan allocation - the same principles apply here, but we split at compute operation boundaries rather than within a single operation.

  **Prior art in tt-metal**: The large-block matmul kernel
  (`bmm_large_block_zm.cpp`) implements this pattern manually: when the
  K dimension is split into multiple blocks, each block's partial sum
  is packed to an intermediate CB (`c_24`), then reloaded with
  `copy_tile` in the next block before accumulating further. The final
  block packs to the output CB. Compute splitting at the TTL level
  would automate this pack-reload pattern.

  **Challenge**: Determining optimal split points to minimize overhead while respecting DST capacity constraints. Could use dynamic programming or greedy heuristics based on interval pressure.

* **Deferred Input Loading + SFPU Binary Input Reuse**: Currently, all block arguments are treated as live from block entry (position 0), even if their first consumer appears later in the block. When block args are consumed by SFPU operations at different points, scheduling each `copy_tile` just before its first consumer shortens the interval and reduces peak liveness.

  **When it helps**: Chains of multiple SFPU binary operations consuming different block args at different points. Representative pattern: `(exp(a) + exp(b)) * exp(c)`:

  ```mlir
  %0 = tile_exp(%in_a)            // needs %in_a in DST at op 0
  %1 = tile_exp(%in_b)            // needs %in_b in DST at op 1
  %2 = add_binary_tile(%0, %1)    // SFPU binary at op 2
  %3 = tile_exp(%in_c)            // needs %in_c in DST at op 3 (LATE)
  %4 = mul_binary_tile(%2, %3)    // SFPU binary at op 4
  yield %4
  ```

  With eager loading (current): all three block args live from op 0. Peak = 4 at op 2 (three inputs + one SFPU binary result). At f32 capacity = 4, `inputs_and_intermediates_footprint = 4`, no room for outputs.

  With deferred loading of `%in_c` to just before op 3, plus SFPU binary writing to an expiring input slot (`add_binary(...) -> DST[0]` reusing `%in_a`'s slot):

  ```
  load %in_a -> DST[0], load %in_b -> DST[1]       // 2 live
  exp(DST[0]), exp(DST[1])                          // 2 live
  add_binary(DST[0], DST[1]) -> DST[0]              // reuse %in_a's slot; 2 live
  load %in_c -> DST[1]                               // deferred; 2 live
  exp(DST[1])                                        // 2 live
  mul_binary(DST[0], DST[1]) -> DST[2]              // output; 3 live
  ```

  Peak drops to 3. `inputs_and_intermediates_footprint = 2`, `available_for_outputs = 2`, `unrollFactor = 2` (vs. 0 without the optimization).

  **When it does NOT help**:
  - All SFPU inputs needed at the same operation (can't defer)
  - Only one SFPU binary (both inputs needed simultaneously)
  - Inputs are CB-only (FPU binary, dest_reuse CB operand, copy_tile, broadcast, reduce, matmul)
  - Pure unary chains (1 input at a time)

  **Implementation**: Deferred loading is a Phase 0 scheduling decision — reorder the implicit `copy_tile` operations so they appear just before their first consumer. SFPU binary input reuse is a Phase 3 allocation refinement — allow the allocator to assign an SFPU binary output to an expiring input slot when the input's interval ends at the same operation.

* **Adaptive Scheduling Heuristics**: Extend Phase 0 with adaptive cost tuple ordering based on computation characteristics (similar to LARS adaptivity in Section III-F of Rawat et al. SC'18). For computations with high intra-statement reuse, prioritize `Pl` (critical path) over `Paff` (affinity) to avoid excessive interleaving.

* **Implement Operation Category Traits**: The four orthogonal traits
  described in Operation Category Traits above (`TTLCBInputTileOpTrait`,
  `TTLDSTInputsTrait`, `TTLInPlaceOpTrait`, `TTLAccumulatingOpTrait`)
  need to be defined in TableGen and added to each operation definition.
  `TTLCBInputTileOpTrait` already exists; the remaining three must be
  implemented. See `DST_Utilization.md` (component 7) for how these
  traits integrate into the DST utilization pipeline.
