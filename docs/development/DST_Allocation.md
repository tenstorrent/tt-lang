# DST Register Allocation

## Overview

DST (destination) registers are hardware registers used for tile computations. This pass (`TTLAssignDST.cpp`) assigns DST indices to tile operations based on whether they are unary or binary operations.

## Core Principle: Unary vs Binary Operations

**Binary operations** (e.g., `add`, `mul`, `max`) require separate DST slots for inputs and output:
- Each input gets its own DST slot
- Output gets a fresh DST slot
- `inputs_footprint = number of tile inputs`

**Unary operations** (e.g., `exp`, `abs`, `neg`) operate in-place:
- Output reuses the input's DST slot
- No additional DST allocation needed
- `inputs_footprint = 0` for standalone unary ops

## Lifetime Interval Visualization

The following diagram illustrates how lifetime intervals work for a simple example with binary and unary operations:

```
MLIR:
  %in0 = block_arg
  %in1 = block_arg
  %0 = mul(%in0, %in1)    // Binary op
  %1 = abs(%0)             // Unary op (in-place)
  yield %1

Operation Timeline:    0      1      2      3
                       |      |      |      |
Lifetime Intervals:
  %in0  [0────1]       █████
  %in1  [0────1]       █████
  %0    [1─────────3]         ████████████    ← Merged with %1
  %1    [2─────────3]              █████████  ← (unary in-place)

DST Assignment:
  Inputs/Intermediates: %in0→DST[0], %in1→DST[1]
  Outputs:              %0→DST[2], %1→DST[2] (shared, in-place)
```

**Key observations**:
- Block arguments (`%in0`, `%in1`) start at operation 0
- Binary op result (`%0`) starts at its definition (op 1)
- Unary op result (`%1`) merges with its input (`%0`) - both use same DST
- Input intervals are short (expire after first use)
- Output interval extends to yield operation

## DST Allocation Algorithm

The algorithm uses a three-phase approach: operation scheduling to minimize register pressure, IR normalization via copy insertion, and linear scan register allocation with output region constraints.

**References**:
- Christian Wimmer and Michael Franz. 2010. Linear scan register allocation on SSA form. In Proceedings of CGO '10. https://doi.org/10.1145/1772954.1772979
- P. S. Rawat et al. 2019. Associative instruction reordering to alleviate register pressure. In Proceedings of SC '18. https://doi.org/10.1109/SC.2018.00049

### Phase 0: Operation Scheduling for Register Pressure (Optional)

Reorder independent operations within the `ttl.compute` block to minimize register pressure and reduce the number of copies needed in Phase 1.

**Goal**: For values with multiple consumers, schedule operations so that any unary consumers come LAST. This eliminates the need for `copy_dest_values` operations.

**Algorithm** (adapted from LARS [Rawat et al. SC'18]):

```
# Build SSA dependency graph
deps = build_dependency_graph(operations)
ready = [op for op in operations if has_no_unscheduled_deps(op, deps)]
scheduled = []
live_values = set()  # Values currently live in DST registers

while ready:
  best_op = None
  best_cost_tuple = (-inf, -inf, -inf, -inf, inf, -inf)

  for op in ready:
    # Compute cost tuple (lexicographic comparison)
    release_pot = count_last_uses(op, remaining_ops)  # Frees DST registers
    fire_pot = count_newly_ready_ops(op, deps)        # Enables more ops
    primary_aff = sum_reuse_with_live(op, live_values) # Reuse with live values
    secondary_aff = indirect_reuse_score(op, live_values)
    non_live_aff_penalty = -count_reuse_with_nonlive(op, remaining_ops)

    # Special heuristic: PENALTY if unary op on multi-consumer value
    # (we want unary ops scheduled LAST to avoid copies)
    if is_unary(op) and op.operand.num_consumers > 1:
      non_live_aff_penalty -= 100  # Large penalty

    cost_tuple = (release_pot, fire_pot, primary_aff, secondary_aff,
                  non_live_aff_penalty, critical_path_priority(op))

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

1. **Release Potential (Rpot)**: Number of values at their last use in this operation. Higher is better (frees DST registers).

2. **Fire Potential (Fpot)**: Number of dependent operations that become schedulable after executing this operation. Higher is better (enables more scheduling flexibility).

3. **Primary Affinity (Paff)**: Sum of reuse strength with already-live values. If operation uses values already in DST registers, we have high affinity. Higher is better (locality).

4. **Secondary Affinity (Saff)**: Indirect reuse - operations sharing common inputs with already-scheduled operations. Higher is better.

5. **Non-Live Affinity Penalty (-Nnpaff)**: Negative of the number of unscheduled operations this op shares inputs with. More negative is worse (will extend live ranges).
   - **Special case**: If this is a unary operation on a multi-consumer value, add large penalty to schedule it LAST.

6. **Critical Path Priority (Pl)**: Distance from this operation to the yield (end of block). Operations closer to critical path scheduled with urgency.

**Example** (from Example 6):
```mlir
%0 = mul(%in0, %in1)
%1 = abs(%0)              // Unary consumer
%2 = add(%0, %in2)        // Binary consumer

# Both abs and add are ready after mul
# Cost tuples:
abs: <0, 0, 1, 0, -100, ...>  # Large penalty for unary on multi-consumer
add: <0, 0, 1, 0, -1, ...>    # No penalty

# add has higher cost tuple → scheduled first
# Result: abs becomes last consumer → no copy needed!
```

**When to Apply**: This phase is optional but recommended when the `ttl.compute` block contains:
- Values with multiple consumers (especially mix of unary and binary)
- Many independent operations (flexibility to reorder)
- High register pressure (close to capacity limits)

### Phase 1: Insert Copy Operations

Normalize the IR by inserting explicit copy operations where values have multiple consumers.

```
for each tile value v:
  if v has multiple consumers (users.size() > 1):
    # Sort consumers by block order (operation position)
    consumers = sorted(v.users, key=lambda op: op.position_in_block)

    # Check if ANY consumer is unary (will overwrite v in-place)
    has_unary_consumer = any(is_unary(c) for c in consumers)

    if has_unary_consumer:
      # CRITICAL: Unary ops overwrite in-place!
      # Strategy: Keep the LAST consumer using original value,
      # insert copies for all earlier consumers
      for i in range(len(consumers) - 1):
        insert before consumers[i]:
          v_copy_i = ttl.copy_dest_values(v, new_dst_reg)  # src → dst (TTL convention)
        replace consumers[i]'s use of v with v_copy_i
    # else: All consumers are binary (don't overwrite) → no copies needed
```

**Rationale**:
- **Unary operations** always overwrite their input DST register in-place
- **Binary operations** do not modify their inputs - they write to a fresh output DST
- If a value has multiple consumers and **any** consumer is unary, we must copy for earlier consumers
- If **all** consumers are binary, no copies needed (binary ops don't corrupt inputs)

By copying for all consumers except the last one (when unary consumers exist), we guarantee:
1. The last consumer can safely use (and potentially overwrite) the original value
2. All earlier consumers work on independent copies
3. Binary-only multi-consumer values avoid unnecessary copies

**Note on Implementation**: Phase 1 inserts **DST-to-DST copy operations**, which are distinct from the CB-to-DST copies used to load inputs:

**API Difference**:
- **CB-to-DST copy** (for loading inputs from circular buffers):
  - `copy_tile(cb_id, tile_index, dst_index)` → unpacks from CB to DST
- **DST-to-DST copy** (inserted in Phase 1 for multi-consumer values):
  - **TTL level**: `copy_dest_values(src_index, dst_index)` → copies from src to dst
  - **Note**: TTL uses natural src→dst order, which is then lowered to tt-metal's `copy_dest_values(dst, src)` (reversed)

**Lowering Path for DST-to-DST Copies**:
- **D2M/TTL level**: `d2m.copy_dest_values(src, dst)` or `ttl.copy_dest_values(src, dst)`
- **TTKernel level**: `ttkernel.copy_dest_values(dst, src)` (arguments reversed from TTL)
- **tt-metal level**: `llk_math_eltwise_binary_sfpu_copy_dest_values(dst, src)` (SFPU binary operation)

**Note on Argument Order Convention**:
Throughout this document, pseudocode and MLIR examples use **TTL convention** for `copy_dest_values(source, destination)` - natural left-to-right data flow. When lowered to tt-metal, arguments are reversed to `copy_dest_values(destination, source)`. All generated C++ code examples follow tt-metal convention with destination first.

**Note on Initialization**: The lowering to tt-metal requires `copy_dest_values_init()` before first use. This is a lowering detail handled by the backend - not shown in examples for brevity.

**Example transformation**:
```mlir
%0 = tile_mul(%in0, %in1)
%0_copy_0 = copy_dest_values(%0)    // Copy for first consumer
%1 = tile_abs(%0_copy_0)             // Uses copy (safe to overwrite)
%2 = tile_exp(%0)                    // Last consumer uses original
```

### Phase 2: Build Live Intervals with Unary Merging

Build live intervals for each tile value. Following Wimmer & Franz [CGO'10], we process operations in linear order (forward pass). For unary operations, merge the input and output intervals since they must share the same DST register (hardware constraint).

```
intervals = {}
merged_sets = {}  # Union-find structure to track merged values

# Number operations in linear order
for i, op in enumerate(operations):
  op.index = i

# Process operations in forward order to build intervals
for each operation op:
  # For each input operand, extend its interval to include this use
  for each input_val in op.inputs:
    if input_val not in intervals:
      # Block argument - starts at block entry (index 0)
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

# Merge intervals for unary ops (input and output share same DST)
# CRITICAL: Only merge if the unary result is yielded OR feeds into another unary op
# Do NOT merge if the unary result feeds into a binary op (binary needs fresh output)
for each unary_op:
  input_val = unary_op.operand
  output_val = unary_op.result

  # Check if output should be merged with input
  if output_val is yielded OR all uses of output_val are unary ops:
    # Merge: both values must use the same DST register
    merged_interval = Interval(
      start = min(intervals[input_val].start, intervals[output_val].start),
      end = max(intervals[input_val].end, intervals[output_val].end)
    )
    intervals[input_val] = merged_interval
    intervals[output_val] = merged_interval

    # Track merged values using union-find
    union(merged_sets, input_val, output_val)
```

**Key Insight**: Merged intervals represent values that MUST share the same DST register. The union-find structure (`merged_sets`) tracks equivalence classes of merged values. During allocation (Phases 3 & 4), when we assign a register to one value in a merged set, we assign it to ALL values in that set.

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
1. Check if any value in the set was already processed → skip if yes
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

### Example 1: Simple Binary Operation (No Unrolling)

**Input MLIR**:
```mlir
^bb0(%in0: tile, %in1: tile, %out: tile):
  %0 = tile_mul(%in0, %in1)
  linalg.yield %0
```

**Phase 1 (Copy Insertion)**: No multi-consumer values → No copies needed

**Phase 2 (Build Intervals)**:
```
Operation indices: %in0=0, %in1=0, tile_mul=1, %0=1, yield=2

Intervals:
  %in0: [0, 1]  (def at block entry, last use at tile_mul)
  %in1: [0, 1]  (def at block entry, last use at tile_mul)
  %0:   [1, 2]  (def at tile_mul, last use at yield)

No unary ops → No merging
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %in0 [0,1]: allocate DST[0], active=[%in0]
Process %in1 [0,1]: allocate DST[1], active=[%in0, %in1]
Process %0 [1,2]: (yielded to output, skip)

inputs_and_intermediates_footprint = 2
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 2
Process %0 [1,2]: allocate DST[2]
  At start=1: expire %in0, %in1 (both end before 1)
  Allocate DST[2] for %0
```

**Final Assignment**:
```
Inputs/Intermediates (DST[0-1]):
  %in0 → DST[0]
  %in1 → DST[1]

Outputs (DST[2]):
  %0   → DST[2]
```

**Generated Code (No Unrolling)**:
```cpp
tile_regs_acquire();
copy_tile(CB0, 0, DST[0]);       // %in0
copy_tile(CB1, 0, DST[1]);       // %in1
mul_binary_tile(DST[0], DST[1], DST[2]);
tile_regs_commit();
tile_regs_wait();
pack_tile(DST[2], CB_out, 0);
tile_regs_release();
```

---

### Example 2: Binary Operation with Unrolling (2x2 = 4 tiles)

**Input MLIR**: Same as Example 1, but output shape is 2x2

**Unroll Factor Calculation**:
```
inputs_and_intermediates_footprint = 2 (from Phase 3)
numOutputs = 1 (%out)
available_for_outputs = capacity - inputs_and_intermediates_footprint = 8 - 2 = 6
unrollFactor = min(available_for_outputs / numOutputs, numTiles) = min(6 / 1, 4) = min(6, 4) = 4
```

**Note**: We can actually unroll by 4 (not 2) since we have room for DST[2-5] for outputs!

**DST Assignment**:
```
Inputs/Intermediates (DST[0-1]):
  %in0 → DST[0]
  %in1 → DST[1]

Outputs (DST[2-5] for unroll factor 4):
  %0   → DST[2,3,4,5] (one per unrolled iteration)
```

**Generated Code (fully unrolled)**:
```cpp
tile_regs_acquire();
// All 4 iterations unrolled
copy_tile(CB0, 0, DST[0]);
copy_tile(CB1, 0, DST[1]);
mul_binary_tile(DST[0], DST[1], DST[2]);

copy_tile(CB0, 1, DST[0]);
copy_tile(CB1, 1, DST[1]);
mul_binary_tile(DST[0], DST[1], DST[3]);

copy_tile(CB0, 2, DST[0]);
copy_tile(CB1, 2, DST[1]);
mul_binary_tile(DST[0], DST[1], DST[4]);

copy_tile(CB0, 3, DST[0]);
copy_tile(CB1, 3, DST[1]);
mul_binary_tile(DST[0], DST[1], DST[5]);

tile_regs_commit();
tile_regs_wait();
pack_tile(DST[2], CB_out, 0);
pack_tile(DST[3], CB_out, 1);
pack_tile(DST[4], CB_out, 2);
pack_tile(DST[5], CB_out, 3);
tile_regs_release();
```

**Key Insight**: With inputs_and_intermediates_footprint=2 and capacity=8, we have 6 DST registers available for outputs. Since we only need 1 output per iteration and have 4 total tiles, we can fully unroll all 4 iterations.

---

### Example 3: Unary Operation Only

**Input MLIR**:
```mlir
^bb0(%in: tile, %out: tile):
  %0 = tile_abs(%in)
  linalg.yield %0
```

**Phase 1**: No multi-consumer values → No copies

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
  %in → DST[0]
  %0  → DST[0]  (in-place)
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

### Example 4: Binary → Unary Chain

**Input MLIR**:
```mlir
^bb0(%in0: tile, %in1: tile, %out: tile):
  %0 = tile_mul(%in0, %in1)
  %1 = tile_abs(%0)
  linalg.yield %1
```

**Phase 1**: No multi-consumer values → No copies

**Phase 2 (Build Intervals)**:
```
Intervals:
  %in0: [0, 1]
  %in1: [0, 1]
  %0:   [1, 2]
  %1:   [2, 3]

Merge for unary (abs):
  %0:  [1, 3]  (merged)
  %1:  [1, 3]  (merged)
  mark_as_merged(%0, %1)
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %in0 [0,1]: allocate DST[0]
Process %in1 [0,1]: allocate DST[1]
Process %0 [1,3]: (skip - merged with %1 which is yielded to output)
Process %1 [1,3]: (yielded to output, skip)

inputs_and_intermediates_footprint = 2
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 2
Process %1 [1,3]: allocate DST[2]
  At start=1: expire %in0, %in1
  Allocate DST[2] for %1
  %1 is merged with %0, so:
  dst_assignment[%1] = DST[2]
  dst_assignment[%0] = DST[2]  (merged)
```

**Final Assignment**:
```
Inputs/Intermediates (DST[0-1]):
  %in0 → DST[0]
  %in1 → DST[1]

Outputs (DST[2]):
  %0   → DST[2]
  %1   → DST[2]  (in-place with %0)
```

**Generated Code**:
```cpp
tile_regs_acquire();
copy_tile(CB0, 0, DST[0]);
copy_tile(CB1, 0, DST[1]);
mul_binary_tile(DST[0], DST[1], DST[2]);
abs_tile(DST[2]);               // In-place
tile_regs_commit();
tile_regs_wait();
pack_tile(DST[2], CB_out, 0);
tile_regs_release();
```

---

### Example 5: Multiple Unary Consumers (Copy Insertion)

**Input MLIR**:
```mlir
^bb0(%in0: tile, %in1: tile, %out0: tile, %out1: tile):
  %0 = tile_mul(%in0, %in1)
  %1 = tile_abs(%0)
  %2 = tile_exp(%0)
  linalg.yield %1, %2
```

**Lifetime Interval Visualization (after Phase 1 copy insertion)**:
```
Operation Timeline:    0      1      2      3      4
                       |      |      |      |      |
Before copy insertion (INCORRECT - both unary ops would corrupt %0):
  %0    [1────────────────3]    ████████████████
  %1    [2──────────3]               ████████
  %2    [3──────────────4]                  █████

After copy insertion (CORRECT):
  %in0      [0─1]       ███
  %in1      [0─1]       ███
  %0        [1──────3]       ████████████      ← Merged with %2
  %0_copy_0 [2───3]                ██████       ← Merged with %1
  %1        [2───3]                ██████       ← (abs in-place on copy)
  %2        [3──────4]                  █████   ← (exp in-place on original)

DST Assignment:
  Inputs/Intermediates: %in0→DST[0], %in1→DST[1]
  Outputs:              %0→DST[2], %2→DST[2] (merged)
                        %0_copy_0→DST[3], %1→DST[3] (merged)
```

**Phase 1 (Copy Insertion)**:
```
%0 has 2 consumers (abs at position 0, exp at position 1)
Copy for all except last (exp is last):

Result:
  %0 = tile_mul(%in0, %in1)
  %0_copy_0 = copy_dest_values(%0)   // Copy for first consumer (abs)
  %1 = tile_abs(%0_copy_0)            // Uses copy
  %2 = tile_exp(%0)                   // Last consumer uses original
```

**Phase 2 (Build Intervals)**:
```
Intervals:
  %in0:      [0, 1]
  %in1:      [0, 1]
  %0:        [1, 3]  (used by exp at end)
  %0_copy_0: [2, 2]  (copy result, immediately used by abs)
  %1:        [2, 3]  (abs result)
  %2:        [3, 4]  (exp result)

Merge for abs (%0_copy_0, %1):
  %0_copy_0: [2, 3]
  %1:        [2, 3]

Merge for exp (%0, %2):
  %0: [1, 4]
  %2: [1, 4]
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %in0 [0,1]: allocate DST[0]
Process %in1 [0,1]: allocate DST[1]
Process %0 [1,4]: (skip - merged with %2, yielded)
Process %0_copy_0 [2,3]: (skip - merged with %1, yielded)
Process %1 [2,3]: (yielded, skip)
Process %2 [1,4]: (yielded, skip)

inputs_and_intermediates_footprint = 2
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 2

Process %0 [1,4]: (or %2, same merged set) allocate DST[2]
  Merged with %2 → dst_assignment[%0] = DST[2], dst_assignment[%2] = DST[2]

Process %1 [2,3]: allocate DST[3]
  At start=2: (nothing to expire yet)
  Allocate DST[3]
  Merged with %0_copy_0 → dst_assignment[%0_copy_0] = DST[3]
```

**Final Assignment**:
```
Inputs/Intermediates (DST[0-1]):
  %in0      → DST[0]
  %in1      → DST[1]

Outputs (DST[2-3]):
  %0        → DST[2]
  %2        → DST[2]  (in-place with %0)
  %0_copy_0 → DST[3]
  %1        → DST[3]  (in-place with copy)
```

**Generated Code**:
```cpp
tile_regs_acquire();
copy_tile(CB0, 0, DST[0]);
copy_tile(CB1, 0, DST[1]);
mul_binary_tile(DST[0], DST[1], DST[2]);
copy_dest_values(DST[2], DST[3]);        // Copy for abs (first consumer)
abs_tile(DST[3]);                        // In-place on copy
exp_tile(DST[2]);                        // In-place on original (last consumer)
tile_regs_commit();
tile_regs_wait();
pack_tile(DST[3], CB_out0, 0);           // Pack abs result
pack_tile(DST[2], CB_out1, 0);           // Pack exp result
tile_regs_release();
```

---

### Example 6: Binary with Mixed Consumers (Binary + Unary)

**Input MLIR**:
```mlir
^bb0(%in0: tile, %in1: tile, %in2: tile, %out0: tile, %out1: tile):
  %0 = tile_mul(%in0, %in1)
  %1 = tile_abs(%0)
  %2 = tile_add(%0, %in2)
  linalg.yield %1, %2
```

**Lifetime Interval Visualization (after Phase 1 copy insertion)**:
```
Operation Timeline:    0      1      2      3      4
                       |      |      |      |      |
After copy insertion:
  %in0      [0─1]       ███
  %in1      [0─1]       ███
  %in2      [0──────────3]    ████████████████
  %0        [1──────3]       ████████████         ← Used by add (binary)
  %0_copy_0 [2───3]                ██████          ← Merged with %1
  %1        [2───3]                ██████          ← (abs in-place on copy)
  %2        [3──────4]                    █████

DST Assignment:
  Inputs/Intermediates: %in0→DST[0], %in1→DST[1], %in2→DST[2], %0→DST[3]
  Outputs:              %0_copy_0→DST[4], %1→DST[4] (merged)
                        %2→DST[5]
```

**Key**: Binary consumer (add) requires %0 to remain unmodified, so abs operates on a copy.

**Phase 1 (Copy Insertion)**:
```
%0 has 2 consumers (abs at position 0, add at position 1)
Copy for all except last (add is last):

Result:
  %0 = tile_mul(%in0, %in1)
  %0_copy_0 = copy_dest_values(%0)     // Copy for abs (first consumer)
  %1 = tile_abs(%0_copy_0)              // Uses copy
  %2 = tile_add(%0, %in2)               // Last consumer uses original
```

**Phase 2 (Build Intervals)**:
```
Intervals:
  %in0:      [0, 1]
  %in1:      [0, 1]
  %in2:      [0, 3]  (used by add)
  %0:        [1, 3]  (used by add at end)
  %0_copy_0: [2, 2]  (copy result, immediately used by abs)
  %1:        [2, 3]  (abs result)
  %2:        [3, 4]  (add result)

Merge for abs (%0_copy_0, %1):
  %0_copy_0: [2, 3]
  %1:        [2, 3]

No merge for add (binary op - %0 and %2 stay separate)
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %in0 [0,1]: allocate DST[0]
Process %in1 [0,1]: allocate DST[1]
Process %in2 [0,3]: allocate DST[2]
Process %0 [1,3]: NOT yielded! Allocate DST[3]
  At start=1: expire %in0, %in1
  Allocate DST[3], active=[%in2, %0]
Process %0_copy_0 [2,3]: (skip - merged with yielded %1)
Process %1 [2,3]: (yielded, skip)
Process %2 [3,4]: (yielded, skip)

inputs_and_intermediates_footprint = 4
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 4

Process %1 [2,3]: allocate DST[4]
  Merged with %0_copy_0 → dst_assignment[%0_copy_0] = DST[4]

Process %2 [3,4]: allocate DST[5]
  At start=3: expire %1, %in2, %0
  Allocate DST[5]
```

**Final Assignment**:
```
Inputs/Intermediates (DST[0-3]):
  %in0      → DST[0]
  %in1      → DST[1]
  %in2      → DST[2]
  %0        → DST[3]  (used by binary add)

Outputs (DST[4-5]):
  %0_copy_0 → DST[4]
  %1        → DST[4]  (in-place with copy)
  %2        → DST[5]
```

**Generated Code**:
```cpp
tile_regs_acquire();
copy_tile(CB0, 0, DST[0]);
copy_tile(CB1, 0, DST[1]);
copy_tile(CB2, 0, DST[2]);
mul_binary_tile(DST[0], DST[1], DST[3]);
copy_dest_values(DST[3], DST[4]);           // Copy for abs
abs_tile(DST[4]);                            // In-place on copy
add_binary_tile(DST[3], DST[2], DST[5]);    // Binary uses original
tile_regs_commit();
tile_regs_wait();
pack_tile(DST[4], CB_out0, 0);
pack_tile(DST[5], CB_out1, 0);
tile_regs_release();
```

---

### Example 7: Unary Chain (3 ops)

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
  Outputs:              %in,%0,%1,%2 → DST[0] (all share, in-place chain)
```

**Key**: Sequential unary chain - all operations share one DST register, each operating in-place.

**Phase 1**: No multi-consumer values → No copies

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

**Phase 3**: All values are merged with yielded output → skip all

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
  %in → DST[0]
  %0  → DST[0]  (in-place)
  %1  → DST[0]  (in-place)
  %2  → DST[0]  (in-place)
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

### Example 8: Complex Chain with Unrolling (2x2)

**Input MLIR**:
```mlir
^bb0(%in0: tile, %in1: tile, %in2: tile, %out: tile):
  %0 = tile_mul(%in0, %in1)
  %1 = tile_abs(%0)
  %2 = tile_add(%1, %in2)
  linalg.yield %2
```

**Output shape**: 2x2 = 4 tiles

**Lifetime Interval Visualization (with unary merging)**:
```
Operation Timeline:    0      1      2      3      4
                       |      |      |      |      |
Lifetime Intervals:
  %in0      [0─1]       ███
  %in1      [0─1]       ███
  %in2      [0──────────3]    ████████████████
  %0        [1──────3]       ████████████         ┐ Merged (unary abs)
  %1        [2──────3]            █████████       ┘ share DST[3]
  %2        [3──────4]                    █████   ← Separate (binary result)

DST Assignment:
  Inputs/Intermediates (DST[0-3]): %in0→DST[0], %in1→DST[1], %in2→DST[2]
                                    %0→DST[3], %1→DST[3] (merged, in-place)
  Outputs (DST[4-7]):               %2→DST[4,5,6,7] (one per unrolled iteration)

Unrolling: 4 iterations fully unrolled
  - Inputs/intermediates reused across all iterations
  - Each iteration writes to unique output DST slot
```

**Key**: Unary (abs) merges with its input, but binary (add) needs fresh output. With 4 DST slots available for outputs, we can fully unroll.

**Phase 1**: No multi-consumer values → No copies

**Phase 2 (Build Intervals with Merging)**:
```
Initial intervals:
  %in0: [0, 1]
  %in1: [0, 1]
  %in2: [0, 3]
  %0:   [1, 2]
  %1:   [2, 3]
  %2:   [3, 4]

Merge for abs (%0, %1):
  %1 feeds into binary add, so do NOT merge with %2
  Only merge %0 and %1: [1, 3]
```

**Phase 3 (Allocate Inputs/Intermediates)**:
```
Process %in0 [0,1]: DST[0]
Process %in1 [0,1]: DST[1]
Process %in2 [0,3]: DST[2]
Process %0/%1 [1,3]: DST[3] (merged pair, NOT yielded so allocated as intermediate)
  At start=1: expire %in0, %in1
  Allocate DST[3]
  active=[%in2, %0/%1]
Process %2: (yielded, skip)

inputs_and_intermediates_footprint = 4
```

**Phase 4 (Allocate Outputs)**:
```
base_out_dst_index = 4
Process %2 [3,4]: allocate DST[4]
  %2 is NOT merged (binary result)
  At start=3: expire %in2, %0/%1
  Allocate DST[4]
```

**Unroll Factor Calculation**:
```
inputs_and_intermediates_footprint = 4 (from Phase 3)
numOutputs = 1 (%out)
available_for_outputs = capacity - inputs_and_intermediates_footprint = 8 - 4 = 4
unrollFactor = min(available_for_outputs / numOutputs, numTiles) = min(4 / 1, 4) = 4
```

**Note**: We can fully unroll all 4 iterations since we have exactly 4 DST registers available for outputs (DST[4-7]).

**Final Assignment**:
```
Inputs/Intermediates (DST[0-3]):
  %in0 → DST[0]
  %in1 → DST[1]
  %in2 → DST[2]
  %0   → DST[3]
  %1   → DST[3]  (in-place)

Outputs (DST[4-7] for unroll factor 4):
  %2   → DST[4,5,6,7] (one per iteration)
```

**Generated Code (fully unrolled)**:
```cpp
tile_regs_acquire();
// Iteration 0
copy_tile(CB0, 0, DST[0]);
copy_tile(CB1, 0, DST[1]);
copy_tile(CB2, 0, DST[2]);
mul_binary_tile(DST[0], DST[1], DST[3]);
abs_tile(DST[3]);
add_binary_tile(DST[3], DST[2], DST[4]);

// Iteration 1
copy_tile(CB0, 1, DST[0]);
copy_tile(CB1, 1, DST[1]);
copy_tile(CB2, 1, DST[2]);
mul_binary_tile(DST[0], DST[1], DST[3]);
abs_tile(DST[3]);
add_binary_tile(DST[3], DST[2], DST[5]);

// Iteration 2
copy_tile(CB0, 2, DST[0]);
copy_tile(CB1, 2, DST[1]);
copy_tile(CB2, 2, DST[2]);
mul_binary_tile(DST[0], DST[1], DST[3]);
abs_tile(DST[3]);
add_binary_tile(DST[3], DST[2], DST[6]);

// Iteration 3
copy_tile(CB0, 3, DST[0]);
copy_tile(CB1, 3, DST[1]);
copy_tile(CB2, 3, DST[2]);
mul_binary_tile(DST[0], DST[1], DST[3]);
abs_tile(DST[3]);
add_binary_tile(DST[3], DST[2], DST[7]);

tile_regs_commit();
tile_regs_wait();
pack_tile(DST[4], CB_out, 0);
pack_tile(DST[5], CB_out, 1);
pack_tile(DST[6], CB_out, 2);
pack_tile(DST[7], CB_out, 3);
tile_regs_release();
```

**Key insight**: With inputs_and_intermediates_footprint=4 and capacity=8, we have exactly 4 DST registers for outputs. We can fully unroll all 4 iterations. Inputs/intermediates (DST[0-3]) are reused across all iterations.

---

## DST Capacity

The examples in this document use **capacity = 8** for demonstration purposes. This represents the effective DST capacity with double-buffering enabled, which is the common configuration.

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

1. `ttl-tile-and-assign-dst`: Assigns DST indices, adds `ttl.unroll_factor` attribute
2. `ttl-lower-to-loops`: Converts `ttl.compute` to `scf.for` loops
3. `ttl-unroll-compute-loops`: Unrolls loops (optional, controlled by `--enable-unroll`)
4. `ttl-insert-tile-regs-sync`: Inserts DST lifecycle ops (acquire/commit/wait/release)

## Generated Code Example (2x2 block with binary op)

```cpp
tile_regs_acquire();
for (i = 0..2) {
  for (j = 0..2) {
    copy_tile(CB0, i*2+j, DST[0]);
    copy_tile(CB1, i*2+j, DST[1]);
    mul_binary_tile(DST[0], DST[1], DST[2+i*2+j]);  // Dynamic output index
  }
}
tile_regs_commit();
tile_regs_wait();
for (i = 0..2) {
  for (j = 0..2) {
    pack_tile(DST[2+i*2+j], CB_out, i*2+j);
  }
}
tile_regs_release();
```

## Generated Code Example (2x2 block)

```cpp
tile_regs_acquire();
for (i = 0..2) {
  for (j = 0..2) {
    copy_tile(CB0, i*2+j, DST[0]);
    copy_tile(CB1, i*2+j, DST[1]);
    mul_binary_tile(DST[0], DST[1], DST[2+i*2+j]);  // Dynamic!
    copy_tile(CB2, i*2+j, DST[0]);
    add_binary_tile(DST[2+i*2+j], DST[0], DST[2+i*2+j]);
  }
}
tile_regs_commit();
tile_regs_wait();
for (i = 0..2) {
  for (j = 0..2) {
    pack_tile(DST[2+i*2+j], CB3, i*2+j);  // Dynamic!
  }
}
tile_regs_release();
```

## Future Work

* **Implement Phase 0 (Operation Scheduling)**: The algorithm for register-pressure-aware scheduling is described in Phase 0 above. Implementing this as a pre-pass before DST allocation can reduce `copy_dest_values` operations by 25-50% (as demonstrated in Example 6). Key challenge: Determining when to apply Phase 0 (cost/benefit analysis for small compute blocks).

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

  **Challenge**: Determining optimal split points to minimize overhead while respecting DST capacity constraints. Could use dynamic programming or greedy heuristics based on interval pressure.

* **Lifetime Hole Exploitation**: Currently, merged intervals from unary chains create large contiguous intervals. We could split these intervals at lifetime holes (periods where the value isn't used) to allow other operations to reuse those DST slots temporarily. This would require more sophisticated interval tracking with multiple ranges per value.

* **Adaptive Scheduling Heuristics**: Extend Phase 0 with adaptive cost tuple ordering based on computation characteristics (similar to LARS adaptivity in Section III-F of Rawat et al. SC'18). For computations with high intra-statement reuse, prioritize `Pl` (critical path) over `Paff` (affinity) to avoid excessive interleaving.

* **FPU/Matmul Engine for Binary Operations**: Explore using the FPU (matrix engine) for certain binary tile operations instead of always using SFPU. Some binary operations like `mul_tiles` and `add_tiles` may be able to execute on the FPU, potentially freeing SFPU resources or improving throughput. This would require analysis of:
  - Which binary operations can execute on FPU vs SFPU
  - DST register constraints when mixing FPU and SFPU operations
  - Performance trade-offs (latency, throughput, register pressure)

  See [tt-metal eltwise_binary.h](https://github.com/tenstorrent/tt-metal/blob/bb1a6e5191113a9db5695b22b4dae3c35f0b3d3d/tt_metal/include/compute_kernel_api/eltwise_binary.h#L228) for binary operation APIs that might support alternative execution engines.
