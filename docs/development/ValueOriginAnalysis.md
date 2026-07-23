# Value Origin Analysis

## Purpose

`getOrigins(value)` follows `value` through control-flow merges, loop-carried
values, and casts, then returns its possible source definitions. For a
`tensor.extract` result, it also determines which tensor insertion may define
the extracted element. This is required when compiler correctness
depends on properties of every possible definition. For example, `ttl.wait` is
valid only if every possible transfer-handle origin is an asynchronous
transfer. Selecting one definition from a control-flow merge or one insertion
from a tensor update can accept an unrelated value and produce incorrect
synchronization.

Conceptually, this is a may-analysis of value origins. At a control-flow merge
or tensor update, it joins the possible source definitions by set union. It
also distinguishes which tensor update may define the extracted element. A
conservative result may include definitions that do not execute, but it must
not omit a possible source definition.

SSA use-def traversal alone is insufficient because values may pass through
control-flow merges, loop-carried values, and functional tensor updates.
Tensor extraction additionally requires proving which insertions may
define the extracted element across dynamic loop iterations. When that relation
cannot be proven, the analysis must retain every possible definition so the
consumer rejects an unsupported program instead of accepting incomplete
information.

MLIR provides the required components but not an analysis with these
semantics. Its control-flow predecessor utility follows selects, block
branches, and region branches. `LoopLikeOpInterface` associates loop initial,
iterated, yielded, and result values and provides static trip counts. tt-lang's
shared `LoopIterationUtils` evaluates additional `scf.for` trip counts and
enumerates finite loop assignments using `IntegerExpressionEvaluator` and
MLIR's `constantTripCount`. None of these components tracks tensor elements
through `tensor.insert` and `tensor.extract`, compares finite access domains, or
invalidates index correlation across a loop backedge. `ValueOriginAnalysis`
composes these building blocks into one conservative analysis.

This analysis does not infer storage attachment, arbitrary expression
dependency, or resource lifetime. `getAttachedCB`, coordinate-dependency
analysis, and DFB acquire/release ownership compute those separate relations.

## Terms

An **origin** is an SSA value at which this traversal stops because its producer
is not modeled. A tensor value remains an unresolved origin when the extracted
element cannot be resolved further.

When analyzing `tensor.extract`, the analysis records the tensor value, index
tuple, and extraction operation. The extraction operation provides the dynamic
execution context for its indices.

An **access domain** is the set of concrete index tuples evaluated by an
extraction or insertion over the finite enclosing loop iterations. Assignments
to shared enclosing loops remain correlated. Independent writer and reader
loops contribute independent iteration dimensions.

Following a loop-carried tensor to its prior-iteration value makes the access
relation unknown for writes inside that loop. Comparing aggregate index sets
would ignore temporal ordering and could use a current or future write to
justify a read of prior state. An unknown relation retains the inserted value
and the preceding tensor state.

Loop results include the initial value when the loop may execute zero times.
Loop body arguments include the initial value and, when the loop may repeat,
the prior iteration's yielded value. The analysis obtains these associations
from `LoopLikeOpInterface` and exact trip counts from the shared loop iteration
utilities.

## Semantics

Ordinary SSA definitions are origins. At a select or control-flow merge, the
analysis continues from every possible input:

```text
origins(select(condition, true_value, false_value))
    = origins(true_value) union origins(false_value)

origins(region_or_block_argument)
    = union(origins(predecessor) for every control-flow predecessor)
```

The analysis uses `RegionBranchOpInterface`, `BranchOpInterface`, and
`SelectLikeOpInterface` through MLIR's control-flow predecessor utility.
Region branches use successor-relative indexing to avoid LLVM issue #175168
when a region has leading arguments that are not successor inputs. One-input
and one-result unrealized casts are also traversed. Tensor elements are traced
through `tensor.cast`, whose semantics preserve element indices.

For a value extracted from a tensor, the analysis retains the extracted indices:

```text
origins(tensor.extract tensor[read_indices])
    = element_origins(tensor, read_indices)
```

For `tensor.insert`, the read access domain is compared with the write access
domain:

```text
if read_domain is a subset of write_domain:
    element_origins = origins(inserted_scalar)
else if read_domain and write_domain are disjoint:
    element_origins = element_origins(previous_tensor, read_indices)
else:
    element_origins = origins(inserted_scalar)
                      union element_origins(previous_tensor, read_indices)
```

The subset rule models a loop-carried tensor updated at multiple indices. For
example, a writer loop that inserts at every index in `[0, 4)` fully defines a
later reader loop that extracts every index in `[0, 4)`. Partial writer
coverage retains the initial tensor as a possible origin.

## Conservative results

Finite index domains are evaluated with `LoopIterationUtils`. Separate limits
bound the loop iterations examined and index tuples produced. Bounds or index
expressions that cannot be evaluated, unsupported loop interfaces, and
exhaustion of either limit make the access relation unknown. An unknown
relation follows both the inserted scalar and the previous tensor contents.

Unmodeled tensor producers remain origins. This preserves soundness: a
consumer that requires a specific origin rejects the result instead of
accepting an incomplete traversal.
