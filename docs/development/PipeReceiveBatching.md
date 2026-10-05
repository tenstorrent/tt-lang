# Initial PipeNet receive batching

A receiver that waits for each sender before notifying the next serializes
otherwise independent transfers. When all destination slots are available at
entry, the receiver can notify every sender first. It must still wait for and
publish each payload in the original record order.

```text
Before: reserve(1), post(A), wait(A), push(1), reserve(1), post(B), wait(B), push(1)
After:  reserve(2), post(A), post(B), atomic_barrier(), wait(A), push(1), wait(B), push(1)
```

`post(A)` notifies sender A that its destination slot is reserved. It does not
publish payload to the compute kernel. `push(1)` does that, after `wait(A)`
confirms completion. Thus batching does not reorder reduction inputs.

## Storage proof before lowering

`annotateInitialPipeReceiveBatches` uses the existing PipeGraph and selected
transfer resource plan while logical DFB identities are still available. It
accepts only a local record loop with one receiver post and one physical-index
reservation across the module. Every writer endpoint for that DFB must belong
to this post, execute once, and occupy a distinct slot. PipeGraph must prove
that all producer advances belong to receives, and resource planning must have
chosen sender-computed DFB addresses.

The existing resource allocator treats selected-record transfers as overlapping
in `pipeResourceUnitsInterfere`, giving them distinct readiness words at each
sender. Completion counters distinguish transfers at the same receiver; their
only grouping mechanism joins alternative posts of one wait-any channel. The
single-post producer proof and regular completion-wait requirement exclude that
case. Batching reuses these guarantees instead of building another counter
ownership analysis. A sender's readiness reset therefore cannot discard another
record's notification.

The enclosing function has one block; between it and the record loop only
`scf.if` operations are accepted. Enclosing loops, external calls, resets, and
reconfiguration are excluded because the initial-state proof does not model
them. Physical-index aliases with additional reservations are also excluded.

These restrictions prove the receive sequence starts before any producer has
filled this storage. Merely checking DFB capacity would be insufficient: an
upfront reservation could otherwise wait for old payload whose consumption
depends on an earlier receive completing.

## Scheduling after specialization

`ttkernel-batch-static-pipenet-receives` runs before record-loop unrolling.
It requires a constant positive trip count greater than one and checks that
the combined page count fits the proven capacity. Checked arithmetic rejects
unrepresentable page counts and induction values before mutation.

The loop must contain exactly one reserve, readiness increment, completion
wait, and push in that order. Other operations must be pure, except accesses
to nonescaping stack counters after the readiness notification. Those counter
accesses stay in their original order and cannot be observed by senders.
Nested regions, receiver-published addresses, callbacks, and other memory
effects retain sequential execution. DFB identity and NoC
selection must be independent of the record loop.

The pass plans all eligible loops on immutable IR, then emits the combined
reservation, cloned address calculations and readiness increments, one atomic
barrier, and the original ordered completion/push sequences. The accepted
loops contain no nested regions, so their plans do not overlap. Temporary
capacity annotations are removed for accepted and rejected loops.

MLIR supplies static trip-count evaluation, constant matching, pure-operation
classification, and SSA cloning through `IRMapping`; PipeGraph supplies
DFB ownership and address recurrence facts. Batching does not introduce a
second address model or modify sender code.

## Conservative behavior

Dynamic bounds, zero or one receive, insufficient capacity, repeated receive
sequences, and unproven state retain the original transfer protocol. Core
specialization is not required, but commonly resolves the per-worker record
count needed for batching. The pass runs in both pipeline configurations so
temporary annotations never reach generated C++.

The full C++ pipeline, standalone specialization pipeline, and Python compiler
share `ttkernel-cleanup-and-finalize-runtime-args`'s implementation. This
sequence batches receives, unrolls records, lowers affine index arithmetic,
folds table lookups, reapplies
TTKernel cleanup, and finally removes unused runtime arguments. Python invokes
the registered sequence rather than maintaining its own pass list. The
TreeReduce device test checks the expanded Python pass order and numerical
results with specialization enabled and disabled.
