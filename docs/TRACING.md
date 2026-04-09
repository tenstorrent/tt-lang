# Simulator Tracing Design

## Overview

This document describes the design of the simulator's tracing system. The goal is to
collect structured execution traces during operation simulation and use them to derive
performance statistics and produce visualizations.

Tracing is opt-in: it adds zero overhead when disabled.

---

## What We Trace

Tracing is organized around the events that matter for understanding operation behavior.
Only semantic events are recorded -- simulator internals (scheduler bookkeeping,
context management) are not traced.

### Operation scope

A span covering the entire execution of a `ttl.operation` on a single node:

- **operation_start** / **operation_end** -- node index, grid shape

### Kernel lifecycle

One span per kernel per node:

- **kernel_start** / **kernel_end** -- node index, kernel type (COMPUTE / DM0 / DM1)
- **kernel_block** -- the kernel could not proceed (DFB slot unavailable); records
  which DFB and which operation (reserve / wait)
- **kernel_unblock** -- the kernel resumed after a block; the tick gap between
  kernel_block and kernel_unblock is the stall in logical time, i.e. the number of
  other kernel activations that occurred while this kernel was waiting

### DataflowBuffer operations

Instant events at every DFB operation. Events inside a counted loop include an
`iteration` attribute for per-iteration analysis; events outside any loop omit it:


| Event | When | Attributes |
|-------|------|------------|
| dfb_reserve_begin | `reserve()` called | dfb name, node, kernel |
| dfb_reserve_end | slot acquired (may follow a stall) | dfb name, occupied slots after |
| dfb_push | `push()` called | dfb name, occupied slots after |
| dfb_wait_begin | `wait()` called | dfb name, node, kernel |
| dfb_wait_end | slot acquired | dfb name, occupied slots after |
| dfb_pop | `pop()` called | dfb name, occupied slots after |

Recording occupied slots at push/pop gives buffer occupancy over time, which shows
whether a pipeline is producer-bound or consumer-bound.

### Copy operations

A span covering each `copy()` call:

- **copy_start** -- emitted at the end of `copy()`, after the source and destination
  blocks have been validated and locked for transfer.
- **copy_end** -- emitted at the end of `tx.wait()`, after the data transfer has
  completed and the blocks have been released. Fields: source type, destination type,
  node, kernel.

`copy_start` and `copy_end` are therefore the simulator equivalents of issuing and
completing a DMA: the tick difference between them is the number of scheduler
activations the issuing kernel spent blocked waiting for the transfer to finish.

---

## Logical Time

Logical time in the simulator is a **tick counter**: a monotonically increasing integer
incremented by the scheduler each time it activates a kernel. One tick = one scheduler
activation.

Formally:

    tick(t+1) = tick(t) + 1  for every greenlet activation in the scheduling loop

One activation runs a kernel from its current position until it blocks or completes.
A single activation may execute many statements -- DFB operations, arithmetic, loop
iterations -- without any context switch. Ticks do not count individual events within
a kernel; they count how many times the scheduler handed control to any kernel.

This definition has useful properties:

- **Deterministic**: for the same operation and input, the same sequence of ticks always
  occurs.
- **Comparable across kernels**: if kernel A pushes to a DFB at tick 5 and kernel B
  pops from it at tick 7, the gap of 2 ticks is meaningful -- two activations of other
  kernels happened between the two events.
- **Stall-measurable**: a kernel that blocked from tick 5 to tick 12 was stalled for
  7 ticks, during which 7 other kernel activations occurred.
- **No wall-clock dependency**: the trace is reproducible and independent of machine
  load.

Ticks are stored as integers. They carry no wall-clock meaning, and any mapping from
ticks to real time would be misleading. Visualization tools that require a physical
time unit are not suitable for this trace format.

## How Time Keeping Differs Across Schedulers

The assignment of tick numbers to events depends on the scheduling policy. As an
example, consider a single node with `compute`, `dm1`, and `dm2`, where `dm1`
where `dm1` produces to `dfb` twice, `compute` consumes from `dfb` twice and
produces to `dfb` twice and finally `dm1` produces to `dfb` twice. Under a
greedy scheduler, each kernel's two operations fall within the same activation
(same clock tick), so the trace batches them together:

```
tick=1  node=0  kernel=dm1      produce  dfb
tick=1  node=0  kernel=dm1      produce  dfb
tick=2  node=0  kernel=compute  consume  dfb
tick=2  node=0  kernel=compute  consume  dfb
tick=2  node=0  kernel=compute  produce  dfb
tick=2  node=0  kernel=compute  produce  dfb
tick=3  node=0  kernel=dm2      consume  dfb
tick=3  node=0  kernel=dm2      consume  dfb
```

Under a fair scheduler, kernels are interleaved more aggressively, so each activation
does less work before yielding and each operation appears at a distinct clock tick:

```
tick=1  node=0  kernel=dm1      produce  dfb
tick=2  node=0  kernel=compute  consume  dfb
tick=3  node=0  kernel=dm1      produce  dfb
tick=4  node=0  kernel=compute  consume  dfb
tick=5  node=0  kernel=compute  produce  dfb
tick=6  node=0  kernel=dm2      consume  dfb
tick=7  node=0  kernel=compute  produce  dfb
tick=8  node=0  kernel=dm2      consume  dfb
```

The fields shown here are illustrative; actual trace fields are described in the
sections above and examples are given below.

### Implications for trace interpretation

A trace from the greedy scheduler and a trace from the fair scheduler for the same
`ttl.operation` will have different tick values for the same events. Neither is more
"correct" -- they reflect different scheduling policies. When comparing traces or
aggregating statistics across runs, the scheduler algorithm must be noted as context.

### Future schedulers

The tick counter is the only coupling between the tracing system and the scheduling
strategy. Any future greenlet-based scheduler -- whether fair, greedy, or a new
policy -- must satisfy one contract:

    The scheduler increments the tick counter exactly once each time it activates
    a kernel productively.

Different strategies will produce different tick distributions (as illustrated above),
but the tracing infrastructure is unchanged regardless of which strategy is active.

---

## Modularity Provisions

The tracing system is designed so that the scheduler implementation is swappable
without modifying instrumentation call sites.

### Timestamp source abstraction

A single function provides the current timestamp:

```python
def get_trace_timestamp() -> int:
    return get_scheduler().tick
```

All `trace()` calls go through `get_trace_timestamp()`. Any greenlet-based
scheduler exposes a `tick` property that increments on each productive activation;
no other contract is required.

### Context storage

Trace events are stored in `SimulatorContext.trace`, which is already greenlet-local.
All instrumentation reads from `get_context()`. No instrumentation call site depends
on the scheduling strategy.

### Trace collection interface

A single primitive records any named event:

```python
trace(event: str)
```

The current node, kernel, and tick are read automatically from `get_context()` and
the scheduler. No context needs to be passed at the call site.

For event-specific data that cannot be derived from the simulator context -- for
example, the number of occupied slots after a DFB push -- the value is passed as
a positional argument alongside the event name. The exact signature is an
implementation detail; the design principle is that the call site passes only what
the context does not already know.

Some operations produce paired events -- one when the operation begins and one when
it completes, which may be in a later activation if the kernel blocked. For example,
a DFB reserve that may block:

```python
trace("dfb_reserve_begin")
# ... acquire slot, possibly blocking ...
trace("dfb_reserve_end")
```

The tick difference between the two is the stall for that reserve. There is no
special construct for this: it is just two `trace()` calls at the appropriate points
in the code.

### Example

Consider a `ttl.operation` with two kernels: `dm_read` (producer) and `compute`
(consumer), connected by a single-slot DFB. Each kernel runs two iterations.
`dm_read` is registered before `compute`.

The call sites in the implementation look like this:

```python
# scheduler: emit on kernel entry and exit
trace("kernel_start")
trace("kernel_end")

# scheduler: emit around a blocking switch
trace("kernel_block", op=operation, dfb=blocking_obj.name)
self._main_greenlet.switch()
trace("kernel_unblock")   # resumes here after being unblocked

# dfb: emit on every reserve/push/wait/pop
trace("dfb_reserve_begin", dfb=self._name)
trace("dfb_reserve_end",   dfb=self._name, occupied=self._occupied)
trace("dfb_push",          dfb=self._name, occupied=self._occupied)
```

Running this `ttl.operation` with tracing enabled produces the following event stream:

```
tick=1  kernel=node0-dm_read   event=kernel_start
tick=1  kernel=node0-dm_read   event=dfb_reserve_begin  dfb=dfb1
tick=1  kernel=node0-dm_read   event=dfb_reserve_end    dfb=dfb1  occupied=1
tick=1  kernel=node0-dm_read   event=dfb_push           dfb=dfb1  occupied=1
tick=1  kernel=node0-dm_read   event=dfb_reserve_begin  dfb=dfb1
tick=1  kernel=node0-dm_read   event=kernel_block       op=reserve  dfb=dfb1
tick=2  kernel=node0-compute   event=kernel_start
tick=2  kernel=node0-compute   event=dfb_wait_begin     dfb=dfb1
tick=2  kernel=node0-compute   event=dfb_wait_end       dfb=dfb1  occupied=1
tick=2  kernel=node0-compute   event=dfb_pop            dfb=dfb1  occupied=0
tick=2  kernel=node0-compute   event=dfb_wait_begin     dfb=dfb1
tick=2  kernel=node0-compute   event=kernel_block       op=wait  dfb=dfb1
tick=3  kernel=node0-dm_read   event=kernel_unblock
tick=3  kernel=node0-dm_read   event=dfb_reserve_end    dfb=dfb1  occupied=1
tick=3  kernel=node0-dm_read   event=dfb_push           dfb=dfb1  occupied=1
tick=3  kernel=node0-dm_read   event=kernel_end
tick=4  kernel=node0-compute   event=kernel_unblock
tick=4  kernel=node0-compute   event=dfb_wait_end       dfb=dfb1  occupied=1
tick=4  kernel=node0-compute   event=dfb_pop            dfb=dfb1  occupied=0
tick=4  kernel=node0-compute   event=kernel_end
```

Several things to observe:

- All events within a single activation share the same tick. `dm_read`'s first
  activation (tick=1) covers two reserve attempts, one push, and finally the block.
- `kernel_block` and `kernel_unblock` bound the stall: `dm_read` blocked at tick=1
  and unblocked at tick=3, meaning one other activation (tick=2) occurred while it
  waited.
- `dfb_reserve_begin` is always emitted when `reserve()` is entered, regardless of
  whether a block follows. The gap between `dfb_reserve_begin` and `dfb_reserve_end`
  on the same DFB is the stall for that specific acquire.
- The `occupied` field after each push/pop shows buffer occupancy evolving over time.

### Enabling tracing

Tracing is controlled by a `--trace` flag on `ttlang-sim`:

```
ttlang-sim examples/matmul.py --trace trace.jsonl
```

This enables tracing and writes the collected events to `trace.jsonl` on exit, in
the same way that `--show-stats` enables statistics and prints them. When no
`--trace` flag is given, `trace()` is a no-op and adds no overhead.

### Filtering

By default all event categories are collected. Two mutually exclusive flags control
which categories are written to the output file.

**Inclusive filter** -- collect only the listed categories:

```
ttlang-sim examples/matmul.py --trace trace.jsonl --trace-events dfb,copy
```

**Exclusive filter** -- collect everything except the listed categories:

```
ttlang-sim examples/matmul.py --trace trace.jsonl --no-trace-events dfb
```

Specifying both `--trace-events` and `--no-trace-events` in the same invocation is
an error. Specifying either without `--trace` is also an error.

The defined categories and the events they cover:

| Category | Events |
|----------|--------|
| `operation` | `operation_start`, `operation_end` |
| `kernel` | `kernel_start`, `kernel_end`, `kernel_block`, `kernel_unblock` |
| `dfb` | `dfb_reserve_begin`, `dfb_reserve_end`, `dfb_push`, `dfb_wait_begin`, `dfb_wait_end`, `dfb_pop` |
| `copy` | `copy_start`, `copy_end` |

The special value `all` is equivalent to omitting the filter entirely and is the
default when `--trace` is given without a filter flag.

Filtering is applied at collection time, not at export. Events in excluded categories
are never recorded, reducing both memory usage and output file size. Post-processing
tools can apply further filtering on the already-reduced dataset.

### Output format

Trace events are written as JSON Lines (JSONL): one JSON object per line, one line
per event. Each object has three fixed fields and zero or more event-specific fields
at the top level:

```
{"tick": 1, "kernel": "node0-dm_read",  "event": "kernel_start"}
{"tick": 1, "kernel": "node0-dm_read",  "event": "dfb_reserve_begin", "dfb": "dfb1"}
{"tick": 1, "kernel": "node0-dm_read",  "event": "dfb_reserve_end",   "dfb": "dfb1", "occupied": 1}
{"tick": 1, "kernel": "node0-dm_read",  "event": "dfb_push",          "dfb": "dfb1", "occupied": 1}
{"tick": 1, "kernel": "node0-dm_read",  "event": "dfb_reserve_begin", "dfb": "dfb1"}
{"tick": 1, "kernel": "node0-dm_read",  "event": "kernel_block",      "op": "reserve", "dfb": "dfb1"}
{"tick": 2, "kernel": "node0-compute",  "event": "kernel_start"}
{"tick": 2, "kernel": "node0-compute",  "event": "dfb_wait_begin",    "dfb": "dfb1"}
...
```

The extra fields are flat (not nested) so that tools do not need to unwrap a sub-object
to access them. JSONL is streamable and can be read line by line without loading the
full file into memory.

**Querying with `jq`**:

```bash
# All stall events for dfb1
jq 'select(.event == "kernel_block" and .dfb == "dfb1")' trace.jsonl

# Stall duration for each block/unblock pair (ticks blocked)
jq -s 'group_by(.kernel) | .[] |
  [.[] | select(.event == "kernel_block" or .event == "kernel_unblock")] |
  [range(0; length; 2) as $i | {kernel: .[$i].kernel, stall: (.[$i+1].tick - .[$i].tick)}]
  | .[]' trace.jsonl
```

---

## Replay and Schedule Fuzzing

### Replay

The simulator is deterministic: given the same inputs and the same scheduling order,
execution produces identical results. The scheduling order is fully recorded in the
trace: each event carries the `kernel` that was active at that tick, so the ordered
sequence of `(tick, kernel)` pairs is the complete schedule.

To extract the schedule from a trace:

```bash
jq -s '[.[] | select(.event == "kernel_start" or .event == "kernel_unblock") |
        {tick, kernel}]' trace.jsonl
```

A replay scheduler uses this sequence instead of the normal fair or greedy algorithm:
at each activation it runs the kernel specified in the recorded schedule. If the
recorded kernel is blocked (cannot proceed), the recorded schedule is no longer
valid for the current code -- the replay fails rather than proceeding silently with a
different order.

Replay is useful for:

- Reproducing a specific execution after adding instrumentation or print statements.
- Regression testing: verify that a code change does not alter the trace.
- Debugging: narrow a failure to a particular scheduling sequence and then re-run
  that exact sequence repeatedly.

### Schedule Fuzzing

Schedule fuzzing explores alternative scheduling orders to verify that a kernel
produces correct results regardless of how the scheduler interleaves its
threads.  The idea is that we can fuzz multiple schedules and if one of them
fails we could replay it for debugging purposes. The **choice points** for the
fuzzer is a scheduler activation at which more than one kernel is runnable. A
trace records one path through this choice tree

## Why No Tracing Framework

Several Python tracing libraries were considered and rejected:

**OpenTelemetry** is the closest match in terms of API design -- it has spans,
attributes, and exporters. However, it is built for distributed tracing across service
boundaries: its core abstractions (TracerProvider, Propagator, OTLP exporter, resource
attributes) address network context propagation that is irrelevant here. More
critically, OpenTelemetry propagates span context via Python's `contextvars`, which is
not greenlet-aware. The "current span" would be shared across all greenlets rather than
per-kernel, making spans attributed to the wrong kernel. Adapting around this
limitation would require more code than writing a custom system from scratch.

**viztracer** hooks into `sys.settrace` and records every Python function call
automatically. This is useful for profiling Python code, but produces traces dominated
by simulator internals. We want traces of operation-level semantic events (DFB operations,
copy calls) -- not the Python call graph of the simulator implementation itself.

**eliot** models structured actions
(equivalent to spans) and is the closest Python library to what we need conceptually.
However, it uses Python threading context for action nesting, not greenlet context,
so the "current action" would be shared across all greenlets rather than per-kernel --
the same fundamental problem as OpenTelemetry. Its output format is JSONL, which
requires a separate converter to produce structured timeline output. The context model
mismatch would require the same greenlet-specific workarounds as OpenTelemetry, at
which point eliot adds no benefit over a custom implementation.

**structlog** is a structured logging library: it adds bound key-value context to log
lines and supports processor chains and JSON output. It does not have a concept of
spans or durations -- there is no mechanism for recording that a DFB reserve started
at tick 5 and ended at tick 12. It uses `contextvars` for context binding, which has
the same greenlet-unawareness problem described above. structlog is the right tool for
structured diagnostic logs; it is not a tracing system.

**greenlet.settrace** provides callbacks on every greenlet switch, which is useful for
driving the tick counter. However it only signals that a switch happened -- not why.
Knowing whether a kernel blocked on a DFB `reserve` versus a `wait` versus a copy
requires instrumentation at the call sites regardless. `greenlet.settrace` may be used
as a secondary mechanism to drive the tick counter, but is not sufficient on its own.


More fundamentally, the simulator already has a first-class execution model and
execution context (`SimulatorContext`, `GreenletScheduler`, the greenlet-per-kernel
model). That context is the source of truth for what is happening at runtime.
Introducing a tracing framework's own ambient context alongside it -- span stacks,
context propagation chains, `contextvars` -- would duplicate this model and create
two competing sources of truth. The tracing system should read from the simulator's
own context, not maintain a parallel one.
