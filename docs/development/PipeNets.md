# PipeNets

This document describes how PipeNets are owned, validated, lowered, and
scheduled in tt-lang. Both the compiler and the simulator consume the
same operation-level PipeNet collection; this doc covers the data flow,
the PipeNet guard verification pass that catches missing user guards
before lowering, the simulator launch semantics, and the test coverage.

The launch grid (the grid that `@ttl.operation(grid=...)` schedules
onto) is decoupled from the *work extent* described by the user's
PipeNets — the per-axis bounding box of every pipe coordinate. The
`grid=` argument selects how the launch relates to the work extent:

- `grid="auto"` is an optimization that shrinks the launch to the work
  extent when the operation uses PipeNets, and resolves to the device
  compute grid otherwise. This is the default and minimizes wasted
  kernel launches.
- `grid="full"` always launches on the device compute grid. The user
  must guard pipe-coupled regions with `net.is_src()` / `net.is_dst()`
  / `net.is_active()` (or equivalent coordinate predicates) so that
  nodes outside the work extent skip the pipe-coupled work; the
  verifier rejects any pipe-coupled op that is reachable from a node
  outside its declared role.
- An explicit tuple is used verbatim; the verifier still requires
  guards on any pipe-coupled op reachable from a non-role node.

Under `grid="auto"` every launched node is active and the verifier's
checks are trivially satisfied. Under `grid="full"` (or an over-large
explicit tuple) the verifier rejects unguarded pipe-coupled ops with a
diagnostic that names the offending op, an example out-of-role
coordinate, the contributing PipeNet(s), and a suggested guard.

## Overview

`ttl.PipeNet` describes a logical communication pattern between nodes. A
pipe carries data from a source coordinate (`src`) to either a single
destination (unicast) or a contiguous coordinate range (multicast). When
the launch grid is larger than the union of all pipe sources and
destinations, the extra nodes have no role in the communication. If the
user fails to guard pipe-coupled work from those nodes, the kernel reads
out-of-bounds tensor regions and corrupts the multicast handshake
(issue #541).

The compiler verifies user-written guards: each pipe-coupled operation
must be reachable only from the nodes permitted by its role
(`ttl.copy(cb, pipe)` only from `pipe.src`; `ttl.copy(pipe, cb)` only
from `pipe.dst`; `cb_wait` reachable only within the static producer domain
for that DFB index). The verifier reads the IR and emits diagnostics;
it does not rewrite the program.

The soundness argument for the verifier is published as a
[gist](https://gist.github.com/brnorris03/5c969f4359fa895c9055c00659074f9d).

Three predicate ops — `ttl.is_src`, `ttl.is_dst`, `ttl.is_active`
(the union of source and destination roles) — let user code carry
per-PipeNet guards that the verifier recognizes structurally. Frontend
methods `net.is_src()`, `net.is_dst()`, `net.is_active()` lower to
these ops; coordinate comparisons over `ttl.node(dims=2)` against
integer constants also work and are evaluated per coord.

The verifier requires a `ttl.launch_grid` module attribute (an i64
array of length 2 with positive entries). The frontend stamps this
from the resolved grid; lit tests must declare it explicitly.

## Operation pipenets

`OperationPipeNets` (defined in `python/_pipenets/__init__.py`)
is the per-operation data structure the compiler and the simulator
both consume. It holds:

- A list of `PipeNetUse` entries, each with an operation-local id
  (`0..N-1`, reset per invocation) and a tuple of `PipeUse` records
  (source `NodeCoord`, destination `NodeCoord` for unicast or
  `NodeRange` for multicast).
- `validate()`: empty PipeNet, overlapping multicast destinations,
  mixed unicast/multicast within one PipeNet.
- `work_extent()`: per-axis bounding box of every pipe's source and
  destination coordinates.

The compiler and the simulator both discover PipeNets by walking the
closure cells of the operation function and each kernel function:
body-local PipeNets show up through the kernel functions' closures,
captured ones through the operation function's closure ([spec][spec-pipenet-scope]).

[spec-pipenet-scope]: https://github.com/tenstorrent/tt-lang/blob/<spec-commit>/docs/sphinx/specs/TTLangSpecification.md#L647

Operation-local ids keep `ttl.create_pipe` ids stable across
invocations and keep TTKernel semaphore allocation
(`pipeNetId * 2` / `pipeNetId * 2 + 1`) deterministic. The
`OperationPipeNets` instance is built and validated before MLIR
emission on the compiler side and before `Program(...)` runs on the
simulator side. `PipeNet.__init__` also builds a one-PipeNet
`OperationPipeNets` and runs the same `validate()` synchronously, so
malformed PipeNets error at the construction source line.

## Pass placement

```
... -> ttl-finalize-dfb-indices
    -> ttl-annotate-cb-associations
    -> ttl-verify-pipenet-guards                 (module-scoped)
    -> convert-ttl-to-ttkernel
    -> ttkernel-insert-inits
    ...
```

`ttl-verify-pipenet-guards` runs after DFB-index annotation
(`ttl-annotate-cb-associations`) so DFB wait checks can resolve producer
DFB indices. It runs before `convert-ttl-to-ttkernel` so
diagnostics surface in TTL IR with TTL-level op names (`ttl.copy`,
`ttl.cb_wait`, `ttl.is_src`, etc.) and so the structural marker
`ttl.pipenet_scope` (see below) can be inlined out of existence
before lowering.

Three independent pipeline definitions need to stay in sync: the C++
`createTTLToTTKernelPipeline` in `lib/Dialect/TTL/Pipelines/TTLPipelines.cpp`,
the Python frontend pipeline string in `python/ttl/ttl_api.py`, and the
me2e builder in `test/me2e/builder/pipeline.py`. All three insert the
verifier at the same anchor.

## Domain analysis

The verifier represents the set of nodes a region executes on as a
`Domain` — an explicit `std::set<Coord>` over the launch grid. This
enumeration is sufficient for current 2D grids (≤ ~200 nodes) and
avoids an upstream Presburger dependency. It also matches the
per-pipe role granularity the verifier needs: each PipeNet's source
domain is the union of its pipes' source unit boxes; its destination
domain is the union of its pipes' destination ranges.

Per-pipe role containment is the central check. For a pipe-coupled
op, the verifier computes the current execution domain (the
intersection of every enclosing predicate) and asserts it is a subset
of the role required by the op:

| Op | Required role |
| --- | --- |
| `ttl.copy(cb, pipe)` | `pipe.src` (single coord) |
| `ttl.copy(pipe, cb)` | `pipe.dst` (mcast range) |
| `ttl.if_src %pipe` body | `pipe.src` (op carries the predicate intrinsically) |
| `ttl.if_dst %pipe` body | `pipe.dst` (op carries the predicate intrinsically) |
| `cb_wait` on pipe-coupled DFB | union of producer domains across all `cb_push` to the same DFB index |

DFB wait checking is module-global: `cbProducerDomains` indexes by DFB index,
so a `cb_wait` in one kernel function is checked against `cb_push` domains
from a different kernel function (compute waits for data movement, etc.).
DFB indices are stable post-finalize.

## Predicate recognition

The walker computes the current domain by intersecting predicates as
it descends into nested regions. Recognized parents:

| Parent op | Contribution to current domain |
| --- | --- |
| `scf.if` then-branch | intersect with condition domain |
| `scf.if` else-branch | intersect with negated condition domain |
| `affine.if` then/else | per-coord IntegerSet evaluation |
| `ttl.if_src %pipe` body | intersect with `pipe.src` |
| `ttl.if_dst %pipe` body | intersect with `pipe.dst` |
| `ttl.pipenet_scope` body | unchanged after checking current domain is contained in declared role union |
| `scf.for`/`scf.while`/`affine.for`/region-bearing ops | unchanged (no predication) |

For `scf.if`, the condition's domain is determined structurally:

- `ttl.is_src %net` → that PipeNet's source domain.
- `ttl.is_dst %net` → that PipeNet's destination domain.
- `ttl.is_active %net` → union of source and destination domains.
- `arith.andi` / `arith.ori` decompose: each operand contributes its
  own domain (intersection or union). A coord-independent operand
  (loop iv, runtime flag) acts as identity instead of poisoning the
  result.
- Other coord-dependent expressions (`arith.cmpi` over arithmetic on
  `ttl.core_x` / `ttl.core_y`) are evaluated per coord.
- A coord-independent expression contributes the universe (uniform
  across the grid).
- Unanalyzable coord-dependent expressions make the branch execution
  domain unknown. Any PipeNet-coupled operation reached under that
  domain is rejected as unprovable.

## Diagnostics

Each violation produces one error with notes:

```
error: 'ttl.copy' op may copy outside source role of PipeNet 0
note: example node where the guard does not hold: core_x=1, core_y=0
note: PipeNet 0 declared here  (at create_pipe location)
note: suggested guard: wrap in `if net_0.is_src(): ...` or
      `net_0.if_src(lambda pipe: ...)`
```

`signalPassFailure()` is called once at the end so every site is
reported in a single run rather than failing on the first.

## `ttl.pipenet_scope`

A region op the frontend emits around DFB-context blocks
(`with cb.reserve()`) whose body contains pipe role work. It carries
two parallel attributes: `ttl.pipe_net_ids` (`DenseI64ArrayAttr`) and
`ttl.pipe_net_roles` (`DenseI64ArrayAttr`, one entry per id; 0 =
Source, 1 = Destination — `Active` is a *predicate* via
`ttl.is_active` and is not valid as a scope role). The verifier checks
that the scope's effective execution domain is a subset of the union
of declared role domains, then walks its body with the same incoming
domain because the scope has no runtime predicate. After verification
the verifier inlines and erases the scope so downstream lowering sees a
`pipenet_scope`-free IR.

The frontend emits the scope only around blocks whose context manager
is `reserve()`. A `wait()` block consumes a DFB filled by some other
thread and may sit unguarded next to ancillary pipe ops, so wrapping
it would over-constrain those ops to the wait's PipeNet roles. The
DFB wait check (verifier checks `cb_wait` against the union of
`cb_push` domains) catches static-domain mismatches the absent scope would
otherwise have flagged.

## Invariants

The verifier relies on these input properties.

| Invariant | Rationale |
| --- | --- |
| `ttl.launch_grid` module attribute present | Subset checks against an unbounded universe are meaningless. The pass emits a module-level error and fails if the attribute is missing. |
| `ttl.create_pipe` source/destination coordinates are static `I64Attr`s, encoded both on the op and in the result `PipeType` | Domain construction reads the attributes directly to materialize each pipe's source unit box and destination range as concrete `Coord` sets, and `PipeLowering.cpp` emits `arith.ConstantIndexOp` for each coordinate when building per-node role predicates. The static-attribute encoding is a property of today's IR, not a fundamental constraint of the verifier or lowering — see "Future work: parametric PipeNets" for the path to runtime-bound coordinates. |
| Pipe-coupled ops have stable DFB indices | DFB wait checks require `ttl-annotate-cb-associations` and `ttl-finalize-dfb-indices` to have run already. |
| One operation per module | The verifier walks all pipes in the module to compute role domains; co-compiling multiple operations would require per-operation scoping. |

## Multi-PipeNet operations

The verifier checks each pipe-coupled op against the role of *its
own* PipeNet, not against the union of all PipeNets' active sets.
A `ttl.copy(cb, %pipe_a)` reachable from a node that is in
`net_b.is_active()` but outside `net_a.src` is rejected with a
diagnostic that names `net_a`, not "the active set".

Two mechanisms together carry per-PipeNet correctness in user code
when an operation defines multiple PipeNets over different node groups:

1. `ttl.if_src %pipe { ... }` and `ttl.if_dst %pipe { ... }` carry
   their own per-node predicate: the inner block executes only when
   the current node matches that pipe's source or is in its
   destination range. Per-pipe data movement is therefore correctly
   conditional without any per-PipeNet wrapper.

2. Non-pipe work (dataflow-buffer reserves, compute, address
   arithmetic) is guarded by the user with explicit role-based
   predicates: `if net.is_src()`, `if net.is_dst()`,
   `if net.is_active()`, or coordinate comparisons over
   `ttl.node(dims=2)` against integer constants.

An example is `test_overlapping_pipenets`: two PipeNets with disjoint
source nodes and overlapping destination nodes, where the
data-movement kernel routes work by node coordinate:

```python
@ttl.datamovement()
def dm_read():
    x, _ = ttl.node(dims=2)
    if x == 0:                           # net_a source role
        with a_cb.reserve() as ablk:
            net_a.if_src(...)
    elif x == 3:                         # net_b source role
        with b_cb.reserve() as bblk:
            net_b.if_src(...)
    elif 1 <= x and x <= 2:              # destination role for both
        with a_cb.reserve() as ablk: net_a.if_dst(...)
        with b_cb.reserve() as bblk: net_b.if_dst(...)
```

## Simulator parity

The simulator does not run the MLIR verifier. It builds the
`OperationPipeNets` from the operation's closure plus each kernel
function's closure and validates it before execution. For `grid="auto"`,
the simulator uses `work_extent()` to choose the same launch-grid
bounding box as the compiler. For `grid="full"` and explicit tuple
grids, the simulator launches every node in the resolved grid.

The simulator does not skip nodes outside PipeNet roles. User guards
such as `net.is_active()` or coordinate predicates decide which nodes
execute pipe-coupled work, matching the compiler/runtime semantics.

## Example: 2D mcast matmul

A small mcast matmul with work shape M_BLOCKS=4, N_BLOCKS=3 launched
under `grid="full"` on a Wormhole device (8x7 grid). The launch
covers the entire device; the user wraps each pipe-coupled thread
body in `if net.is_active():` so the verifier accepts it and so
inactive nodes short-circuit at runtime.

```py
@ttl.operation(grid="full")
def small_mcast_matmul(a, w, out):
    a_pipes = [
        ttl.Pipe(src=(0, row), dst=(slice(0, 3), row))   # broadcast A row
        for row in range(4)
    ]
    a_net = ttl.PipeNet(a_pipes)
    b_pipes = [
        ttl.Pipe(src=(col, 0), dst=(col, slice(0, 4)))   # broadcast B col
        for col in range(3)
    ]
    ttl.PipeNet(b_pipes)

    @ttl.compute()
    def compute():
        if a_net.is_active():
            ...
    ...
```

Pipe sources contribute `{(0, 0), (0, 1), (0, 2), (0, 3), (0, 0), (1, 0),
(2, 0)}` and destinations contribute the rectangles `[0,3) x {row}` for
each row plus `{col} x [0,4)` for each col. `a_net.is_active()` covers
exactly `[0, 3) x [0, 4)`, twelve nodes; the remaining 8x7 - 12 = 44
launched nodes evaluate the predicate to `false` and skip the
pipe-coupled work.

## Test coverage

The same pytest file runs on hardware and on the simulator via
`test/scripts/ttlang-sim-pytest`, which patches `sys.modules` with the
simulator's `ttl` and `ttnn` before pytest collects, so hardware and
simulator coverage is the default for any test under `test/python/`.
Sim-only
tests under `test/sim/` are reserved for sim-internal helpers that have
no hardware analogue. Lit tests cover compile-time properties not
runtime-observable.

| #  | Behavior under test                                       | Dev | Sim | Lit |
|----|-----------------------------------------------------------|:---:|:---:|:---:|
|  1 | Empty PipeNet rejected at construction                    |  X  |  X  |     |
|  2 | Within-PipeNet mcast dst overlap rejected (full)          |  X  |  X  |     |
|  3 | Within-PipeNet mcast dst overlap rejected (partial)       |  X  |  X  |     |
|  4 | Unicast gather to same dst allowed                        |  X  |  X  |     |
|  5 | Nonoverlapping mcast pipes in one PipeNet allowed         |  X  |  X  |     |
|  6 | Pipe rejects open-bounded slices                          |  X  |  X  |     |
|  7 | Pipe rejects empty / inverted slices                      |  X  |  X  |     |
|  8 | Mixed unicast + multicast in one PipeNet rejected         |  X  |  X  |     |
|  9 | All-unicast PipeNet allowed                               |  X  |  X  |     |
| 10 | All-multicast PipeNet allowed                             |  X  |  X  |     |
| 11 | Pipe.src strict 2-tuple rejection                         |  X  | (2) |     |
| 12 | Scatter on subgrid (work < launch, single mcast)          |  X  |  X  |     |
| 13 | Per-row scatter (multi-pipe disjoint dst, 2D active set)  |  X  |  X  |     |
| 14 | Cross-PipeNet destination overlap permitted               |  X  |  X  |     |
| 15 | Loopback mcast (src in dst range)                         |  X  |  X  |     |
| 16 | Nested `if_src` / `if_dst` across two PipeNets (relay)    |  X  |  X  |     |
| 17 | Captured (closure) PipeNet works                          |  X  |  X  |     |
| 18 | Module-scope PipeNet works                                |  X  |  X  |     |
| 19 | Mixed scope: module-scope + body-local PipeNets in one op |  X  |  X  |     |
| 20 | 1D scatter (existing pattern)                             |  X  |  X  |     |
| 21 | 1D gather (existing pattern)                              |  X  |  X  |     |
| 22 | 1D gather, multiple tiles per source (existing)           |  X  |  X  |     |
| 23 | Ring forward (1D unicast +1, existing)                    |  X  |  X  |     |
| 24 | 2D broadcast (existing)                                   |  X  |  X  |     |
| 25 | Pipe chain / conv multi-stage (existing)                  |  X  |  X  |     |
| 26 | 1D mcast matmul auto-grid baseline (existing)             |  X  |  X  |     |
| 27 | Issue #541 regression: 4x3 work extent under grid="full"  |  X  |  X  |     |
| 28 | Issue #541 regression: 2x2 work extent under grid="full"  |  X  |  X  |     |
| 29 | 2D mcast matmul (work < launch via `_even_split`) [fixed] |  X  | (1) |     |
| 30 | Balanced 2D matmul (A on dm_read, B on dm_write) [fixed]  |  X  | (1) |     |
| 31 | Balanced 2D matmul + fused relu [fixed]                   |  X  |  X  |     |
| 32 | OperationPipeNets: src coord + dst range (mcast unit)     |     |  X  |     |
| 33 | OperationPipeNets: union across PipeNets                  |     |  X  |     |
| 34 | OperationPipeNets: unicast pipe single dst                |     |  X  |     |
| 35 | OperationPipeNets: None when empty                        |     |  X  |     |
| 36 | OperationPipeNets: validate empty PipeNet                 |     |  X  |     |
| 37 | OperationPipeNets: validate overlapping mcast             |     |  X  |     |
| 38 | OperationPipeNets: operation-local id allocation          |     |  X  |     |
| 39 | sim pipe deadlock detection (existing)                    |     |  X  |     |
| 40 | Verifier accepts `if net.is_src/is_dst/is_active()` guards |    |     |  X  |
| 41 | Verifier accepts coordinate-compare guards over `core_x`/`core_y` |     |     |  X  |
| 42 | Verifier accepts `affine.if` guards via IntegerSet eval   |     |     |  X  |
| 43 | Verifier accepts `pipenet_scope` and inlines it post-check |     |     |  X  |
| 44 | Verifier rejects `ttl.copy(cb, pipe)` outside source role |     |     |  X  |
| 45 | Verifier rejects `ttl.copy(pipe, cb)` outside destination role |  |     |  X  |
| 46 | Verifier rejects `cb_wait` with no producer domain coverage |   |     |  X  |
| 47 | Verifier names per-PipeNet role in cross-net diagnostics  |     |     |  X  |
| 48 | `CreatePipeOp::verify` rejects `dstStart > dstEnd` (x)    |     |     |  X  |
| 49 | `CreatePipeOp::verify` rejects `dstStart > dstEnd` (y)    |     |     |  X  |
| 50 | Verifier rejects unanalyzable predicates with location note |   |     |  X  |
| 50a| Verifier rejects missing `ttl.launch_grid` module attribute |   |     |  X  |
| 50b| Pipeline lit confirms `pipenet_scope` is gone post-verifier |   |     |  X  |
| 51 | OperationPipeNets.work_extent: empty / unicast / mcast    |     |  X  |     |
| 52 | OperationPipeNets.work_extent: union, mixed-rank padding  |     |  X  |     |
| 53 | grid="auto" shrinks launch + grid="full" keeps launch     |  X  |  X  |     |

(1) Device-only due to a pre-existing simulator divergence orthogonal
to PipeNet verification: the simulator's block-state machine accepts
in-place `+=` only on a *temporary* block (the result of a `fill` or
a block expression), not on a dataflow-buffer block that has already
been written via `store(...)`. Hardware accepts both. The matmul
kernels in these tests use `out_blk += a @ b` after an initial
`out_blk.store(fill(...))`, which the simulator rejects.

(2) Hardware-only by design. The hardware-side `ttl.Pipe.src` is
strictly `Tuple[int, int]` (the dialect is 2D), but the simulator's
`Pipe.src` accepts 1D coordinates because the existing
`matmul_1d_mcast` example uses them. The test pins the hardware-side
rejection contract; it `pytest.skip`s on the simulator runner.

## Limitations

* Work larger than launch: the verifier checks role containment but
  does not add nodes or split work. Operations that distribute more
  work than launched nodes via per-node block tiling (e.g. `_even_split`
  in `test_mcast_matmul.py`) are unaffected when every launched node
  appears in the active set.
* Typos in pipe coordinates change role domains. An operation whose
  pipe writes `dst=(slice(0, 5), 0)` instead of `dst=(slice(0, 4), 0)`
  has a one-node larger destination domain, and that extra node will
  be accepted by the verifier even if the user did not intend it. The
  domains are exactly what the PipeNet says, no more.
* The verifier does not constrain non-pipe work. Under `grid="full"`,
  nodes outside any PipeNet role may still execute compute, plain
  DFB pushes, or other SPMD-over-the-full-device work. Only ops
  coupled to a PipeNet (pipe-typed copies, pipe-coupled DFB waits,
  `if_src` / `if_dst` bodies) require role containment.
* Domain representation is `std::set<Coord>` over the launch grid.
  Sufficient for current 2D grids (≤ ~200 nodes); revisit when grids
  grow to 3D or thousands of nodes.
* Three pipeline definitions: the verifier is wired into three
  separate strings (C++ pipeline, Python frontend, me2e builder). A
  future refactor consolidating these would prevent future passes from
  drifting between them.

## Future work

* Issue #505: lift the within-PipeNet multicast destination overlap
  restriction. Today a single PipeNet shares one semaphore pair across
  all its pipes, so a node receiving from two multicast sources cannot
  disambiguate the handshake. Per-source semaphore increments via
  `noc_semaphore_inc_multicast` in TTKernel would let one PipeNet
  describe true scatter-gather and all-to-all patterns. This is a
  TTKernel dialect + tt-metal change; it is unrelated to PipeNet
  guard verification, but unblocking it would let `test_scatter_gather` and a
  single-PipeNet all-to-all version of `test_overlapping_pipenets` come
  off `@pytest.mark.skip`.
* Cross-chip (Galaxy / QuietBox / N300) PipeNets. tt-lang's
  `@ttl.operation` is a per-chip program by contract today; PipeNet
  coordinates are interpreted by the NoC, so they always refer to
  nodes on a single chip. Users running on Galaxy already do so by
  composing per-chip operations and handling cross-chip data movement
  outside tt-lang (typically via ttnn CCL ops over the `tt_fabric`
  layer). There is no language construct for "this pipe crosses to
  chip (i, j)"; adding one is a language extension, not a free
  behavior change in the lowering. A future cross-chip PipeNet would
  introduce an explicit inter-chip pipe variant (e.g. carrying a
  `MeshCoordinate` for source and destination) that lowers to fabric
  ops alongside the existing intra-chip lowering. The
  `OperationPipeNets` data structure is small enough to grow that
  variant without affecting today's intra-chip path. Verifier
  bound-checking against the operation's grid extent (still future
  work) would also reject out-of-chip coordinates that today silently
  miscompile.
* If multiple operations are ever co-compiled into one module, scope
  the verifier walk to the enclosing operation by a marker attribute or
  by using a per-operation pass driver.
* `CreatePipeOp` verifier could additionally bound-check coordinates
  against the device grid extent (the `dstStart <= dstEnd` ordering is
  already enforced).
* For larger grids (3D, thousands of nodes) the explicit
  `std::set<Coord>` representation should be replaced with a Presburger
  set or axis-aligned rectangle set so domain operations stay
  tractable.
* **Parametric PipeNets** — runtime-bound pipe coordinates resolved at
  kernel-launch time rather than `@ttl.operation` decoration time. The
  current pipeline resolves `ttl.Pipe(src=..., dst=...)` arguments to
  Python `int` / `slice` literals during frontend tracing, materializes
  them as `I64Attr`s on `ttl.create_pipe`, and bakes them into the
  result `PipeType`. A parametric variant requires three coordinated
  changes:
  1. **IR.** Extend `ttl.create_pipe` with an alternative form whose
     source/destination coordinates are SSA `index` operands rather
     than attributes, and replace the static coordinate fields on
     `PipeType` with a static *bounding-box* attribute (so the
     verifier and downstream passes still have a coarse-grained type
     invariant). The static form remains the lowering target for
     `@ttl.operation` invocations whose coordinates are known at
     trace time.
  2. **Verifier.** Replace the `std::set<Coord>` `Domain` with a
     symbolic representation — either an upstream Presburger set
     (`mlir::presburger::IntegerRelation`) or a structured
     axis-aligned-rectangle set with parametric bounds — and recast
     `pipeSourceDomain` / `pipeDestinationDomain` /
     `getBranchDomains` to produce symbolic constraints over the
     pipe's coordinate operands and the launch-grid extents. Per-pipe
     role containment then becomes a Presburger emptiness check
     (`current ∩ ¬role` is empty) parameterized by the static bounds.
     The `ttl.is_src` / `ttl.is_dst` / `ttl.is_active` recognition
     stays structural; the per-coord enumeration in `evalBool`
     becomes a constraint constructor.
  3. **Lowering.** `PipeLowering.cpp` emits `arith.ConstantIndexOp` at
     lines 363, 391, 437, 545 for source / destination coordinates
     today. Threading SSA values through to `noc_async_write_multicast`
     and the per-pipe match expressions is mechanical: tt-metal's
     multicast NoC primitives already accept runtime coordinates, and
     `IsSrcLowering` / `IsDstLowering` already construct per-pipe
     `arith.cmpi` / `arith.andi` / `arith.ori` chains over the pipe's
     coordinate values — they currently chain against constants but
     would chain against the SSA operands instead.

  Frontend surface: `ttl.Pipe(src=ttl.runtime_arg("M"), ...)` or a
  similar SSA-typed coordinate, with the `OperationPipeNets`
  data structure carrying static bounds plus a record of which axes
  are runtime-resolved. `grid="auto"` shrinks to the static bounding
  box rather than the resolved work extent. The `@ttl.operation`
  caching key includes the bounds (not the runtime values), so a
  single compiled kernel covers every invocation that fits the
  declared bounds.

  Out of scope for parametric PipeNets: per-iteration dynamic routing
  decided inside a kernel function. The TTKernel multicast handshake
  allocates one semaphore pair per PipeNet at kernel compile time
  (`pipeNetId * 2` / `pipeNetId * 2 + 1`) and reconfiguring an mcast
  group mid-kernel is not a tt-metal-supported operation; data-
  dependent routing would be expressed as point-to-point unicast with
  runtime destination, not as a PipeNet.
