#ifndef TTLANG_DIALECT_TTL_PASSES_TD
#define TTLANG_DIALECT_TTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def TTLInsertCBSync
    : Pass<"ttl-insert-cb-sync", "::mlir::func::FuncOp"> {
  let summary = "Insert missing cb_push/cb_pop for unmatched cb_reserve/cb_wait";
  let description = [{
    Finds `ttl.cb_reserve` ops without a matching `ttl.cb_push` and
    `ttl.cb_wait` ops without a matching `ttl.cb_pop`, then inserts the
    missing release op after the last operation that accesses the CB's data.

    The pass computes a transitive use closure starting from the acquire op:
    direct uses of the DFB value and the acquire result, plus uses of values
    those ops produce (e.g. copy -> transfer_handle -> wait). Direct DFB uses
    are bounded by the next acquire in the same DFB sync class (`cb_reserve`
    for producer intervals, `cb_wait` for consumer intervals) because they do
    not carry an SSA link to the acquire that owns the slot.

    Explicit push/pop calls and context-manager-generated push/pop are
    preserved: the pass only fills in what is missing.

    A receive request used by `ttl.wait_any` has conditional completion. Its
    destination reservation is not published automatically because only the
    selected request is known to be complete. Publication must be explicit and
    conditioned on the corresponding selection, or follow an exact wait.
  }];

  let dependentDialects = [];
}

def TTLCoalesceDFBAcquires
    : Pass<"ttl-coalesce-dfb-acquires", "::mlir::func::FuncOp"> {
  let summary = "Coalesce consecutive same-DFB acquires into one multi-tile acquire";
  let description = [{
    Rewrites the source pattern

    ```
    t1 = cb.wait();  t2 = cb.wait();  t3 = cb.wait();
    use(t1);         use(t2);         use(t3);
    ```

    into the canonical tt-metal "cumulative wait + indexed reads +
    coalesced pop" pattern

    ```
    cb_wait_front(cb, 3*k);
    copy_tile(cb, /*src_idx=*/0, dst);  // t1
    copy_tile(cb, /*src_idx=*/k, dst);  // t2
    copy_tile(cb, /*src_idx=*/2*k, dst);// t3
    cb_pop_front(cb, 3*k);
    ```

    used in tt-metal compute kernels (`eltwise_binary.cpp`, `bcast_h.cpp`,
    matmul kernels, etc.) when a consumer holds a fixed multi-tile window
    before processing it.

    Pre-coalesce, each `ttl.cb_wait` lowers to its own `cb_wait_front(k)`
    and each `ttl.cb_pop` to `cb_pop_front(k)`. Because metal's
    `cb_wait_front`/`cb_pop_front` are non-cumulative, this races: the
    first pop advances the front before the producer has pushed enough
    tiles to satisfy the next read.

    At the IR level the rewrite is

    ```mlir
    // Before (post `ttl-insert-cb-sync`):
    %t1 = ttl.cb_wait %cb : <[1,k],...,bc> -> tensor<1xkx...>
    %t2 = ttl.cb_wait %cb : <[1,k],...,bc> -> tensor<1xkx...>
    %t3 = ttl.cb_wait %cb : <[1,k],...,bc> -> tensor<1xkx...>
    ... use %t1 ... ttl.cb_pop %cb
    ... use %t2 ... ttl.cb_pop %cb
    ... use %t3 ... ttl.cb_pop %cb

    // After:
    %g  = ttl.cb_wait %cb {num_tiles = 3*k} : ... -> tensor<1x(3k)x...>
    %t1 = tensor.extract_slice %g [0, 0]    [1, k] [1, 1]
    %t2 = tensor.extract_slice %g [0, k]    [1, k] [1, 1]
    %t3 = tensor.extract_slice %g [0, 2*k]  [1, k] [1, 1]
    ... use %t1 ... use %t2 ... use %t3 ...
    ttl.cb_pop %cb {num_tiles = 3*k}
    ```

    `addSliceOffset` already folds each `extract_slice` offset into the
    per-tile `src_idx` at lowering time, so no lowering changes are
    required.

    The producer side (`cb_reserve` / `cb_push`) is rewritten by the same
    templated routine: a run of `N` consecutive `cb_reserve` ops on one
    DFB collapses into one `cb_reserve {num_tiles = N*k}` plus per-block
    `tensor.extract_slice` views routed into the downstream
    `ttl.tile_store` / `ttl.store` ops, and the `N` matching `cb_push`
    ops collapse into one `cb_push {num_tiles = N*k}`.

    Detection: forward walk per block; an acquire group is a maximal run
    of same-kind same-DFB acquires separated only by `ttl.attach_cb` or
    `arith.constant` ops. Acquires already carrying a `num_tiles`
    attribute (e.g. set by `ttl-subblock-compute-for-dst`) are not
    coalesced and terminate the group. The pass is idempotent.

    Result-type constraint: only acquires whose result tensor has shape
    `<1, k>` are coalesced; the coalesced result is `<1, N*k>`. Other
    ranks would require a convention for which axis to extend by `N`,
    which is not specified, and are skipped.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect"
  ];
}

def TTLFormPipeTransports
    : Pass<"ttl-form-pipe-transports", "::mlir::ModuleOp"> {
  let summary = "Group repeated PipeNet transfers";
  let description = [{
    Groups repeated PipeNet transfers into multi-block transport operations.
    The pass runs after DFB synchronization is explicit. It expands high-level
    pipe copies, proves that one static `scf.for` contains the complete source
    and receiver protocol, and strip-mines that loop into a grouped loop plus a
    scalar residual loop.

    A grouped loop advances its induction variable by `R`, marks each
    `ttl.pipe_transfer.create` with `block_span = R`, and widens DFB lifecycle
    operations and contiguous tensor copies to `R` consecutive original DFB
    blocks. The original DFB block shape remains unchanged. The pass changes
    DFB block counts only after proving that every SSA use belongs to the
    transport. Later PipeTransport planning replaces each proven-private
    grouped lifecycle with transport-owned scratch. Scalar residuals continue
    to use the original DFB lifecycle.

    `group-size = 0` prefers a bounded schedule with at least two receiver
    groups, then minimizes completion groups, capacity waits, and L1 allocation
    bytes. Selection uses a target-aligned logical DFB estimate and a
    conservative upper bound for PipeNet address storage, scratch,
    GlobalSemaphore counters, record-selected callback resources,
    synchronized-reset scratch, and DFB reconfiguration state. This estimate
    selects a grouping size; it does not reject the finalized physical DFB
    allocation. `group-size = 1` disables grouping and records the conservative
    PipeNet reservation for physical DFB allocation. A larger value is an upper
    bound; the pass selects a smaller group when the loop trip count or L1
    capacity requires it. Ineligible loops retain scalar transfer IR. After
    PipeNet lowering chooses exact resources, `convert-ttl-to-ttkernel`
    validates the authoritative combined per-core allocation.
  }];

  let options = [
    Option<"groupSize", "group-size", "int64_t", "0",
           "Maximum logical transfers per group. 0 selects automatically; "
           "1 disables grouping.">,
    Option<"l1BudgetOverride", "l1-budget-override", "uint32_t", "0",
           "Override the combined per-core L1 budget in bytes for "
           "transport selection.">
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def TTLInsertCopyWait
    : Pass<"ttl-insert-copy-wait", "::mlir::func::FuncOp"> {
  let summary = "Complete ttl.copy operations without a guaranteed wait";
  let description = [{
    Proves copy completion across every reachable continuation, including
    structured regions and block control flow. An uncovered copy receives an
    implicit `ttl.wait`. When a request participates in a multi-request
    `ttl.wait_any`, uncovered cleanup occurs at each reachable function exit.
    The inserted wait therefore cannot alter the selected request or precede
    work that completes pending candidates. A single-request `ttl.wait_any`
    completes its sole request.
  }];

  let dependentDialects = [];
}

def TTLFormAccumulationScopes
    : Pass<"ttl-form-accumulation-scopes", "::mlir::func::FuncOp"> {
  let summary = "Form semantic accumulation scopes around tensor recurrences";
  let description = [{
    Wraps loop-carried additive tensor recurrences in
    `ttl.accumulation_scope` so later passes can lower the recurrence from an
    explicit accumulation policy instead of recovering it from ordinary tensor
    stores.

    Before:

    ```mlir
    %result = scf.for %iv = %lb to %ub step %step
        iter_args(%acc = %init) -> (tensor<...>) {
      %contribution = ...
      %next = ttl.add %acc, %contribution : tensor<...>
      scf.yield %next : tensor<...>
    }
    ttl.store %result, %out : tensor<...>, tensor<...>
    ```

    After:

    ```mlir
    ttl.accumulation_scope outs(%out : tensor<...>)
        inits(%init : tensor<...>) {
    ^bb0(%scope_acc: tensor<...>):
      %result = scf.for %iv = %lb to %ub step %step
          iter_args(%acc = %scope_acc) -> (tensor<...>) {
        %contribution = ...
        %next = ttl.add %acc, %contribution : tensor<...>
        scf.yield %next : tensor<...>
      }
      ttl.store %result, %out : tensor<...>, tensor<...>
      ttl.yield %result : tensor<...>
    } initial_modes([init])
    ```

    The tensor form requires an init-backed recurrence, one final
    non-accumulating store to a dataflow buffer reservation, and at least one
    legal lowering for the requested accumulation strategy.
  }];

  let options = [
    Option<"strategy", "strategy", "std::string", "\"auto\"",
           "Tensor recurrence accumulation strategy used to filter scopes: auto, dst, or l1-pack.">
  ];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLMaterializeLoopState
    : Pass<"ttl-materialize-loop-state", "::mlir::func::FuncOp"> {
  let summary = "Materialize loop-carried tensor state";
  let description = [{
    Rewrites ranked tensor `scf.for` iter args into explicit TTL dataflow
    buffer state. The pass removes tensor results from the loop while
    preserving non-tensor iter args.

    Additive tensor recurrences are handled earlier by
    `ttl-form-accumulation-scopes` and `ttl-lower-accumulation-scopes`.
    Any tensor recurrence that reaches this pass uses compiler-allocated
    dataflow buffer state:
    ```mlir
    %state = scf.for %iv = %lb to %ub step %step
        iter_args(%arg = %init) -> (tensor<1x1x!ttcore.tile<32x32, bf16>>) {
      %next = ttl.relu %arg : ...
      scf.yield %next : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ```

    becomes:
    ```mlir
    %state_cb = ttl.bind_cb ... {ttl.compiler_allocated}
    %init_reserve = ttl.cb_reserve %state_cb : ...
    ttl.store %init, %init_reserve : ...
    scf.for %iv = %lb to %ub step %step {
      %current_wait = ttl.cb_wait %state_cb : ...
      %current = ttl.attach_cb %current_wait, %state_cb : ...
      %next = ttl.relu %current : ...
      %next_reserve = ttl.cb_reserve %state_cb : ...
      ttl.store %next, %next_reserve : ...
      scf.yield
    }
    %final_wait = ttl.cb_wait %state_cb : ...
    %final = ttl.attach_cb %final_wait, %state_cb : ...
    ```
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLInsertAccumulationScopes
    : Pass<"ttl-insert-accumulation-scopes", "::mlir::func::FuncOp"> {
  let summary = "Insert semantic accumulation scopes";
  let description = [{
    Inserts `ttl.accumulation_scope` regions around user-written DFB
    accumulation after DFB acquire/release structure is canonical and before
    concrete storage strategy selection.

    The pass records accumulation policy only. It does not select DST, L1
    packer accumulation, or explicit dataflow buffer state.
  }];

  let options = [
    Option<"kind", "kind", "std::string", "\"dfb\"",
           "Scope insertion kind. Supported value: dfb.">
  ];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLLowerAccumulationScopes
    : Pass<"ttl-lower-accumulation-scopes", "::mlir::func::FuncOp"> {
  let summary = "Lower semantic accumulation scopes";
  let description = [{
    Lowers `ttl.accumulation_scope` regions to a concrete storage strategy.
    Tensor mode supports additive tensor recurrences formed by
    `ttl-form-accumulation-scopes` and stateful tensor scopes that lower to
    explicit stores in `strategy=auto`. DFB mode lowers user-written
    accumulating stores to L1 packer metadata.
    L1-selected loops carry `ttl.l1_acc_initial` and `ttl.l1_acc_scope_id`
    metadata for TTKernel packer reconfiguration placement.

    For additive tensor scopes, `strategy=auto` compares legal DST and L1
    packer accumulation candidates with the accumulation cost model.
    `strategy=dst` requires DST and emits an error when the tensor scope is not
    DST-compatible. `strategy=l1-pack` always uses the L1 packer strategy.
    Stateful tensor scopes lower only in `strategy=auto`.
    For f32 recurrences the DST strategy feeds the accumulator back through SRCA
    each iteration, truncating to tf32; `l1-pack` accumulates in f32 L1 and
    retains full f32 precision.
    DFB scopes always lower to L1 packer metadata because their storage target
    is the reserved output dataflow buffer slot.
  }];

  let options = [
    Option<"kind", "kind", "std::string", "\"tensor\"",
           "Scope lowering kind: tensor or dfb.">,
    Option<"strategy", "strategy", "std::string", "\"auto\"",
           "Tensor recurrence accumulation strategy: auto, dst, or l1-pack.">
  ];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLConvertTTLToTTKernel
    : Pass<"convert-ttl-to-ttkernel", "::mlir::ModuleOp"> {
  let summary = "Lower TTL data movement, PipeNet, synchronized-reset, and DFB reconfiguration operations to TTKernel";
  let description = [{
    Converts TTL DFB, copy, wait, PipeNet, synchronized-reset, and DFB
    reconfiguration operations to TTKernel DFB, NoC, semaphore, and external
    interface operations. PipeNet lowering selects receiver-published or
    computed destination addresses and receiver-post, capacity-counter, or
    fabric synchronization according to the pass options, transfer eligibility,
    and logical-device transfer.
    PipeNet counters use local semaphore ids before GlobalSemaphore storage by
    default; the global-only option preserves all local semaphore ids for
    application use.

    The pass validates finalized DFB backing plus the exact PipeNet and
    synchronized-reset scratch and DFB reconfiguration state against the
    selected per-core L1 budget before lowering the operations.

    PipeNet IR must first pass `ttl-verify-pipenet-guards` and
    `ttl-verify-pipenet-schedule`. Those passes establish endpoint containment,
    exact launch-node domains, and single-block event regions required by
    protocol planning. After planning, this pass validates the authoritative
    per-core L1 footprint: target-aligned finalized DFB allocations,
    synchronized-reset state, reconfiguration tensors, PipeNet scratch, and
    GlobalSemaphore allocations.
  }];

  let options = [
    Option<"reduceFullFp32", "reduce-full-fp32", "bool", "true",
           "Enable FP32 accumulation for reduce operations.">,
    Option<"pipeComputedAddresses", "pipe-computed-addresses",
           "bool", "true",
           "Use computed receiver DFB addresses for eligible pipe transfers. "
           "When disabled, pipe transfers use receiver-published destination "
           "addresses; multicast still requires proven equal runtime "
           "receiver addresses.">,
    Option<"pipeCapacitySync", "pipe-capacity-sync",
           "bool", "true",
           "Use capacity-counter synchronization when the receiver wait and "
           "pop execute on the receiver NOC thread and the computed-address "
           "transfer passes the DFB ownership and count proofs. "
           "When disabled, computed-address transfers use receiver-post "
           "synchronization.">,
    Option<"pipeGlobalSemaphoresOnly", "pipe-global-semaphores-only",
           "bool", "false",
           "Allocate all compiler-managed PipeNet synchronization counters "
           "in GlobalSemaphore storage, leaving local hardware semaphore ids "
           "available to the application.">,
    Option<"l1BudgetOverride", "l1-budget-override", "uint32_t", "0",
           "Override the per-core L1 budget used for finalized DFB and "
           "scratch validation. 0 selects the target budget.">
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTKernelInsertL1Accumulation
    : Pass<"ttkernel-insert-l1-accumulation", "::mlir::ModuleOp"> {
  let summary = "Insert packer L1 accumulation reconfiguration";
  let description = [{
    Inserts `pack_reconfig_l1_acc` operations so that pack operations inside
    annotated loops accumulate into L1 instead of overwriting. Max-reduce loops
    are excluded because max is not additive.
    Annotated additive loops must pack accumulated outputs to formats supported
    by packer L1 accumulation: f32, f16, bf16, si32, or u8. Packs inside the
    loop that target other dataflow buffers are bracketed with temporary
    disable and restore operations because the packer L1 accumulation state is
    global.

    Loops are grouped into accumulation scopes by `ttl.l1_acc_scope_id`;
    the pass does not infer scope identity from neighboring TTKernel
    operations. Nested annotated loops are folded into the outermost ancestor
    with the same scope id. Each group gets:
    - `pack_reconfig_l1_acc(0)` before the first loop when
      `ttl.l1_acc_initial=overwrite`, or `pack_reconfig_l1_acc(1)` when
      `ttl.l1_acc_initial=accumulate_existing`.
    - `if (iv == lb) pack_reconfig_l1_acc(1)` after each loop's first
      iteration for overwrite-mode loops (the L1 acc state persists across
      `tile_regs` boundaries).
    - For the second+ loop in a group, an enable before the loop to re-enable
      L1 acc after init ops that may reset packer state.
    - `pack_reconfig_l1_acc(0)` before non-accumulating packs in the loop and
      restore operations after those packs when the scope remains active.
    - `pack_reconfig_l1_acc(0)` after the scope when no later pack can observe
      the enabled state, or before a later non-group pack.

    Loops are identified by the `ttl.l1_acc_loop` (user-written) or
    `ttl.reduction_loop` (compiler-generated) attributes on `scf.for`
    ops, with `ttl.l1_acc_loop` taking precedence. Each loop must carry
    `ttl.l1_acc_initial` and `ttl.l1_acc_scope_id` so this pass does not infer
    semantic accumulation state from neighboring TTKernel operations.

  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTLInsertIntermediateDFBs
    : Pass<"ttl-insert-intermediate-dfbs", "::mlir::func::FuncOp"> {
  let summary = "Insert intermediate dataflow buffers for tensor materialization";
  let description = [{
    Walks ops implementing `DFBInputOpInterface` and checks whether the
    operands they declare as requiring DFB-attached values are satisfied.
    It also records direct `ttl.store` operands when one computed SSA value is
    stored by operations in multiple MLIR basic blocks. A `ttl.compute`
    operation is located in one MLIR basic block, and all output stores in its
    body execute whenever the operation executes. Moving stores from different
    MLIR basic blocks into one compute could therefore change their conditional
    execution or violate SSA dominance. The pass materializes the computed
    value once in a compiler DFB; each original store reads the attached value
    and remains in its original MLIR basic block.

    When an operand comes from `ttl.compute`, the pass materializes it as an
    extra compute output backed by a compiler-allocated DFB. The pass creates
    the DFB, reserves the output slot, clones the compute with an extra
    `ttl.tile_store`, emits the producer-side `cb_push`, and rewrites the
    consumer operand to the attached waited value:

      bind_cb -> cb_reserve -> ttl.compute tile_store -> cb_push -> cb_wait -> attach_cb

    Non-compute producers use the tensor-level helper. It emits the reserve,
    store, wait, and attach; `ttl-insert-cb-sync` later inserts the matching
    push and pop to complete this lifecycle:

      bind_cb -> cb_reserve -> store -> cb_push -> cb_wait -> attach_cb -> cb_pop

    Before modifying IR, the pass computes a whole-kernel analysis state of
    exact consumer operands that require DFB materialization. Decisions grow
    monotonically to a fixed point and retain independent evidence. This
    follows the One-Shot Bufferize model: analysis records every operand
    decision first, then a separate immutable plan groups those decisions by
    producer and applies them without repeating policy.

    MLIR dense forward dataflow proves DFB-backed inputs available at their
    eventual compute positions across CFG and structured-region edges.
    Any ambiguous acquisition ownership, partial release, or unrecognized
    storage identity is treated as possibly released and may require an
    additional materialization. A fusion boundary at an independently formable
    producer materializes the exact consumer operand. Multiple reserve
    transactions for one result, and direct stores of one value in multiple
    blocks, materialize all relevant uses through one shared compiler DFB.
    Application begins only after the complete plan is valid.

    In the standard pipeline this pass runs after
    `ttl-create-producer-compute`, so tensor-level materialization is limited
    to IR forms that have not yet been converted to `ttl.compute`.

    All DFBs created by this pass are marked with `ttl.compiler_allocated`;
    `ttl-finalize-dfb-indices` assigns physical indices and emits runtime
    metadata after materialization is complete.

    When `enable` is false, the pass verifies that no operations require
    compiler-allocated DFBs for operand attachment, source-lifetime
    preservation, or stores in multiple blocks. If any operation requires
    materialization, the pass emits an error diagnostic and signals failure.
  }];

  let options = [
    Option<"enable", "enable", "bool", "true",
           "Insert compiler-allocated intermediate DFBs. When false, "
           "emit an error if any operation requires a compiler-allocated DFB.">
  ];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect"
  ];
}

def TTLPrintDFBValueLifetimes
    : Pass<"ttl-print-dfb-value-lifetimes", "::mlir::func::FuncOp"> {
  let summary = "Print DFB lifetime analysis for debugging and tests";
  let description = [{
    This diagnostic pass prints the immutable acquisition, release-ownership,
    transfer-count, and program-point availability facts used by `ComputeOp`
    creation and intermediate DFB materialization. It supports debugging and
    deterministic regression tests; IDs follow kernel walk order rather than
    process addresses. The pass emits text to stderr and does not modify IR.
  }];

  let dependentDialects = [];
}

def TTLPrintComputeOpCreationPlans
    : Pass<"ttl-print-compute-op-creation-plans", "::mlir::func::FuncOp"> {
  let summary = "Print ComputeOp creation plans for debugging and tests";
  let description = [{
    This diagnostic pass prints the immutable ComputeOp creation recipes,
    dependency-safe application order, overlap-preserving uses, unassigned
    output stores, and intermediate DFB materialization requirements. Each
    materialization includes the evidence that requires the consumer operand
    to be materialized. It supports debugging and deterministic regression
    tests; IDs follow kernel walk order rather than process addresses. The pass
    emits text to stderr and does not modify IR.
  }];

  let dependentDialects = [];
}

def TTLProducerComputeCreation
    : Pass<"ttl-create-producer-compute", "::mlir::func::FuncOp"> {
  let summary = "Create ttl.compute ops before compiler DFB materialization";
  let description = [{
    Builds an immutable kernel plan before applying any rewrite. The plan
    records typed tile recipes, input lifetimes, iteration semantics,
    result-use ordering, output transactions, and dependency-safe application
    order. Operations whose plans satisfy every condition are converted to
    `ttl.compute`. Operations with rejected plans are not transformed, so
    `ttl-insert-intermediate-dfbs` can materialize the required intermediate
    values.

    When a block-level store is absorbed into a producer compute, this pass
    relocates any corresponding publication that would otherwise precede the
    new compute. The resulting order is:

      cb_reserve -> ttl.compute tile_store -> cb_push
  }];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTLConvertTTLToCompute
    : Pass<"convert-ttl-to-compute", "::mlir::func::FuncOp"> {
  let summary = "Lower planned tensor operations to ttl.compute";
  let description = [{
    Builds a complete immutable plan before modifying IR, then lowers selected
    tensor operations to `ttl.compute` with recorded tile recipes. The plan
    includes direct and fused inputs, affine indexing, iterator semantics,
    side-effect placement, output reserve/publication transactions, and
    dependency-safe application order. Before modifying IR, final conversion
    verifies that every `ttl.store` will either be absorbed into a planned
    `ttl.compute` or converted to a DFB-to-DFB passthrough. The pass fails if
    neither conversion applies. `ttl-create-producer-compute` may leave such
    stores for `ttl-insert-intermediate-dfbs` to resolve.
  }];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTLAssignDST
    : Pass<"ttl-assign-dst", "::mlir::func::FuncOp"> {
  let summary = "DST allocation with copy insertion, interval analysis, and linear scan";
  let description = [{
    DST register allocator using linear scan allocation with in-place operation
    merging. Implements the algorithm from DST_Allocation.md:

    Phase 1: Copy insertion for multi-consumer values
    - When a value has multiple consumers and any is in-place (overwrites input),
      inserts ttl.copy_dst operations for all but the last consumer

    Phase 2: Build live intervals with in-place merging
    - Builds interval [start, end] for each tile value
    - Merges intervals for in-place ops (input and output share same DST)

    Phase 3: Linear scan allocation for inputs/intermediates
    - Allocates DST[0..k-1] for input block arguments and intermediate values
    - Uses interval-based linear scan (Wimmer & Franz, CGO'10)

    Phase 4: Linear scan allocation for outputs
    - Allocates DST[k..n] for values yielded from the compute body
    - Separate region ensures outputs aren't overwritten during computation

    This pass also:
    - Inserts ttl.copy_tile for block arguments
    - Assigns dst_index SSA operands to tile math operations
    - Checks capacity and emits diagnostics on overflow

    Nested DST operations are preserved when their destination and source
    indices are resolved and they do not require copy insertion. The pass
    diagnoses unsupported nested operations before modifying IR.
  }];

  let options = [
    Option<"dstCapacity", "dst-capacity", "uint32_t", "0",
           "Override DST register capacity (auto-computed by default). "
           "For single-buffering mode (dst_full_sync_en=true), use: "
           "f32=8, f16/bf16=16. For double-buffering (default): f32=4, f16/bf16=8.">,
    Option<"separateOutputRegion", "separate-output-region", "bool", "false",
           "Allocate outputs in separate DST region (needed for reductions and some loop optimizations).">
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLSetComputeKernelConfig
    : Pass<"ttl-set-compute-kernel-config", "::mlir::ModuleOp"> {
  let summary = "Resolve compute-kernel configuration";
  let description = [{
    Selects tile execution strategies and resolves kernel-wide destination
    element width, synchronization mode, and per-dataflow-buffer unpack mode.
    Target capabilities, user policy, and target-independent kernel
    requirements are analyzed separately. The pass validates a complete
    configuration for every function before writing operation or function
    attributes. Module scope provides one immutable launch-domain snapshot and
    a transformation barrier before destination assignment mutates functions.
  }];

  let options = [
    Option<"fp32DestAccEn", "fp32-dest-acc-en", "std::string", "\"auto\"",
           "Select fp32 destination accumulation: auto, enabled, or disabled.">,
    Option<"dstFullSyncEn", "dst-full-sync-en", "std::string", "\"auto\"",
           "Select full DST synchronization: auto, enabled, or disabled.">,
    Option<"reduceFullFp32", "reduce-full-fp32", "bool", "true",
           "Prefer FP32 accumulation for reduce operations when supported.">,
    Option<"matmulFullFp32", "matmul-full-fp32", "bool", "true",
           "Prefer FP32 accumulation for matmul operations when supported.">,
    Option<"enableFPUBinaryOps", "enable-fpu-binary-ops", "bool", "true",
           "Allow tile add/sub/mul with matching operand coordinates and "
           "DFB-backed operands to select the FPU strategy. "
           "Disable to select the SFPU strategy.">
  ];

  let dependentDialects = [
    "::mlir::tt::ttl::TTLDialect"
  ];
}

def TTLSubblockComputeForDST
    : Pass<"ttl-subblock-compute-for-dst", "::mlir::func::FuncOp"> {
  let summary = "Partition ttl.compute into DST-sized subblocks";
  let description = [{
    Partitions the iteration space of `ttl.compute` operations into subblocks
    that fit in the DST register file. Uses the `ttl.unroll_factor` attribute
    (set by `ttl-assign-dst`) to determine the subblock size.

    `ComputeOp` implements `TilingInterface`, and this pass tiles it along its
    iteration dimensions to produce smaller inner `ttl.compute` ops wrapped in
    `scf.for` loops -- analogous to tiling `linalg` structured ops.

    For each `ttl.compute` with `ttl.unroll_factor > 1`, generates an outer
    `scf.for` loop with `tensor.extract_slice` ops to create sub-tensors for
    each subblock. The inner `ttl.compute` operates on the smaller sub-tensors
    and carries `ttl.full_linearization_strides` for index adjustment.

    Example (4x4 tensor, DST capacity 8, subblock 2x4):
    ```mlir
    // Before (16 tiles, unroll_factor=8):
    %r = ttl.compute ins(%a) outs(%init)
        {ttl.unroll_factor = 8 : i64} {
      ^bb0(%a_tile, %out_tile):
        %exp = ttl.tile_exp %a_tile
        ttl.yield %exp
    } -> tensor<4x4x!ttcore.tile<32x32, f32>>

    // After (outer loop over dim 0 in steps of 2):
    scf.for %iv = %c0 to %c4 step %c2 {
      %s_in  = tensor.extract_slice %a[%iv, 0] [2, 4] [1, 1]
      %s_out = tensor.extract_slice %init[%iv, 0] [2, 4] [1, 1]
      ttl.compute ins(%s_in) outs(%s_out)
          {ttl.full_linearization_strides = dense<[4, 1]>} {
        ^bb0(%a_tile, %out_tile):
          %exp = ttl.tile_exp %a_tile
          ttl.yield %exp
      } -> tensor<2x4x!ttcore.tile<32x32, f32>>
    }
    ```
  }];

  let options = [
    Option<"subblockSync", "subblock-sync", "bool", "false",
           "Refine DFB reserve/push to per-subblock granularity, enabling "
           "pack_tile_block for contiguous subblocks. When disabled (default), "
           "user-placed reserve/push is preserved.">,
    Option<"strictF32Acc", "strict-f32-acc", "bool", "false",
           "Error if a user-written accumulation loop (+=) with non-f32 "
           "output requires subblocking. Subblocking reduces accumulation "
           "precision because bf16 L1 intermediates are narrower than f32 DST.">
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}


def TTLLowerToLoops
    : Pass<"ttl-lower-to-loops", "::mlir::func::FuncOp"> {
  let summary = "Lower ttl.compute structured ops to scf.for loops";
  let description = [{
    Lowers `ttl.compute` operations to nested `scf.for` loops that iterate over
    the tile block. Generates the loop body with `tensor.extract` for input tiles,
    the cloned compute body, and `tensor.insert` to write results back.

    This pass runs BEFORE bufferization. After this pass:
    - `ttl.compute` ops are replaced by `scf.for` loops
    - Loop bodies contain `tensor.extract`, `ttl.tile_*` ops, and `tensor.insert`
    - All types remain tensor types (bufferization converts to memref later)

    Example transformation:
    ```mlir
    // Before:
    %result = ttl.compute ins(%a, %b) outs(%init) {
      ^bb0(%a_tile, %b_tile, %out_tile):
        %sum = ttl.tile_add %a_tile, %b_tile
        ttl.yield
    } -> tensor<2x2x!ttcore.tile<32x32, f32>>

    // After:
    %result = scf.for %i = 0 to 2 iter_args(%out = %init) {
      %inner = scf.for %j = 0 to 2 iter_args(%out_inner = %out) {
        %a_tile = tensor.extract %a[%i, %j]
        %b_tile = tensor.extract %b[%i, %j]
        %sum = ttl.tile_add %a_tile, %b_tile
        %updated = tensor.insert %sum into %out_inner[%i, %j]
        scf.yield %updated
      }
      scf.yield %inner
    } -> tensor<2x2x!ttcore.tile<32x32, f32>>
    ```
  }];

  let options = [
    Option<"dstAccumulation", "dst-accumulation", "bool", "false",
           "Use DST register accumulation for reduction loops. Reorders "
           "loops (parallel-outer, reduction-inner) so DST persists across "
           "reduction iterations. When disabled, uses per-tile L1 "
           "accumulation. reduce_max always uses DST accumulation.">,
    Option<"useBlockMatmul", "use-block-matmul", "bool", "true",
           "Lower matmul to block-level hardware calls "
           "(matmul_block) instead of per-tile loops.">
  ];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def TTLScheduleOperations
    : Pass<"ttl-schedule-operations", "::mlir::func::FuncOp"> {
  let summary = "Group tile operations by kind within sync regions";
  let description = [{
    Requires `ttl-set-compute-kernel-config` to select execution strategies for
    tile operations that support alternatives.

    Reorders tile operations within sync regions (`tile_regs_acquire` to
    `tile_regs_commit`) to group by operation kind. After `ttl-lower-to-loops`
    unrolls a subblocked compute, tile ops from different iterations are
    interleaved. This pass groups them so `ttkernel-insert-inits` can emit
    one init per consecutive group of same-type ops.

    Ops are sorted by (dependency depth, category, operation name, init affinity,
    dst_index). Dependency depth considers SSA RAW, DST RAW, WAW (same DST
    write index), and WAR (write after prior read of same DST index) hazards.
    Ops at the same depth are independent and can be reordered.

    Pure, speculatable bookkeeping operations are transparent to scheduling
    unless they consume a candidate tile result. Every other operation outside
    the scheduled tile categories preserves its relative order and divides the
    tile operations into independent scheduling sequences.

    Only tile ops between `tile_regs_acquire` and `tile_regs_commit` are
    scheduled. Store ops (after the commit/wait boundary) are not reordered.

    Category ordering:
      Bcast < Transpose < CopyTile < FPUBinary < SFPUUnary < SFPUBinary <
      CopyDst < DstIndex

    Example:
    ```mlir
    // Before (interleaved by tile iteration):
    ttl.tile_regs_acquire
    %s0 = ttl.tile_add %a0, %b0 into dst[%c0] : ...
    %e0 = ttl.tile_exp %s0 into dst[%c0] : ...
    %s1 = ttl.tile_add %a1, %b1 into dst[%c1] : ...
    %e1 = ttl.tile_exp %s1 into dst[%c1] : ...
    ttl.tile_regs_commit

    // After (grouped by kind at each dependency depth):
    ttl.tile_regs_acquire
    %s0 = ttl.tile_add %a0, %b0 into dst[%c0] : ...  // depth 0
    %s1 = ttl.tile_add %a1, %b1 into dst[%c1] : ...  // depth 0
    %e0 = ttl.tile_exp %s0 into dst[%c0] : ...       // depth 1
    %e1 = ttl.tile_exp %s1 into dst[%c1] : ...       // depth 1
    ttl.tile_regs_commit
    ```

    References:
    - Hennessy & Patterson, "Computer Architecture", Ch. 3: RAW/WAR/WAW hazards
    - Cooper & Torczon, "Engineering a Compiler", Ch. 12: list scheduling with
      topological depth levels
  }];

  let dependentDialects = [];
}

def TTLAnnotateCBAssociations
    : Pass<"ttl-annotate-cb-associations", "::mlir::func::FuncOp"> {
  let summary = "Annotate ttl.compute block args with CB index associations";
  let description = [{
    Analysis pass that annotates ttl.compute block arguments with the CB index
    (0-31) of their associated circular buffer. This enables subsequent conversion
    passes to look up the correct CB without fragile state management.

    For each ttl.compute input that is produced by ttl.attach_cb or ttl.cb_wait,
    this pass:
    1. Traces the input to find the associated CB via getAttachedCB()
    2. Extracts the cb_index attribute from the CB
    3. Annotates the corresponding block argument with ttl.cb_index = N

    This annotation survives subsequent IR transformations and allows copy_tile
    lowering to find the correct CB by cb_index lookup.
  }];

  let dependentDialects = [];
}

def TTLVerifyPipeNetGuards
    : Pass<"ttl-verify-pipenet-guards", "::mlir::ModuleOp"> {
  let summary = "Verify PipeNet-coupled DFB and pipe operations are role-guarded";
  let description = [{
    Checks that every operation coupled to a PipeNet executes only on launch
    coordinates where its role is defined. A launch coordinate is a
    `(core_x, core_y)` point in the 2D grid declared by the module's
    `ttl.launch_grid` attribute; each kernel function runs once per
    coordinate. A domain is a subset of those coordinates.

    A forward dataflow analysis tracks, at each program point, the domain on
    which it may execute under the enclosing region predicates (`scf.if`,
    `affine.if`, `ttl.if_src`, `ttl.if_dst`, `ttl.pipenet_scope`). At each
    `ttl.copy`, `ttl.cb_wait`, and `ttl.cb_push` the analysis verifies the
    point's domain is contained in the pipe's source or destination domain;
    out-of-domain uses, scopes broader than their declared role, unmatched
    `ttl.cb_wait`s, and unanalyzable coord-dependent predicates are reported.
    Each diagnostic includes an example offending coordinate, the contributing
    PipeNet(s), and a suggested guard.

    The pass expects one module per `@ttl.operation`, a well-formed
    `ttl.launch_grid` (i64 array of length 2 with positive entries), static
    `I64Attr` coordinates on `ttl.create_pipe`, and explicit DFB synchronization
    inserted by `ttl-insert-cb-sync`. Every user-declared DFB must have its
    module-wide logical `dfb_id`; compiler-created DFBs receive analysis-only
    logical IDs until physical index finalization. DFB lifecycle operands used
    by the guard analysis must resolve to `ttl.bind_cb` declarations.

    When `TTL_RELAX_DFB_SPSC` is set, this pass skips the check that each DFB
    wait has a producer on the same dynamically active launch nodes. The
    program must enforce that correspondence. The later SPSC verifier still
    rejects a waited DFB with neither a compiler-visible push nor uncontracted
    external access that may contain one. PipeNet endpoint guards, transfer
    correspondence, and synchronization schedules remain mandatory.
  }];
}

def TTLVerifyPipeNetSchedule
    : Pass<"ttl-verify-pipenet-schedule", "::mlir::ModuleOp"> {
  let summary = "Verify PipeNet synchronization correspondence and ordering";
  let description = [{
    Verifies the synchronization protocol formed by pipe sends, receiver posts,
    and receiver waits. A send waits until every destination has posted a
    dataflow buffer reservation, so each send requires one corresponding post
    at every destination. A receiver wait observes the send associated with
    the exact receive handle it consumes. Repeated waits on one handle observe
    the same completed transfer and do not require additional sends. Posting a
    reservation does not wait for payload, and a sender does not wait for the
    receiver to consume the payload. Receiver posts may therefore exist
    without receiver waits, and sends do not require receiver waits. When a
    pipe has sends, each destination must have the same number of receiver
    posts as sends. A pipe with no sends or waits may contain unused receiver
    posts.

    Receiver-published addressing supports one outstanding post for each pipe
    and destination. Each repetition must contain its corresponding send or an
    exact receive wait that proves the send completed. Before a distinct post
    executes, the graph must prove that the preceding send has completed or
    that the posts are mutually exclusive.

    Static event definitions are paired by program order for each pipe
    endpoint. Each pair must have equal dynamic execution counts and equivalent
    enclosing conditions. Alternative event definitions under runtime control
    flow remain distinct unless one is proven not to execute. Direct calls to
    helper functions are expanded at each call site so caller order and
    invocation multiplicity are preserved. Every region containing pipe events
    directly or through helper calls must have one block. Functions containing
    pipe events must be kernel-thread entry points or be reachable from one
    through direct calls. Definitions of one event kind for a pipe endpoint
    must belong to one kernel-thread function because independent kernel
    threads have no program order. Events with statically analyzable predicates
    use their exact launch-node domain. Otherwise, the static PipeNet endpoint
    role provides a conservative domain that includes every possible event;
    execution-count analysis still removes coordinates proven not to execute.
    Multi-block regions that contribute pipe events, recursive calls containing
    pipe events, and schedules with more than 4096 expanded event occurrences
    at one launch node are rejected because the verifier cannot construct a
    correct bounded event graph.

    The pass also constructs the same-thread wait-for graph. Program-order
    edges connect consecutive events on one launched node. Protocol edges
    connect each receiver post to its send and that send to every wait on the
    post's exact receive handle.
    A cycle means no event in the cycle can execute and is rejected.

    The pass expects one module per `@ttl.operation`, a well-formed
    `ttl.launch_grid` attribute, and static `ttl.create_pipe` endpoints within
    that launch grid.
    `ttl-verify-pipenet-guards` must run first in the compiler pipeline so
    invalid launch domains are reported separately. Both verifiers run after
    DFB synchronization insertion and before DFB acquire coalescing and physical
    index finalization.
  }];
}

def TTLVerifyTransferProvenance
    : Pass<"ttl-verify-transfer-provenance", "::mlir::ModuleOp"> {
  let summary = "Verify transfer values have consistent reachable origins";
  let description = [{
    Checks non-local provenance required by transfer lowering. Transfer objects
    must resolve to one `ttl.pipe_transfer.create`; pipe receive destinations
    must derive from `ttl.cb_reserve`; public transfer handles must have one
    supported wait behavior; and every pipe token must derive from a receive
    post in the token's PipeNet. Statically unreachable operations are ignored.
  }];

  let dependentDialects = [];
}

def TTLVerifyDFBSPSC
    : Pass<"ttl-verify-dfb-spsc", "::mlir::ModuleOp"> {
  let summary = "Verify DFB producers and consumers across launch domains";
  let description = [{
    Walks DFB protocol actions exposed through `DFBAccessOpInterface`, groups
    them by module-wide logical `dfb_id` and enclosing
    `ttl.kernel_thread`-tagged `func.func`, and tracks the launch-node domain for
    each producer or consumer participant. The pass requires every waited DFB
    to have a compiler-visible push or uncontracted external access that may
    contain one and, for each launched node, at most one producer kernel and at
    most one consumer kernel. Disjoint domains across different kernel
    functions are accepted. Distinct logical DFBs remain separate when physical
    allocation assigns them the same `cb_index`.

    tt-metal circular buffers expose a counter pair (`pages_received`,
    `pages_acked`) that is not multi-writer safe on either side, so the
    per-launched-node single-producer single-consumer property is required for
    correctness; see `docs/development/DFBManagement.md` for the runtime
    contract. The pass emits an error for waits without any possible producer,
    overlapping producer or consumer domains, and unknown domains when
    multiple threads participate and SPSC cannot be proven statically.

    The module must contain the `ttl.dfb_allocations` metadata emitted by
    `ttl-finalize-dfb-indices`, and every DFB declaration and lifecycle operand
    must resolve to a `ttl.bind_cb` with `dfb_id`. Modules with DFB acquire ops
    must also carry a valid `ttl.launch_grid` attribute.

    Setting `TTL_RELAX_DFB_SPSC` skips per-launch-node producer and consumer
    ownership verification. The program must enforce that ownership contract
    through synchronization not represented in the IR. The verifier still
    rejects every waited DFB without a compiler-visible push or uncontracted
    external access that may contain one. Finalized DFB identity,
    physical-index, and launch-grid preconditions remain mandatory. Strict
    verification is the default.
  }];

  let dependentDialects = [];
}

def TTLErasePipeNetScopes
    : Pass<"ttl-erase-pipenet-scopes", "::mlir::ModuleOp"> {
  let summary = "Inline and erase `ttl.pipenet_scope` regions";
  let description = [{
    `ttl.pipenet_scope` is a structural marker emitted by the frontend and
    consumed by `ttl-verify-pipenet-guards`. After verification the markers
    serve no further purpose; this pass inlines each scope's body into its
    parent block and erases the scope op so downstream lowering sees a
    scope-free IR. Runs after `ttl-verify-pipenet-guards`.
  }];

  let dependentDialects = ["::mlir::tt::ttl::TTLDialect"];
}

def TTLLowerDPrintToEmitC
    : Pass<"ttl-lower-dprint-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Lower ttl.dprint ops to EmitC verbatim for debug printing";
  let description = [{
    Converts ttl.dprint operations to EmitC verbatim code that emits
    device debug print calls. Resolves CB indices, DST register slots,
    and data formats from the IR. Runs before convert-ttl-to-ttkernel
    so that DST assignments and CB annotations are still available.

    Circular-buffer, tile, and CB-backed tensor prints keep an i32
    `ttkernel.get_compile_time_arg_val` use as an `emitc.verbatim` operand
    instead of baking the index into C++ text. `ttkernel-annotate-dfb-use`
    can then drop prints whose DFB has no remaining non-print use on a
    specialized core.
  }];

  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTLLowerSignpostToEmitC
    : Pass<"ttl-lower-signpost-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Lower ttl.signpost ops to EmitC verbatim for profiling";
  let description = [{
    Converts ttl.signpost operations to EmitC verbatim code that emits
    DeviceZoneScopedN markers for Tracy profiling. This pass runs after
    convert-ttl-to-ttkernel to keep profiling concerns separate from
    the main dialect conversion.
  }];
}

def TTLFinalizeDFBIndices
    : Pass<"ttl-finalize-dfb-indices", "::mlir::ModuleOp"> {
  let summary = "Assign physical DFB indices and emit allocation metadata";
  let description = [{
    Module-level pass that runs after all DFB-creating and synchronization
    passes. Must precede `ttl-set-compute-kernel-config` and
    `ttl-annotate-cb-associations`, which copy DFB indices into attributes and
    would otherwise capture provisional indices.

    User-declared DFBs enter this pass with module-wide logical `dfb_id`
    attributes. Compiler-created DFB declarations receive unique logical IDs.
    The logical identity remains independent of the physical `cb_index`, so
    later analyses can distinguish DFBs that share one hardware slot.

    With `reuse-user-dfbs` enabled, the pass analyzes all logical DFBs,
    including declarations shared across kernels. Within one configuration
    epoch, reuse requires exact `CircularBufferType` equality, equal transaction
    tile counts that divide the DFB capacity, and identical read- and
    write-pointer processors on each launch node. These requirements preserve
    hardware counter and ring-pointer state. Logical DFBs active only in
    disjoint synchronized configuration epochs may share a physical index with
    different geometry or storage. A DFB with live protocol state crosses a
    reconfiguration unchanged and remains active in every epoch it spans, so
    its physical index cannot be reassigned at those boundaries.

    Physical-index coloring and storage coloring enforce different contracts.
    A physical index identifies one active hardware descriptor and therefore
    requires compatible page format and protocol state. After selecting the
    physical indices, the pass may assign distinct physical descriptors to one
    `storage_index` when every use of one descriptor completes before any use
    of the other begins on every launch node where both are resident. The
    descriptors may use different element types, page sizes, or capacities
    because each retains its own runtime descriptor.
    The shared backing allocation uses the maximum required byte capacity with
    an alignment divisible by every member page size. Tensor-backed descriptors
    and descriptors without a proven storage-lifetime order remain separate.

    One unresolved structured condition is supported when every active DFB
    access belongs to one reserve, push, wait, and pop transaction controlled
    by equivalent `scf.if` or `affine.if` frames. Branch polarity, nesting, and
    control values must match. Within one kernel function, ordinary SSA
    equality can establish equivalence. Separately evaluated `scf.if`
    conditions can match across logical kernels when every unresolved frame is
    a structurally equal expression rooted in the same typed dispatch-condition
    identities at the same launch coordinate. Untyped evaluations, mixed typed
    and untyped nesting, and unresolved loops remain conservative. A logical
    DFB without a matched lifecycle or a proven lifetime order conflicts with
    every other DFB.

    Explicit allocation groups may reuse one physical index across
    reset-delimited epochs from different logical DFBs. Every pair of active
    epochs on a launch node must have one strict proven order. Each adjacent
    handoff must either follow a reset that establishes canonical pointer state
    or preserve both pointer owners. The allocator simulates every epoch's
    transaction sequence against the group's physical capacity envelope and
    rejects boundary crossings or unequal read and write offsets. This
    epoch-level contract applies only to explicit allocation groups; automatic
    reuse continues to compare complete logical-DFB lifetimes.

    Two DFBs used by one compute kernel may share a physical index only when
    every target-supported destination width assigns compatible static unpack
    modes to that index. This constraint is added to the conflict graph before
    coloring.

    `unsafe-assume-allocation-groups` applies only to explicit allocation
    groups. It permits missing launch-domain, access-completion,
    pointer-handoff, and lifetime-order proofs while preserving page-format,
    storage, static compute-configuration, ring-envelope, target-capacity, and
    L1-budget
    validation. Each group accepted this way emits a warning and is recorded
    in `ttl.assumed_dfb_allocation_groups`. Automatic DFB reuse remains
    proof-based. The option is disabled by default.

    An access whose launch domain is otherwise unknown is refined to an exact
    domain when execution-count analysis proves a count on every base launch
    node. Nodes with positive counts belong to the domain; nodes with zero
    counts do not. The logical DFB domain is exact only when every access has an
    exact domain. A logical DFB is inactive when the union of all its access
    domains is empty. An inactive scratch DFB may share a compatible physical
    descriptor without a lifetime-order proof because no launched node accesses
    it. Tensor-backed empty domains retain their existing allocation diagnostic.
    Operations proven to execute zero times on every launch node do not impose
    runtime compute-configuration requirements, but strategy-dependent tile
    operations still receive a strategy required by later lowering. Unknown
    execution remains configuration-constraining. The subsequent compute
    configuration pass uses the same exact-empty proof.

    The allocator represents each logical DFB as a vertex, each pair that
    cannot share as an edge, and each physical index as a color. First-fit
    quickly produces a valid assignment by giving each DFB the lowest index
    not used by one of its conflicts, but its result depends on DFB order and
    may use more indices than necessary. A found set of pairwise-conflicting
    DFBs proves that at least that many indices are required.

    Exhaustive assignment search can grow exponentially, so it runs only when
    first-fit prevents acceptance and the pairwise-conflict lower bound does
    not already prove failure. The physical-index check asks directly whether
    an assignment fits the available indices. When first-fit exceeds the
    DFB-plus-fixed-state L1 budget, or the provisional threshold after a
    conservative PipeNet reservation, weighted search asks whether another
    assignment fits using the sum of the target-aligned maximum allocation
    assigned to each physical index. The reservation triggers search but cannot
    make finalization reject an assignment that satisfies the authoritative
    budget. Conversion validates the finalized allocation against exact PipeNet
    resources.
    Each search examines at most `exact-coloring-search-limit` deterministic
    states to bound compile time. Reaching that limit reports an inconclusive
    allocation diagnostic only when acceptance requires the search result. A
    reservation-only search may retain an authoritative-budget-valid first-fit
    assignment. Limit exhaustion does not report that hardware capacity was
    proved insufficient.

    An assertions-enabled build accepts
    `-debug-only=ttl-finalize-dfb-indices` to print the immutable liveness facts
    and typed conflicts before allocation succeeds or fails. The report includes
    logical descriptors, access domains, protocol effects, exact execution
    counts, transaction sizes, pointer owners, lifetime frontiers, and conflict
    evidence. Lifecycle epochs identify every configuration in which live state
    must be retained. Lifetime frontiers use access-occurrence indices that map
    to the report's numbered access records. For an unknown launch-node domain,
    it separately labels a possible-domain per-node analysis that treats unknown
    accesses as possible while preserving exact-zero inactivity. It may
    establish conditional boundedness only for one complete transaction whose
    active accesses share one structured at-most-once condition on every
    possible node. Reuse remains restricted to two conditionally bounded
    unknown-domain DFBs with matching transactions, pointer owners, and
    possible-domain lifetime order. Per-access counts and lifetime-frontier
    indices remain report-only. If node-dependent IR has no launch grid, the
    report retains logical DFB and access facts but omits per-node rows because
    no base launch nodes are available.

    Every storage-accessing operation with a direct DFB operand is treated as a
    runtime access from operation entry through completion. Every direct DFB
    operand of `ttl.opaque_call` is a possible read or write unless a typed
    summary describes that dependency occurrence. An `inspect` access is
    bounded by the call and contributes no queue-state transition because the
    callee does not publish, consume, or leave the DFB changed. Its storage
    lifetime must still be strictly ordered relative to another logical DFB on
    the same physical index. Identity-only `ttl.attach_cb` and
    `ttl.get_dfb_id` operations do not access storage. An external call whose
    C++ body implements the DFB protocol does not provide the visible reserve,
    push, wait, and pop operations required to prove a bounded lifecycle unless
    it supplies exact typed protocol effects.

    Disabling `reuse-user-dfbs` compacts distinct provisional user indices
    without introducing new sharing between user DFBs, then assigns only
    compiler-created DFBs. It also assigns separate storage to every physical
    descriptor. Both modes use the same lifetime proof and conflict relation
    for every DFB whose index they select.

    Every used compiler-created DFB must contain `ttl.cb_reserve`,
    `ttl.cb_push`, `ttl.cb_wait`, and `ttl.cb_pop`. An unused declaration is
    legal and remains live through the end of its kernel.

    The pass rejects copied DFB-index attributes from later passes. It computes
    all physical assignments and runtime metadata and validates the selected
    target's physical DFB-index capacity before modifying the IR. Applying the
    validated plan updates physical indices and `ttl.base_cta_index`, assigns
    missing compiler-created logical IDs, and builds `ttl.dfb_allocations` for
    the Python runtime. The attribute contains one descriptor per physical
    index, including user-declared and compiler-created DFBs, and records the
    page size computed from the finalized element type. Its `storage_index`
    identifies the byte allocation used by that descriptor; multiple distinct
    descriptors may share one storage index only when their storage lifetimes
    are proven disjoint. A descriptor also records the exact union of launch
    nodes that access its physical index when that domain is known. An exact
    empty domain causes the runtime to install no DFB descriptor; omitted domain
    metadata requires conservative whole-grid allocation. See
    docs/development/DFBManagement.md.

    Storage assignment processes descriptors by decreasing target-rounded
    size. For each descriptor, it selects the legal storage index with the
    lowest current maximum allocation over launch nodes, then the lowest global
    allocation. A descriptor contributes only on nodes in its known allocation
    domain; an unknown domain contributes on every launch node.
  }];

  let options = [
    Option<"reuseUserDFBs", "reuse-user-dfbs", "bool", "true",
           "Reuse physical DFB indices and compiler-managed storage for "
           "lifetimes proven not to overlap across concurrent kernels.">,
    Option<"unsafeAssumeAllocationGroups", "unsafe-assume-allocation-groups",
           "bool", "false",
           "Trust explicit DFB allocation groups when runtime handoff cannot "
           "be proven. Emits warnings and assumption metadata.">,
    Option<"exactColoringSearchStateLimit", "exact-coloring-search-limit",
           "uint64_t", "1000000",
           "Maximum states examined by exact DFB allocation before reporting "
           "an inconclusive result.">,
    Option<"l1BudgetOverride", "l1-budget-override", "uint32_t", "0",
           "Override the per-core L1 budget used by DFB allocation and "
           "conservative PipeNet reservation. 0 selects the target budget.">
  ];
}

def TTLValidateCBBudget
    : Pass<"ttl-validate-cb-budget", "::mlir::ModuleOp"> {
  let summary = "Validate total DFB allocation fits within per-core L1 budget";
  let description = [{
    Module-level pass that runs after `ttl-finalize-dfb-indices`, before
    `convert-ttl-to-ttkernel`. It sums one target-allocation-aligned maximum
    static backing-store size per compiler-selected DFB storage index on each
    launch node, allocator-rounded synchronized-reset state, and one
    target-aligned configuration tensor per synchronized reconfiguration
    boundary, then compares the maximum node result against a per-core L1
    budget. Tensor-backed DFB storage is excluded because the tensor allocator
    owns that allocation. Unknown allocation domains conservatively contribute
    to every launch node; missing residency metadata uses the global descriptor
    sum.

    For each `BindCBOp`, the pass computes:
      dfb_payload_bytes = total_elements * tile_size_bytes
      dfb_allocation_bytes = align_to_target_quantum(dfb_payload_bytes)

    When multiple physical `cb_index` descriptors use the same `storage_index`,
    the pass computes one compatible byte capacity and alignment for that
    storage allocation. Repeated declarations of one physical index contribute
    only their largest allocation.

    On overflow the pass emits an error with a per-storage breakdown and a
    hint to reduce block shapes or block counts.

    L1 budget is taken from the module's TTCore system descriptor and
    default device when present (`ChipDescAttr::getUsableL1Size()` for the
    selected chip). When the module has no system descriptor, the pass uses
    the same static per-core DFB region limit as Wormhole and Blackhole
    (total L1 minus reserved kernel space).

    This is a static DFB and fixed-state check. After exact PipeNet planning,
    `convert-ttl-to-ttkernel` validates the same maximum finalized per-node DFB,
    reset, and reconfiguration footprint together with target-aligned PipeNet
    scratch and one target allocation quantum per GlobalSemaphore. Neither
    check models kernel binaries, stacks, or unrelated runtime allocations.
  }];

  let options = [
    Option<"l1BudgetOverride", "l1-budget-override", "uint32_t", "0",
           "Override L1 DFB budget (bytes). 0 means auto-detect from "
           "device attributes or use architecture default.">
  ];
}

def TTLDumpCBFlowGraph
    : Pass<"ttl-dump-cb-flow-graph", "::mlir::ModuleOp"> {
  let summary = "Analyze and dump CB producer/consumer flow graph";
  let description = [{
    Analysis pass that builds a flow graph showing how data moves through
    dataflow buffers between kernels. Logical DFB identity separates flows that
    reuse one physical CB index. For each DFB, the graph identifies:
    - Producers: operations that write to the CB (cb_reserve + copy/store)
    - Consumers: operations that read from the CB (cb_wait)
    - DMA operations: copy ops with their direction (read/write)
    - Wait operations: ttl.wait barriers

    Outputs the graph in a human-readable format and optionally as JSON
    for consumption by the auto-profiler. The graph maps source locations
    to CB operations, enabling correlation with runtime profiler data. JSON
    entries include the logical `dfb_id` when available and the physical
    `cb_index` used by runtime events.

    Example output:
    ```
    CB Flow Graph:
    CB[0] "inp_cb":
      producers: dm_read:74 (copy DRAM->CB)
      consumers: compute:52 (cb_wait)
    CB[1] "out_cb":
      producers: compute:55 (cb_reserve)
      consumers: dm_write:106 (cb_wait)
    ```
  }];

  let options = [
    Option<"outputPath", "output", "std::string", "\"\"",
           "Path to write JSON output (empty for stderr debug only)">
  ];
}

def TTKernelLowerScalarFpTypes
    : Pass<"ttkernel-lower-scalar-fp-types", "::mlir::func::FuncOp"> {
  let summary = "Lower scalar float types to integer bit patterns";
  let description = [{
    Converts all scalar float types to integer bit-pattern types using MLIR
    dialect conversion. After `convert-ttl-to-ttkernel`, float values from L1
    loads are represented as integer bit patterns bridged by
    `unrealized_conversion_cast(iN -> fN)`. This pass eliminates those bridges
    by converting all scalar `FloatType` values to the corresponding
    `IntegerType` (f32 -> i32, bf16 -> i16).

    The following conversions are performed:

    1. **arith.cmpf**: Converts scalar float comparisons to TTKernel soft-float
       comparison ops (`ttkernel.float32_greater` / `ttkernel.bfloat16_greater`).
       Supported predicates: `ogt` and `olt` (swapped operands). Unsupported
       predicates cause the pass to fail.

    2. **arith.truncf**: Converts float truncation (e.g. f32 -> bf16) to
       integer bit extraction via `arith.shrui` + `arith.trunci`.

    3. **arith.constant**: Converts float constants to their integer bit-pattern
       equivalents.

    4. **SCF control flow**: Propagates integer types through `scf.if`,
       `scf.for`, `scf.while`, and `scf.yield` using upstream SCF structural
       type conversion patterns.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTKernelInsertInits
    : Pass<"ttkernel-insert-inits", "::mlir::ModuleOp"> {
  let summary = "Insert minimal init ops before TTKernel compute ops";
  let description = [{
    Inserts hardware initialization ops required before TTKernel compute
    operations. Two categories of inits are inserted:

    Common inits (`init_sfpu`, `binary_op_init_common`) configure the
    UNPACK and PACK data format routing for a sync region. These are placed
    once per sync region, hoisted above any enclosing loops.

    Per-op inits (`exp_tile_init`, `add_tiles_init`, etc.) configure the
    MATH pipeline for a specific operation type. An init is emitted
    whenever the op type changes from the previous compute op in
    linear block order. Init tracking resets at sync boundaries
    (`tile_regs_acquire`/`commit`/`wait`/`release`).

    Example (consecutive same-type ops share one init):
    ```mlir
    // Before:
    ttkernel.tile_regs_acquire
    ttkernel.exp_tile(%c0)
    ttkernel.exp_tile(%c1)
    ttkernel.log_tile(%c2)
    ttkernel.tile_regs_commit

    // After:
    ttkernel.init_sfpu(%cb0, %cb1)
    ttkernel.tile_regs_acquire
    ttkernel.exp_tile_init
    ttkernel.exp_tile(%c0)
    ttkernel.exp_tile(%c1)      // no re-init (same type)
    ttkernel.log_tile_init       // re-init on type switch
    ttkernel.log_tile(%c2)
    ttkernel.tile_regs_commit
    ```

    Example (interleaved ops each get their own init):
    ```mlir
    // Before:
    ttkernel.tile_regs_acquire
    ttkernel.exp_tile(%c0)
    ttkernel.log_tile(%c1)
    ttkernel.exp_tile(%c2)
    ttkernel.tile_regs_commit

    // After:
    ttkernel.init_sfpu(%cb0, %cb1)
    ttkernel.tile_regs_acquire
    ttkernel.exp_tile_init
    ttkernel.exp_tile(%c0)
    ttkernel.log_tile_init
    ttkernel.log_tile(%c1)
    ttkernel.exp_tile_init       // re-init required (type switched back)
    ttkernel.exp_tile(%c2)
    ttkernel.tile_regs_commit
    ```
  }];

  let dependentDialects = [
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTKernelSpecializeCores
    : Pass<"ttkernel-specialize-cores", "::mlir::ModuleOp"> {
  let summary = "Clone kernels per core when control flow branches on a coordinate";
  let description = [{
    Per-core specialization for TTKernel functions. Requires the
    `ttl.launch_grid` attribute, missing or malformed `ttl.launch_grid` is a hard error.

    For each kernel whose scf.if region branches on a core coordinate, clones
    the function once per launch coordinate, replaces
    ttkernel.my_logical_x_ / ttkernel.my_logical_y_ with arith.constant's
    for that core, and tags the cloned function with ttl.core_coord (the coordinate it
    serves). Downstream canonicalize / cse then fold the now-constant
    branch conditions and delete untaken regions. The runtime bridge reads
    ttl.core_coord to build per-kernel core ranges for dispatch.

    NOTE: functions with symbol uses (e.g. `func.call` targets) are left
    unspecialized with a warning so erasing the original does not leave dangling
    `SymbolRefAttr`s; unrelated functions in the module are still specialized.

    Example (`ttl.launch_grid = [1, 2]`):
    ```mlir
    // Before:
    func.func @k() -> index {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c9 = arith.constant 9 : index
      %y = ttkernel.my_logical_y_ : () -> index
      %pred = arith.cmpi eq, %y, %c0 : index
      %r = scf.if %pred -> (index) {
        scf.yield %c7 : index
      } else {
        scf.yield %c9 : index
      }
      return %r : index
    }

    // After:
    func.func @k_c0_0() -> index attributes {ttl.core_coord = [[0, 0]]} {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c9 = arith.constant 9 : index
      %y = arith.constant 0 : index
      %pred = arith.cmpi eq, %y, %c0 : index
      %r = scf.if %pred -> (index) {
        scf.yield %c7 : index
      } else {
        scf.yield %c9 : index
      }
      return %r : index
    }
    func.func @k_c0_1() -> index attributes {ttl.core_coord = [[0, 1]]} {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c9 = arith.constant 9 : index
      %y = arith.constant 1 : index
      %pred = arith.cmpi eq, %y, %c0 : index
      %r = scf.if %pred -> (index) {
        scf.yield %c7 : index
      } else {
        scf.yield %c9 : index
      }
      return %r : index
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect"
  ];
}

def TTKernelAnnotateDFBUse
    : Pass<"ttkernel-annotate-dfb-use", "::mlir::ModuleOp"> {
  let summary = "Record physical DFB indices used by each TTKernel function";
  let description = [{
    Runs after per-core specialization and canonicalization.
    For each `ttkernel.thread` function, records the
    sorted physical DFB indices reached through surviving
    `ttkernel.get_compile_time_arg_val` operations below
    `ttl.base_cta_index` in `ttl.used_dfb_indices`. Helpers stay in the
    call-graph analysis so their uses propagate to kernel entries.
    Declarations are not annotated. The compile-time ABI places
    every DFB before tensor-accessor arguments, including DFB ids passed to
    opaque externs as scalar values.

    Calls are resolved through the MLIR call graph. Uses are propagated from
    callees to callers in SCC order, including recursive calls. An unknown or
    external callee is handled conservatively by recording every DFB
    compile-time argument below `ttl.base_cta_index`.

    Print-only `get_compile_time_arg_val` operations (those whose only users
    are `emitc.verbatim` debug prints, through unrealized casts) are excluded
    from the recorded set. After call-graph propagation, those prints are
    erased when the function does not otherwise use that DFB index. Keeping
    them would require allocating the DFB on a specialized core that no
    longer uses it. Debug prints of a DFB remain on functions that still
    have a non-print use of it, including uses inherited from callees.
  }];

  let dependentDialects = [];
}

def TTKernelFinalizeTensorRuntimeArgs
    : Pass<"ttkernel-finalize-tensor-runtime-args", "::mlir::ModuleOp"> {
  let summary = "Finalize tensor common runtime arguments";
  let description = [{
    Compacts each kernel function's tensor-address prefix after dead-code
    elimination. Surviving `ttkernel.get_common_arg_val` indices are rewritten
    to the compact prefix, compiler-managed indices following that prefix are
    shifted by the number of removed tensor addresses, and
    `ttl.crta_indices` retains the corresponding global tensor indices.

    The pass also records the surviving tensors used to construct
    `LocalTensorAccessor` values in `ttl.local_tensor_indices`. The runtime uses
    this metadata to require local tensor shards on every executing core.

    Dynamic common-argument indices represented by
    `ttkernel.experimental.constant_table_lookup` are rewritten with the same
    mapping. Functions containing common-argument accesses hidden in verbatim
    C++ retain the complete tensor prefix because those indices cannot be
    analyzed structurally.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}

def TTKernelCombinePackTiles
    : Pass<"ttkernel-combine-pack-tiles", "::mlir::func::FuncOp"> {
  let summary = "Combine consecutive pack_tile ops into pack_tile_block";
  let description = [{
    Scans for consecutive `ttkernel.pack_tile` ops targeting the same dataflow
    buffer with contiguous DST and CB tile indices, and replaces each run with
    a single `ttkernel.pack_tile_block`. This reduces the number of hardware
    pack operations after loop unrolling.

    Only combines runs where both DST indices and CB tile indices step by 1,
    and the first CB tile index is 0 (pack_tile_block always writes to the CB
    starting from index 0, per the op definition). Runs are broken by different
    CBs, non-contiguous indices, non-constant indices, or intervening
    non-constant ops. Runs of length 1 are skipped.

    Example:
    ```mlir
    // Before:
    ttkernel.pack_tile(%c0, %cb, %c0, true)
    ttkernel.pack_tile(%c1, %cb, %c1, true)
    ttkernel.pack_tile(%c2, %cb, %c2, true)
    ttkernel.pack_tile(%c3, %cb, %c3, true)

    // After:
    ttkernel.pack_tile_block(%c0, %cb, %c4)
    ```
  }];

  let dependentDialects = [
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

#endif // TTLANG_DIALECT_TTL_PASSES_TD
