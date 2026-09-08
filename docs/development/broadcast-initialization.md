# Broadcast initialization

The init insertion pass emits one `compute_kernel_hw_startup` at function entry,
before CB waits, control flow, and compute API calls. Startup configures PACK and
DST synchronization once. Region and operation changes use source and pack
reconfiguration and short pipeline initialization, preserving the DST bank.

The pass analyzes all sync regions on immutable IR before changing any function.
Each region plan records the insertion point, physical SrcA/SrcB inputs, output,
and compute category. All configuration operands must dominate the insertion
point, checked with MLIR dominance analysis. The first region supplies startup's
CBs: entry block arguments are already available, and operand-free, pure
`get_compile_time_arg_val` definitions can move to entry. Other startup CB
producers are rejected before applying any plan. Moving compile-time lookups
neither reads buffer data nor changes synchronization or instrumentation.

Each region configures input and output formats and tile geometry. Existing
compiler-loop hoisting is retained for this configuration; user loops and
conditionals keep their region configuration inside their control flow. Per-op
input reconfiguration also occurs before copy, binary FPU, destination reuse,
matmul, reduction, transpose, and broadcast inits. It restores the source state
on compiler-loop backedges and after intervening operation families. Matmul
passes its right input as physical SrcA and its left input as physical SrcB to
the regular-order startup and reconfiguration APIs.

Output reconfiguration enables tile-dimension changes and preserves packer
address modes, L1 accumulation state, and DST synchronization. Transpose uses
the short `transpose_init` API. Matmul uses `mm_block_init_short`; the pass no
longer inserts legacy full matmul, SFPU, or binary common initialization.

On Wormhole and Blackhole, format reconfiguration also restores the FPU source
zero-substitution default. Copy init reestablishes the unary preserve-zero flag
through the pinned LLK helper, retaining the legacy SFPU init's treatment of
BF16 negative zero. This changes only the MATH zero-substitution setting, without
resetting DST synchronization. Device coverage checks `signbit(-0.0)` explicitly.

Consecutive broadcasts with the same operation kind, broadcast dimension, and
input values share an init under the existing init-key algorithm. Each new
broadcast init reconfigures its inputs unconditionally. In particular, the first
pair in a sync region cannot assume that the region's formats are still
active: a surrounding tile loop may retain a different pair from the previous
iteration. Unary broadcast configures both source registers from its input,
consistent with Metal's unary hardware-startup overload.

Neither per-operation init resets MATH/PACK synchronization. With half-sync,
successive DST acquisitions alternate between the lower and upper banks. A full
unary broadcast init after binary broadcasts resets the active bank to the lower
bank, leaving the preceding live results in the upper bank. Repeating binary
hardware startup hides this failure by repeatedly resetting both pipelines.
Using short inits for both broadcast forms preserves the bank selected by the
acquire/commit/wait/release protocol.

The implementation retains the existing init-key analysis. The restriction that
all outputs of a sync region share the same PACK tile type still applies. This
change does not extend per-op init tracking to arbitrary nested control flow.
Legacy full-init dialect operations remain available for explicitly authored
TTKernel IR, but are not emitted by automatic init insertion.

Compiler coverage checks unconditional startup before a conditional first
region, output format and geometry changes, physical matmul source order,
input-pair changes, a loop backedge, and unary broadcast following live binary
results. Device coverage checks the mixed sequence over multiple DST
acquisitions with both accumulation widths and both sync modes. The flash-chain
test additionally exercises transpose, matmul, reduction, and normalization in
one kernel.
