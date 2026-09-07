# Broadcast initialization

The init insertion pass separates common hardware setup from per-operation
broadcast configuration. Common setup configures PACK and DST synchronization
before a sync region. Unary and binary broadcast inits configure only their
UNPACK and MATH pipelines, after reconfiguring both source formats and tile
geometry through the Metal `reconfig_data_format` API.

Consecutive broadcasts with the same operation kind, broadcast dimension, and
input values share an init under the existing init-key algorithm. Each new
broadcast init reconfigures its inputs unconditionally. In particular, the first
pair in a sync region cannot assume that the common init's formats are still
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

The implementation retains the existing common-init placement and init-key
analysis. Common initialization remains per sync region, with compiler-loop
hoisting; migrating all operator families to exactly one hardware startup at
kernel entry is a separate change. The existing restriction that all outputs of
a sync region share the same PACK format still applies. This change does not
extend init tracking to arbitrary nested control flow.

Compiler coverage checks input-pair changes, a loop backedge, and a unary
broadcast following live binary results. Device coverage checks the mixed
sequence over multiple DST acquisitions with both accumulation widths and both
sync modes. The flash-chain test additionally exercises the sequence in a
multi-node reduction and normalization kernel.
