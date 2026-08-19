# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DSL operators for tensor operations and data movement."""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

from ttl.dialects import arith, ttl
from ttl.ir import (
    Context,
    F32Type,
    BF16Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    RankedTensorType,
    Type,
)

# Re-export generated elementwise operations
from ._generated_elementwise import *  # noqa: F401,F403
from ._generated_elementwise import __all__ as _generated_all
from ._src.ttl_ast import syntax
from .condition import DispatchCondition
from .constants import DEFAULT_TILE_SIZE
from .kernel import ExternalKernelSelection, ReleaseKernelSelection
from .pipe import Pipe
from .scalar import ScalarType
from .dfb_reset import DFBReset
from .dfb_reconfiguration import DFBReconfiguration


def reconfigure_dfbs(boundary: DFBReconfiguration) -> None:
    """Enter the next compiler-derived worker-local DFB configuration epoch."""
    raise RuntimeError("ttl.reconfigure_dfbs() is valid only in a compiled kernel")


def call_extern_func(
    header: str,
    callee: str,
    *,
    template_args=None,
    func_args=None,
    dfb_dependencies=None,
    dfb_effects=None,
    dfb_accesses=None,
    unknown_dfb_access: bool = False,
    include_paths=None,
    kernel: Optional[ExternalKernelSelection] = None,
    result_type: Optional[ScalarType] = None,
    condition_result: Optional[DispatchCondition] = None,
) -> Optional[int]:
    """Call external C++ in selected logical kernels.

    Args:
        header: Header that declares the external function.
        callee: External C++ function name.
        template_args: Static values and explicit DFB wrappers emitted as C++
            template arguments.
        func_args: Values emitted as C++ function arguments. Repeated opaque
            DFBs are valid. Summarized occurrences must use distinct parameters
            of a composed operation.
        dfb_dependencies: DFBs accessed by external C++ without adding C++
            arguments. Entries must identify distinct source occurrences and
            must not repeat an automatic dependency source in ``func_args`` or
            DFB descriptor template arguments.
        dfb_effects: Optional call-wide sequence of synchronous DFB protocol
            actions performed on every call execution. A complete summary can
            permit physical-index reuse and does not emit protocol calls.
        dfb_accesses: Optional synchronous DFB inspections performed by the
            call without publishing, consuming, or changing DFB state.
        unknown_dfb_access: Whether external C++ may access unlisted
            user-managed DFBs, conservatively restricting physical-index reuse.
        include_paths: Compile-time directories added to external header
            lookup.
        kernel: Logical kernel selector or nonempty tuple of distinct
            selectors.

    ``kernel`` accepts one ``KernelKind`` or operation-local ``Kernel``.
    ``KernelKind`` values may be combined with ``|``. A nonempty tuple also
    supports multiple selectors, including operation-local kernels. The call is
    emitted once in each selected logical kernel. The unified-operation splitter
    removes the selector before AST lowering.

    ``result_type`` declares one scalar integer result as ``ScalarType.I32`` or
    ``ScalarType.I64``. Omitting it or passing ``None`` declares a void external
    function.

    ``condition_result`` declares that the result evaluates one immutable
    dispatch-stable condition. Its scalar type comes from the declaration. The
    call must be repeat-safe and cannot access DFB state.

    """
    raise RuntimeError("ttl.call_extern_func() is valid only in a compiled kernel")


def reset_dfbs(reset: DFBReset, /, *, dfbs) -> None:
    """Synchronize DFB interface owners and reset the listed interfaces.

    The operation restores pointer, initialization, and occupancy state to an
    empty queue. It preserves descriptor configuration and payload bytes. It
    makes each participating data movement RISC drain its own outstanding NoC
    commands before publishing boundary arrival. It cannot complete commands
    issued by another core or a non-participating RISC, so every producer must
    issue its required transfers before its local reset occurrence.
    """
    raise RuntimeError("ttl.reset_dfbs() is valid only in a compiled kernel")


def reset_all_dfbs(reset: DFBReset, /) -> None:
    """Apply ``reset_dfbs`` semantics to every worker-local DFB interface."""
    raise RuntimeError("ttl.reset_all_dfbs() is valid only in a compiled kernel")


class DFBEffect:
    """Ordered synchronous DFB actions performed by an external call.

    Tile and repeat counts accept static integer expressions over literals,
    integer captures, and module globals using ``+``, ``-``, ``*``, ``//``,
    and ``%``. Runtime values and booleans are invalid counts. Floor-division
    and modulo divisors must be nonzero.
    """

    @staticmethod
    def repeat(count: int, effects, /):
        """Repeat an ordered DFB-effect sequence a static nonnegative count."""
        raise RuntimeError("ttl.DFBEffect.repeat() is valid only in a compiled kernel")

    @staticmethod
    def reserve(dfb, *, tiles: int):
        """Declare a producer reservation completed by the external call."""
        raise RuntimeError("ttl.DFBEffect.reserve() is valid only in a compiled kernel")

    @staticmethod
    def push(dfb, *, tiles: int):
        """Declare a producer publication completed by the external call."""
        raise RuntimeError("ttl.DFBEffect.push() is valid only in a compiled kernel")

    @staticmethod
    def wait(dfb, *, tiles: int):
        """Declare a consumer wait completed by the external call."""
        raise RuntimeError("ttl.DFBEffect.wait() is valid only in a compiled kernel")

    @staticmethod
    def pop(dfb, *, tiles: int):
        """Declare that the external call returns consumed DFB capacity."""
        raise RuntimeError("ttl.DFBEffect.pop() is valid only in a compiled kernel")


class DFBAccess:
    """Typed synchronous DFB access by an external call."""

    @staticmethod
    def inspect(dfb):
        """Read a DFB without changing its contents or queue position."""
        raise RuntimeError("ttl.DFBAccess.inspect() is valid only in a compiled kernel")


def dfb_descriptor(dfb):
    """Use finalized DFB allocation metadata as a C++ template type."""
    raise RuntimeError("ttl.dfb_descriptor() is valid only in a compiled kernel")


def get_dfb_id(dfb):
    """Use a finalized physical DFB index as an integer value."""
    raise RuntimeError("ttl.get_dfb_id() is valid only in a compiled kernel")


def raw_addr(tensor):
    """Use a base tensor's runtime buffer address as an integer value."""
    raise RuntimeError("ttl.raw_addr() is valid only in a compiled kernel")


def _arith_constant_op(val):
    """If val is (or is the result of) an arith.constant, return the typed ConstantOp."""
    if isinstance(val, arith.ConstantOp):
        return val
    owner = getattr(val, "owner", None)
    if owner is None:
        return None
    if isinstance(owner, arith.ConstantOp):
        return owner
    if getattr(owner, "name", None) == "arith.constant":
        return arith.ConstantOp(owner)
    return None


def get_constant_int_value(val) -> Optional[int]:
    """Python analog of mlir::getConstantIntValue.

    Returns the underlying Python int when val is a Python int, an IntegerAttr,
    an arith.ConstantOp, or a Value defined by arith.constant; otherwise None.
    """
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, IntegerAttr):
        return val.value
    op = _arith_constant_op(val)
    if op is not None:
        return op.literal_value
    return None


def get_constant_float_value(val) -> Optional[float]:
    """Python analog of mlir::getConstantIntValue for floats.

    Returns the underlying Python float when val is a Python int/float, a
    FloatAttr, an arith.ConstantOp, or a Value defined by arith.constant;
    otherwise None.
    """
    if isinstance(val, bool):
        return None
    if isinstance(val, (float, int)):
        return float(val)
    if isinstance(val, FloatAttr):
        return float(val.value)
    op = _arith_constant_op(val)
    if op is not None:
        return float(op.literal_value)
    return None


def _as_host_scalar(val):
    """Return val as a Python float for host-side scalar constants."""
    return get_constant_float_value(val)


def _get_constant_int(val) -> int:
    v = get_constant_int_value(val)
    if v is None:
        raise ValueError(f"Expected constant int, got {type(val).__name__}")
    return v


def _get_constant_float(val) -> float:
    v = get_constant_float_value(val)
    if v is None:
        raise ValueError(f"Expected constant float, got {type(val).__name__}")
    return v


def _get_constant_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    value = get_constant_int_value(val)
    if value is None:
        raise ValueError(f"Expected constant bool, got {type(val).__name__}")
    return bool(value)


def _tile_hw(elem_type) -> Optional[Tuple[int, int]]:
    """Return (H, W) when ``elem_type`` is a TileType, else None."""
    from ttl.dialects import ttcore

    tile = ttcore.ir.TileType.maybe_downcast(elem_type)
    if tile is None:
        return None
    shape = list(tile.shape)
    return (int(shape[0]), int(shape[1]))


# Type aliases for common patterns
CoreCoordinate = Tuple[int, int]
IndexedTensor = Union["TensorBlock", Tuple["TensorBlock", Tuple[int, ...]]]

# Module-level grid storage for grid_size() function
# Sentinel value (-1, -1) makes uninitialized reads obvious
_current_grid: Tuple[int, int] = (-1, -1)


def _set_current_grid(grid: Tuple[int, int]) -> None:
    """Set the current grid dimensions. Called before compiling threads."""
    global _current_grid
    _current_grid = grid


def _get_current_grid() -> Tuple[int, int]:
    """Get the current grid dimensions."""
    return _current_grid


@syntax("!tensor")
class TensorBlock:
    """
    Represents a block of tensor data in the TTL dialect.

    TensorBlock supports arithmetic operations through operator
    overloading. Operations generate TTL high-level ops that get lowered
    to ttl.compute blocks.
    """

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __add__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """
        Element-wise addition using ttl.add.

        Args:
            rhs: Right operand tensor. Must have the same shape as self.

        Returns:
            Result tensor with the same shape as inputs.
        """
        return ttl.add(ast_self, rhs)

    def __sub__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Element-wise subtraction using ttl.sub."""
        return ttl.sub(ast_self, rhs)

    def __mul__(ast_self: TensorBlock, rhs) -> TensorBlock:
        """Multiplication.

        If `rhs` is a host-side scalar (Python int/float or torch 0-dim
        float tensor), emit `ttl.mul_unary_const(self, rhs)`. Otherwise
        treat `rhs` as a TensorBlock and emit `ttl.mul`.
        """
        c = _as_host_scalar(rhs)
        if c is not None:
            ctx = ast_self.type.context
            value_attr = FloatAttr.get(F32Type.get(ctx), c)
            return ttl.mul_unary_const(ast_self, value_attr)
        return ttl.mul(ast_self, rhs)

    def __rmul__(ast_self: TensorBlock, lhs) -> TensorBlock:
        """Reflected multiplication for `scalar * self`."""
        c = _as_host_scalar(lhs)
        if c is not None:
            ctx = ast_self.type.context
            value_attr = FloatAttr.get(F32Type.get(ctx), c)
            return ttl.mul_unary_const(ast_self, value_attr)
        return NotImplemented

    def __truediv__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Element-wise division using ttl.div."""
        return ttl.div(ast_self, rhs)

    def __gt__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Element-wise greater-than using ttl.gt."""
        return ttl.gt(ast_self, rhs)

    def __lt__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Element-wise less-than using ttl.lt."""
        return ttl.lt(ast_self, rhs)

    def __eq__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:  # type: ignore[override]
        """Element-wise equality using ttl.eq."""
        return ttl.eq(ast_self, rhs)

    def __ne__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:  # type: ignore[override]
        """Element-wise inequality using ttl.ne."""
        return ttl.ne(ast_self, rhs)

    def __matmul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Matrix multiplication using ttl.matmul.

        Computes C[M,N] = A[M,K] * B[K,N]. Both operands must be
        CB-attached tensors of tiles. This is sugar for the non-transposed
        ``ttl.matmul`` free function.
        """
        return _build_matmul(ast_self, rhs, transpose_rhs=False)

    def store(ast_self: TensorBlock, rhs: TensorBlock) -> None:
        """Store a result into a reserved or previously read waited block.

        A waited destination represents ordered replacement of the acquired
        pages. Compiler analysis accepts it only after proving the complete
        consumer-owned mutation contract.
        """
        if not _is_block(ast_self):
            raise ValueError(
                "store() must be called on a block acquired from reserve() or wait()"
            )
        acquired_view = _get_acquired_view_from_block(ast_self)
        _require_matching_tile_shapes(
            rhs.type.element_type,
            acquired_view.type.element_type,
            "source",
            "destination DFB",
        )
        ttl.store(rhs, acquired_view)

    def __iadd__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        """Accumulate into a reserve or replace a previously read wait.

        A reserve-backed block uses L1 packer accumulation. A wait-backed block
        reads the original value, adds ``rhs``, and stores the replacement
        without changing dataflow buffer occupancy or pointers.

        The compiler accepts waited replacement only after proving the
        complete consumer-owned mutation contract.
        """
        if not _is_block(ast_self):
            raise ValueError(
                "+= must be called on a block acquired from reserve() or wait()"
            )
        acquired_view = _get_acquired_view_from_block(ast_self)
        acquire_op_name = _get_acquire_op_name_from_view(acquired_view)
        if acquire_op_name == "ttl.cb_wait":
            ttl.store(ttl.add(ast_self, rhs), acquired_view)
            return ast_self
        if acquire_op_name != "ttl.cb_reserve":
            raise ValueError("block acquisition must be ttl.cb_reserve or ttl.cb_wait")
        ttl.store(rhs, acquired_view, accumulate=True)
        return ast_self

    def push(
        ast_self: TensorBlock,
        *,
        kernel: Optional[ReleaseKernelSelection] = None,
    ) -> None:
        """
        Signal that data is ready in the dataflow buffer (producer release).

        Finalizes a reserve() operation by signaling that the block has been
        written and is ready for consumers. This operation is non-blocking.

        Must be called on a block acquired via reserve(). ``kernel`` assigns
        an otherwise uninferable release to one logical kernel. An explicit
        thread ignores it because its decorator already determines ownership.

        Example:
            block = dfb.reserve()
            ttl.copy(data, block).wait()
            block.push()  # Signal data ready
        """
        if not _is_block(ast_self):
            raise ValueError(
                "push() must be called on a block acquired from reserve(), not a regular tensor"
            )
        cb = _get_cb_from_block(ast_self)
        ttl.cb_push(cb)

    def pop(
        ast_self: TensorBlock,
        *,
        kernel: Optional[ReleaseKernelSelection] = None,
    ) -> None:
        """
        Signal that data has been consumed (consumer release).

        Finalizes a wait() operation by signaling that the block has been
        consumed and space is available for producers. This operation is non-blocking.

        Must be called on a block acquired via wait(). ``kernel`` assigns an
        otherwise uninferable release to one logical kernel. An explicit thread
        ignores it because its decorator already determines ownership.

        Example:
            block = dfb.wait()
            result = compute(block)
            block.pop()  # Signal consumption complete
        """
        if not _is_block(ast_self):
            raise ValueError(
                "pop() must be called on a block acquired from wait(), not a regular tensor"
            )
        cb = _get_cb_from_block(ast_self)
        ttl.cb_pop(cb)


@syntax("!ttl.transfer_handle")
class CopyTransferHandler:
    """
    Transfer handle for asynchronous copy operations.

    CopyTransferHandler objects are returned by copy() calls and must be
    explicitly waited on to ensure transfer completion.
    """

    def wait(ast_self: CopyTransferHandler):
        """Block until the copy operation completes."""
        return ttl.wait(ast_self)


@syntax("!ttl.receive_request")
class ReceiveRequest:
    """Handle for one posted PipeNet receive."""

    def wait(ast_self: ReceiveRequest):
        """Block until this receive request completes."""
        return ttl.wait(ast_self)


@syntax("!ttl.ready_receive")
class ReadyReceive:
    """Completed receive selected by wait_any()."""

    def index(ast_self: ReadyReceive):
        """Return the selected request's tuple index."""
        return ttl.ready_receive_index(ast_self)


@syntax("wait_any")
def wait_any(requests, start=0) -> ReadyReceive:
    """Select the first completed receive in cyclic order from start."""
    if not isinstance(requests, tuple):
        raise TypeError("wait_any() requests must be an explicitly ordered tuple")
    if not requests:
        raise ValueError("wait_any() requires at least one receive request")
    if any(
        ttl.ReceiveRequestType.maybe_downcast(request.type) is None
        for request in requests
    ):
        raise TypeError("wait_any() accepts only PipeNet receive requests")
    if len({id(request) for request in requests}) != len(requests):
        raise ValueError("wait_any() requires distinct receive requests")
    context = requests[0].type.context
    if isinstance(start, bool):
        raise TypeError("wait_any() start must be an integer or index value")
    if isinstance(start, int):
        start = arith.ConstantOp(IndexType.get(context), start)
    elif not hasattr(start, "type"):
        raise TypeError("wait_any() start must be an integer or index value")
    elif not isinstance(start.type, (IndexType, IntegerType)):
        raise TypeError("wait_any() start must be an integer or index value")
    elif not isinstance(start.type, IndexType):
        start = arith.IndexCastOp(IndexType.get(context), start)
    ready_type = ttl.ReadyReceiveType.get(context)
    return ttl.wait_any(list(requests), start, results=[ready_type])


def _make_tensor_slice(tensor, indices, slice_shape):
    """Create a ttl.tensor_slice from a tensor, tile indices, and shape.

    Args:
        tensor: The source tensor to slice from
        indices: Tile indices for the slice start position (one per tensor dim)
        slice_shape: CB shape in tiles. May be lower rank than the tensor,
            in which case the leading ``tensor.rank - len(slice_shape)`` tensor
            dims are squeezed out of the result by scalar indexing (e.g. a
            ``(B, N, S, KVPE)`` tensor read into a ``(S, KVPE)`` block via
            ``t[b, n, s0:s1, 0:KVPE]``). The squeeze matches a numpy-style
            scalar-index reduction: the squeezed scalar index selects one slot
            in each leading dim and contributes its offset to the per-tile
            tensor coordinate. The caller's responsibility to pass scalar (not
            range) indices in the squeezed positions is enforced by
            ``_process_tensor_subscript``.
    """
    tensor_type = tensor.type
    if not isinstance(tensor_type, RankedTensorType):
        raise ValueError(f"Expected RankedTensorType, got {tensor_type}")

    if tensor_type.rank < 2:
        raise ValueError(
            f"TTL tensors must have at least 2 dimensions, got rank {tensor_type.rank}"
        )

    if len(indices) != tensor_type.rank:
        raise ValueError(
            f"Expected {tensor_type.rank} tile indices for rank-{tensor_type.rank} "
            f"tensor, got {len(indices)}"
        )

    if len(slice_shape) > tensor_type.rank:
        raise ValueError(
            f"CB shape rank ({len(slice_shape)}) must be <= tensor rank "
            f"({tensor_type.rank})"
        )

    result_type = RankedTensorType.get(
        list(slice_shape), tensor_type.element_type, tensor_type.encoding
    )
    return ttl.tensor_slice(result_type, tensor, indices)


def _is_block(value) -> bool:
    """Check if a value is a block (result of cb.reserve() or cb.wait()).

    A block is a tensor with an attached CB, produced by ttl.attach_cb.
    BlockArguments (e.g. scf.for iter_args) report a `Block` as their
    owner, not an `Operation`, so they are never blocks in this sense.
    """
    if not hasattr(value, "owner") or value.owner is None:
        return False
    if not hasattr(value.owner, "name"):
        return False
    return value.owner.name == "ttl.attach_cb"


def _is_inactive_guarded_dfb_value(value) -> bool:
    owner = getattr(value, "owner", None)
    return (
        getattr(owner, "name", None) == "builtin.unrealized_conversion_cast"
        and "ttl.inactive_guarded_dfb" in owner.attributes
    )


def _get_then_yielded_guarded_dfb_value(value):
    owner = getattr(value, "owner", None)
    if getattr(owner, "name", None) != "scf.if":
        return None

    result_number = getattr(value, "result_number", None)
    if result_number is None:
        return None

    try:
        then_block = owner.regions[0].blocks[0]
        else_block = owner.regions[1].blocks[0]
        then_yield = list(then_block.operations)[-1]
        else_yield = list(else_block.operations)[-1]
    except (IndexError, TypeError):
        return None

    if then_yield.name != "scf.yield" or else_yield.name != "scf.yield":
        return None
    if result_number >= len(then_yield.operands) or result_number >= len(
        else_yield.operands
    ):
        return None
    if not _is_inactive_guarded_dfb_value(else_yield.operands[result_number]):
        return None
    return then_yield.operands[result_number]


def _get_acquire_op_name_from_view(value):
    while True:
        owner = getattr(value, "owner", None)
        owner_name = getattr(owner, "name", None)
        if owner_name in ("ttl.cb_reserve", "ttl.cb_wait"):
            return owner_name
        if owner_name == "ttl.attach_cb":
            value = owner.operands[0]
            continue
        guarded_value = _get_then_yielded_guarded_dfb_value(value)
        if guarded_value is None:
            return None
        value = guarded_value


def _get_acquired_view_from_block(block):
    """Extract the reserve or wait view from a block.

    The attach_cb op has signature: (tensor, cb) -> tensor
    So the reserve/wait tensor is operand[0].
    """
    if block.owner.name != "ttl.attach_cb":
        raise ValueError(f"expected block from ttl.attach_cb, got {block.owner.name}")
    acquired_view = block.owner.operands[0]
    if _get_acquire_op_name_from_view(acquired_view) is None:
        raise ValueError(
            "ttl.attach_cb tensor must come from ttl.cb_reserve or ttl.cb_wait"
        )
    return acquired_view


def _get_cb_from_block(block):
    """Extract the CB from a block (result of ttl.attach_cb).

    The attach_cb op has signature: (tensor, cb) -> tensor
    So the CB is operand[1].
    """
    if block.owner.name != "ttl.attach_cb":
        raise ValueError(f"expected block from ttl.attach_cb, got {block.owner.name}")
    return block.owner.operands[1]


def _get_cb_shape(cb_val):
    """Extract the block shape from a CB value."""
    cb_type = ttl.CircularBufferType.maybe_downcast(cb_val.type)
    if cb_type is None:
        raise ValueError(f"Expected CircularBufferType, got {cb_val.type}")
    return list(cb_type.shape)


def _require_matching_tile_shapes(lhs_elem, rhs_elem, lhs_name: str, rhs_name: str):
    """Reject copies/stores with inconsistent tilization.

    Both non-tile is allowed. Mixing tile and non-tile, or two different
    tile shapes, is an error.
    """
    lhs = _tile_hw(lhs_elem)
    rhs = _tile_hw(rhs_elem)
    if lhs is None and rhs is None:
        return
    if lhs is None or rhs is None:
        raise ValueError(
            f"cannot mix tiled and non-tiled element types; got "
            f"{lhs_name}={lhs_elem}, {rhs_name}={rhs_elem}"
        )
    if lhs != rhs:
        raise ValueError(
            f"{lhs_name} tile shape {lhs[0]}x{lhs[1]} must match "
            f"{rhs_name} tile shape {rhs[0]}x{rhs[1]}"
        )


def _process_tensor_subscript(subscript_tuple, cb_shape):
    """Process tensor subscript and create tensor slice.

    Args:
        subscript_tuple: (tensor, indices) where indices are [(value, is_range), ...]
        cb_shape: Shape from the CB. Its rank may be less than the tensor rank;
            the leading (tensor_rank - cb_rank) dims are then squeezed via
            scalar indices and the trailing dims map to the CB shape.

    Returns:
        Tensor slice with shape matching cb_shape
    """
    tensor, indices = subscript_tuple

    tensor_type = tensor.type
    if not isinstance(tensor_type, RankedTensorType):
        raise ValueError(f"Expected RankedTensorType, got {tensor_type}")

    expected_indices = tensor_type.rank
    if len(indices) != expected_indices:
        raise ValueError(
            f"Expected {expected_indices} indices for rank-{tensor_type.rank} "
            f"tensor, got {len(indices)}"
        )

    cb_is_multi_tile = any(d > 1 for d in cb_shape)
    rank_diff = expected_indices - len(cb_shape)
    if rank_diff > 0:
        for d in range(rank_diff):
            if indices[d][1]:
                raise ValueError(
                    f"slice rank reduction: leading squeezed index {d} must "
                    f"be scalar (e.g. t[batch, 0, kc:..., 0:...]), got range "
                    f"syntax for a tensor dim being squeezed to match a "
                    f"rank-{len(cb_shape)} CB"
                )
        trailing_indices = indices[rank_diff:]
    else:
        trailing_indices = indices
    uses_ranges = any(is_range for _, is_range in trailing_indices)

    if cb_is_multi_tile and not uses_ranges:
        raise ValueError(
            f"CB shape {cb_shape} requires range syntax "
            f"(e.g., tensor[0:2, 0:2]), but got index syntax"
        )

    # TODO: Validate that range size matches CB shape (requires runtime or
    # constant folding to compare end - start with cb_shape dimensions).

    start_indices = [value for value, _ in indices]
    return _make_tensor_slice(tensor, start_indices, cb_shape)


def _is_pipe(val):
    """Check if a value is a pipe reference."""
    if not hasattr(val, "type"):
        return isinstance(val, Pipe) and hasattr(val, "_mlir_value")
    if ttl.PipeType.maybe_downcast(val.type):
        return True
    if ttl.SelectedPipeSrcType.maybe_downcast(val.type):
        return True
    if ttl.SelectedPipeDstType.maybe_downcast(val.type):
        return True
    return isinstance(val, Pipe) and hasattr(val, "_mlir_value")


def _get_pipe_mlir_value(pipe):
    """Get the MLIR value for a pipe reference."""
    if not hasattr(pipe, "type"):
        return pipe._mlir_value
    if ttl.PipeType.maybe_downcast(pipe.type):
        return pipe
    if ttl.SelectedPipeSrcType.maybe_downcast(pipe.type):
        return pipe
    if ttl.SelectedPipeDstType.maybe_downcast(pipe.type):
        return pipe
    return pipe._mlir_value


@syntax("copy")
def copy(src, dst) -> Union[CopyTransferHandler, ReceiveRequest]:
    """
    Initiate an asynchronous data transfer using ttl.copy.

    Args:
        src: Source tensor/slice (for reads), block (for writes), or Pipe (for pipe receive)
        dst: Destination block (for reads), tensor/slice (for writes), or Pipe (for pipe send)

    Returns:
        ReceiveRequest for a PipeNet receive; CopyTransferHandler otherwise.

    For multi-tile CBs (shape > 1x1), use range syntax: tensor[0:2, 0:2]
    For single-tile CBs (shape 1x1), use index syntax: tensor[0, 0]

    For pipe transfers:
        ttl.copy(block, pipe) - send from DFB block to pipe
        ttl.copy(pipe, block) - receive from pipe to DFB block
    """
    # Check for pipe operands first
    src_is_pipe = _is_pipe(src)
    dst_is_pipe = _is_pipe(dst)

    if src_is_pipe or dst_is_pipe:
        # Pipe transfer: CB <-> Pipe
        if src_is_pipe and dst_is_pipe:
            raise ValueError("copy() cannot transfer directly between two pipes")

        if dst_is_pipe:
            # DFB -> Pipe send.
            if not _is_block(src):
                raise ValueError(
                    "copy() to pipe requires block src (from cb.reserve() or cb.wait())"
                )
            src_cb = _get_cb_from_block(src)
            pipe_val = _get_pipe_mlir_value(dst)
            ctx = src_cb.type.context
            xf_type = Type.parse("!ttl.transfer_handle<write>", ctx)
            return ttl.copy(xf_type, src_cb, pipe_val)
        else:
            # Pipe -> DFB receive. The sender writes into the receiver-owned block.
            if not _is_block(dst):
                raise ValueError(
                    "copy() from pipe requires block dst (from cb.reserve() or cb.wait())"
                )
            pipe_val = _get_pipe_mlir_value(src)
            ctx = dst.type.context
            xf_type = ttl.ReceiveRequestType.get(ctx)
            return ttl.copy(xf_type, pipe_val, dst)

    # Non-pipe transfers: tensor subscript <-> block
    src_is_subscript = isinstance(src, tuple)
    dst_is_subscript = isinstance(dst, tuple)

    # Identify the block argument to get CB shape
    if dst_is_subscript:
        if not _is_block(src):
            raise ValueError("copy() with tensor subscript dst requires block src")
        cb_shape = _get_cb_shape(_get_cb_from_block(src))
    elif src_is_subscript:
        if not _is_block(dst):
            raise ValueError("copy() with tensor subscript src requires block dst")
        cb_shape = _get_cb_shape(_get_cb_from_block(dst))
    else:
        raise ValueError(
            "copy() requires at least one tensor subscript argument "
            "(e.g., tensor[row, col] or tensor[r0:r1, c0:c1])"
        )

    # Process subscripted tensors into tensor slices
    if src_is_subscript:
        src = _process_tensor_subscript(src, cb_shape)
    if dst_is_subscript:
        dst = _process_tensor_subscript(dst, cb_shape)

    ctx = src.type.context

    # Check if src/dst is a block (result of cb.reserve()/cb.wait())
    src_is_block = _is_block(src)
    dst_is_block = _is_block(dst)

    # Extract CB from block if needed
    src_cb = _get_cb_from_block(src) if src_is_block else None
    dst_cb = _get_cb_from_block(dst) if dst_is_block else None

    if dst_is_block and not src_is_block:
        # Read: device tensor/slice -> block (CB)
        dst_cb_ty = ttl.CircularBufferType.maybe_downcast(dst_cb.type)
        if dst_cb_ty is None:
            raise ValueError(f"Expected CircularBufferType, got {dst_cb.type}")
        _require_matching_tile_shapes(
            src.type.element_type, dst_cb_ty.element_type, "tensor", "CB"
        )
        xf_type = Type.parse("!ttl.transfer_handle<read>", ctx)
        return ttl.copy(xf_type, src, dst_cb)
    elif src_is_block and not dst_is_block:
        # Write: block (CB) -> device tensor/slice
        src_cb_ty = ttl.CircularBufferType.maybe_downcast(src_cb.type)
        if src_cb_ty is None:
            raise ValueError(f"Expected CircularBufferType, got {src_cb.type}")
        _require_matching_tile_shapes(
            dst.type.element_type, src_cb_ty.element_type, "tensor", "CB"
        )
        xf_type = Type.parse("!ttl.transfer_handle<write>", ctx)
        return ttl.copy(xf_type, src_cb, dst)
    else:
        raise ValueError(
            f"copy() requires exactly one block argument (result of cb.reserve() or cb.wait()). "
            f"Got src_is_block={src_is_block}, dst_is_block={dst_is_block}"
        )


@syntax("node")
def node(*, dims):
    """
    Get the coordinates of the current core.

    Currently only dims=1 and dims=2 are supported (temporary restriction).

    Args:
        dims: Number of dimensions to return (must be 1 or 2)

    Returns:
        For dims=2: Tuple (x, y) where x is column coordinate and y is row coordinate
        For dims=1: The node's index within the flattened grid

    Raises:
        ValueError: If dims is not 1 or 2

    Example:
        x, y = ttl.node(dims=2)
        n = ttl.node(dims=1)
    """
    dims_val = _get_constant_int(dims)
    if dims_val not in (1, 2):
        raise ValueError(
            f"core() currently only supports dims=1 and dims=2, got dims={dims_val}. "
            "Multi-dimensional grids are not yet supported."
        )
    x = ttl.core_x()
    if dims_val == 2:
        return (x, ttl.core_y())
    # The specification orders the second coordinate contiguously.
    rows = _get_current_grid()[1]
    ctx = x.type.context
    stride = arith.ConstantOp(IndexType.get(ctx), rows).result
    column_base = arith.MulIOp(x, stride).result
    return arith.AddIOp(column_base, ttl.core_y()).result


@syntax("grid_size")
def grid_size(*, dims):
    """
    Get the size of the grid.

    Currently only dims=1 and dims=2 are supported (temporary restriction).

    Args:
        dims: Number of dimensions to return (must be 1 or 2)

    Returns:
        For dims=2: Tuple (x_size, y_size) where x_size is columns and y_size is rows
        For dims=1: The total number of nodes in the grid

    Raises:
        ValueError: If dims is not 1 or 2

    Example:
        x_size, y_size = ttl.grid_size(dims=2)
        total = ttl.grid_size(dims=1)
    """
    dims_val = _get_constant_int(dims)
    if dims_val not in (1, 2):
        raise ValueError(
            f"grid_size() currently only supports dims=1 and dims=2, got dims={dims_val}. "
            "Multi-dimensional grids are not yet supported."
        )
    # grid is stored as (cols, rows) = (x, y), matching tt-metal convention
    cols, rows = _get_current_grid()
    if dims_val == 2:
        return (cols, rows)
    return cols * rows


@syntax("signpost")
def signpost(name: str):
    """
    Mark a profiling scope visible in Tracy.

    Use as a context manager to wrap a region of interest:

        with ttl.signpost("my_region"):
            ...

    Generates a DeviceZoneScopedN in the emitted C++ code. Enable
    TTLANG_SIGNPOST_PROFILE=1 to collect per-region cycle counts.

    Args:
        name: Name for the profiling region (must be a string literal)
    """
    return ttl.signpost(name)


@syntax("broadcast")
def broadcast(input: TensorBlock, *, dims: List[int], shape) -> TensorBlock:
    """
    Broadcast a block over specified dimensions to a target shape.

    Matches the spec form ``ttl.block.broadcast(expr, dims, shape)``. For
    tiled blocks, broadcast happens in two steps: intra-tile scalar broadcast
    for any innermost dimension listed in ``dims``, and inter-tile broadcast
    for every other dimension where the target shape is greater than 1.

    ``dims`` uses Python-style indexing: each index must lie in
    ``[-rank, rank-1]``. Every dimension ``d`` in ``dims`` must have
    ``input.shape[d] == 1``; every dimension not in ``dims`` must equal the
    corresponding ``shape`` entry.

    Args:
        input: Input tensor (CB-attached)
        dims: Dimensions to broadcast over
        shape: Target shape of the result

    Returns:
        Result tensor with broadcast values
    """
    from ttl.ir import DenseI64ArrayAttr

    if not dims:
        raise ValueError("dims must be a non-empty list of dimension indices")

    if not isinstance(input.type, RankedTensorType):
        raise ValueError(f"broadcast input must be a ranked tensor, got {input.type}")

    rank = input.type.rank
    # Inside @ttl.compute(), int literals in a tuple come through as
    # arith.ConstantOp values; unwrap to Python ints for verifier checks
    # and the DenseI64ArrayAttr.
    shape_list = [_get_constant_int(s) for s in shape]
    if len(shape_list) != rank:
        raise ValueError(
            f"shape size {len(shape_list)} does not match input rank {rank}"
        )

    norm_dims = set()
    for d in dims:
        if d < -rank or d >= rank:
            raise ValueError(
                f"Invalid broadcast dimension {d}: for rank-{rank} tensors, "
                f"each index must satisfy {-rank} <= dim <= {rank - 1}"
            )
        norm_dims.add(d + rank if d < 0 else d)

    input_shape = list(input.type.shape)
    for i in range(rank):
        if i in norm_dims:
            if input_shape[i] != 1:
                raise ValueError(
                    f"broadcast dim {i} requires input shape 1, got "
                    f"{input_shape[i]}"
                )
        elif input_shape[i] != shape_list[i]:
            raise ValueError(
                f"non-broadcast dim {i}: input has {input_shape[i]} but shape "
                f"has {shape_list[i]}"
            )

    result_type = RankedTensorType.get(
        shape_list, input.type.element_type, input.type.encoding
    )

    dims_attr = DenseI64ArrayAttr.get(list(dims))
    shape_attr = DenseI64ArrayAttr.get(shape_list)
    return ttl.block_broadcast(result_type, input, dims_attr, shape_attr)


def _warn_if_reduce_shape_omitted(shape) -> None:
    if shape is not None:
        return
    warnings.warn(
        "Omitting the reduce shape argument is deprecated; pass shape explicitly",
        DeprecationWarning,
        stacklevel=3,
    )


def _reduce_impl(
    input: TensorBlock,
    dims: List[int],
    reduce_type: int,
    shape=None,
) -> TensorBlock:
    """Shared implementation for reduce_sum and reduce_max."""
    from ttl.ir import IntegerAttr, IntegerType, DenseI64ArrayAttr

    input_type = input.type
    input_shape = list(input_type.shape)
    rank = len(input_shape)
    if rank < 2:
        raise ValueError(f"reduce requires rank 2 or greater, got rank {rank}")
    if not dims:
        raise ValueError("dims must be non-empty")

    for d in dims:
        if d < -rank or d >= rank:
            raise ValueError(
                f"dim {d} out of range for rank {rank}: "
                f"must be in [{-rank}, {rank - 1}]"
            )
    norm_dims = sorted({d % rank for d in dims})

    expected_shape = [1 if i in norm_dims else s for i, s in enumerate(input_shape)]
    if shape is None:
        # Keep accepting the legacy compiler spelling while supporting the
        # explicit result shape required by the language specification.
        result_shape = expected_shape
    else:
        result_shape = [_get_constant_int(s) for s in shape]
        if len(result_shape) != rank:
            raise ValueError(
                f"reduce shape {tuple(result_shape)} has {len(result_shape)} "
                f"dimensions but input has rank {rank}"
            )
        if result_shape != expected_shape:
            raise ValueError(
                f"reduce shape {tuple(result_shape)} does not match expected "
                f"result shape {tuple(expected_shape)} (input shape "
                f"{tuple(input_shape)}, reducing dims {dims})"
            )

    result_type = RankedTensorType.get(
        result_shape, input_type.element_type, input_type.encoding
    )

    ctx = input_type.context
    i32_type = IntegerType.get_signless(32, ctx)
    reduce_type_attr = IntegerAttr.get(i32_type, reduce_type)
    dims_attr = DenseI64ArrayAttr.get(dims, ctx)
    scaler_type = RankedTensorType.get(
        [1, 1], input_type.element_type, input_type.encoding
    )
    scaler = ttl.fill(scaler_type, FloatAttr.get(F32Type.get(ctx), 1.0))
    return ttl.reduce(result_type, input, scaler, reduce_type_attr, dims_attr)


@syntax("reduce_sum")
def reduce_sum(input: TensorBlock, *, dims: List[int], shape=None) -> TensorBlock:
    """Sum reduction over specified dimensions.

    ``shape`` is the result shape required by the language specification. It
    must be 1 in reduced dimensions and match the input in all other
    dimensions. Omitting it is deprecated; it is currently inferred for
    backward compatibility.

    To scale the result by a constant, multiply: `c * reduce_sum(x, dims=...)`.
    """
    _warn_if_reduce_shape_omitted(shape)
    return _reduce_impl(input, dims, reduce_type=0, shape=shape)


@syntax("reduce_max")
def reduce_max(input: TensorBlock, *, dims: List[int], shape=None) -> TensorBlock:
    """Max reduction over specified dimensions.

    ``shape`` is the result shape required by the language specification. It
    must be 1 in reduced dimensions and match the input in all other
    dimensions. Omitting it is deprecated; it is currently inferred for
    backward compatibility.

    To scale the result by a constant, multiply: `c * reduce_max(x, dims=...)`.
    """
    _warn_if_reduce_shape_omitted(shape)
    return _reduce_impl(input, dims, reduce_type=1, shape=shape)


def _resolve_transpose_flag(val) -> bool:
    """Resolve a transpose keyword into a Python bool.

    Inside ``@ttl.compute``, ``True``/``False`` literals are lowered to i1
    ``arith.constant`` SSA values by the AST compiler, so accept either a
    Python bool (from the default argument) or a constant int/i1 value.
    """
    if isinstance(val, bool):
        return val
    iv = get_constant_int_value(val)
    if iv is not None:
        return bool(iv)
    raise ValueError("transpose_rhs must be a compile-time boolean constant")


def _build_matmul(lhs: TensorBlock, rhs: TensorBlock, *, transpose_rhs: bool):
    """Build a ttl.matmul op, computing the result shape from the operands.

    For the non-transposed form ``rhs`` is ``[K, N]``; for the transposed
    form (``transpose_rhs``) ``rhs`` is ``[N, K]`` and the matmul computes
    ``C[M, N] = A[M, K] * B[N, K]^T``.
    """
    transpose = _resolve_transpose_flag(transpose_rhs)
    lhs_type = lhs.type
    rhs_type = rhs.type
    lhs_shape = list(lhs_type.shape)
    rhs_shape = list(rhs_type.shape)
    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        raise ValueError(
            f"matmul requires rank-2 operands, got lhs rank {len(lhs_shape)} "
            f"and rhs rank {len(rhs_shape)}"
        )

    # K is lhs columns. For transposed rhs [N, K], K is rhs.shape[1] and N is
    # rhs.shape[0]; otherwise rhs is [K, N], so K is rhs.shape[0] and N is
    # rhs.shape[1].
    rhs_k = rhs_shape[1] if transpose else rhs_shape[0]
    if lhs_shape[1] != rhs_k:
        raise ValueError(
            f"matmul K dimension mismatch: lhs has {lhs_shape[1]} columns but "
            f"rhs has {rhs_k} {'columns' if transpose else 'rows'}"
        )

    from ttl.dialects import ttcore

    lhs_tile = ttcore.ir.TileType.maybe_downcast(lhs_type.element_type)
    rhs_tile = ttcore.ir.TileType.maybe_downcast(rhs_type.element_type)
    if lhs_tile is None or rhs_tile is None:
        raise ValueError(
            "matmul requires tile-typed operands, got "
            f"lhs={lhs_type.element_type}, rhs={rhs_type.element_type}"
        )
    lhs_dtype = ttcore.DataType(lhs_tile.data_type_as_int)
    rhs_dtype = ttcore.DataType(rhs_tile.data_type_as_int)
    is_bfloat16_by_bfp = (
        not transpose
        and lhs_dtype == ttcore.DataType.BFloat16
        and rhs_dtype in (ttcore.DataType.BFP_BFloat4, ttcore.DataType.BFP_BFloat8)
    )
    if lhs_dtype != rhs_dtype and not is_bfloat16_by_bfp:
        raise ValueError(
            "unsupported matmul operand tile data type combination: "
            f"lhs={lhs_type.element_type}, rhs={rhs_type.element_type}"
        )

    lhs_tile_height, lhs_tile_width = map(int, lhs_tile.shape)
    rhs_tile_height, rhs_tile_width = map(int, rhs_tile.shape)
    rhs_tile_k = rhs_tile_width if transpose else rhs_tile_height
    if lhs_tile_width != rhs_tile_k:
        raise ValueError(
            "matmul tile K dimension mismatch: lhs tile width "
            f"{lhs_tile_width} does not match rhs tile "
            f"{'width' if transpose else 'height'} {rhs_tile_k}"
        )

    n = rhs_shape[0] if transpose else rhs_shape[1]
    result_shape = [lhs_shape[0], n]
    result_tile_width = rhs_tile_height if transpose else rhs_tile_width
    result_tile = ttcore.ir.TileType.get(
        lhs_type.context, lhs_tile_height, result_tile_width, lhs_dtype
    )
    result_type = RankedTensorType.get(result_shape, result_tile, lhs_type.encoding)
    if transpose:
        return ttl.matmul(result_type, lhs, rhs, transpose_rhs=True)
    return ttl.matmul(result_type, lhs, rhs)


@syntax("matmul")
def matmul(lhs: TensorBlock, rhs: TensorBlock, *, transpose_rhs=False) -> TensorBlock:
    """Matrix multiply two CB-attached tensors of tiles.

    Computes ``C[M, N] = A[M, K] * B[K, N]``. When ``transpose_rhs`` is set,
    ``rhs`` is provided as ``[N, K]`` and the matmul computes
    ``C[M, N] = A[M, K] * B[N, K]^T`` using the hardware transpose path.
    """
    return _build_matmul(lhs, rhs, transpose_rhs=transpose_rhs)


@syntax("transpose")
def transpose(input: TensorBlock) -> TensorBlock:
    """Transpose a 2D block: (M, N) -> (N, M)."""
    input_type = input.type
    input_shape = list(input_type.shape)
    if len(input_shape) != 2:
        raise ValueError(
            f"transpose only supports 2D tensors, got rank {len(input_shape)}"
        )
    result_shape = [input_shape[1], input_shape[0]]
    result_type = RankedTensorType.get(
        result_shape, input_type.element_type, input_type.encoding
    )
    return ttl.transpose(result_type, input)


@syntax("fill")
def fill(
    value,
    *,
    shape,
    dtype=None,
    tile=(DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE),
) -> TensorBlock:
    """Produce a block of ``shape`` filled with a constant value.

    ``dtype`` selects the per-element dtype and defaults to bf16. ``tile``
    selects TT-Metal-constructible physical dimensions and defaults to 32x32;
    the compiler validates target-specific fill support. The downstream
    ``ttl.store`` determines the output DFB used during lowering; no output
    operand is required.
    """
    from ttl.dialects import ttcore
    from .dtype_utils import normalize_tile_dimensions
    from .dtype_utils import tensor_dtype_to_ttcore_datatype

    fill_val = _get_constant_float(value)
    shape_list = [_get_constant_int(s) for s in shape]
    if not shape_list:
        raise ValueError("fill requires a non-empty shape")
    if any(s <= 0 for s in shape_list):
        raise ValueError(f"fill shape must be all-positive, got {tuple(shape_list)}")
    tile_dimensions = normalize_tile_dimensions(
        tuple(_get_constant_int(dimension) for dimension in tile)
    )

    if dtype is None:
        ttcore_dtype = ttcore.DataType.BFloat16
    elif isinstance(dtype, ttcore.DataType):
        ttcore_dtype = dtype
    else:
        ttcore_dtype = tensor_dtype_to_ttcore_datatype(dtype)

    ctx = Context.current
    tile_type = ttcore.ir.TileType.get(ctx, *tile_dimensions, ttcore_dtype)
    result_type = RankedTensorType.get(shape_list, tile_type)
    value_attr = FloatAttr.get(F32Type.get(ctx), fill_val)
    return ttl.fill(result_type, value_attr)


def _is_supported_typecast_dtype(ttcore_dtype) -> bool:
    from ttl.dialects import ttcore

    return ttcore_dtype in {
        ttcore.DataType.Float32,
        ttcore.DataType.BFloat16,
        ttcore.DataType.BFP_BFloat8,
        ttcore.DataType.BFP_BFloat4,
    }


def _is_supported_typecast_tile_type(tile_type) -> bool:
    from ttl.dialects import ttcore

    return _is_supported_typecast_dtype(ttcore.DataType(tile_type.data_type_as_int))


@syntax("typecast")
def typecast(input: TensorBlock, dtype) -> TensorBlock:
    """
    Elementwise typecast: convert each element of ``input`` to ``dtype``.

    Args:
        input: Input tensor (CB-attached). Each element is a tile.
        dtype: Target data type. Accepts a ``ttcore.DataType`` enum value
            or a torch/ttnn dtype convertible via ``dtype_utils``.

    Returns:
        Result tensor with the same shape as ``input`` but with the element
        type derived from ``dtype``.
    """
    from ttl.dialects import ttcore
    from .dtype_utils import tensor_dtype_to_ttcore_datatype

    if isinstance(dtype, ttcore.DataType):
        ttcore_dtype = dtype
    else:
        ttcore_dtype = tensor_dtype_to_ttcore_datatype(dtype)
    if not _is_supported_typecast_dtype(ttcore_dtype):
        raise ValueError(
            f"typecast only supports floating-point destination dtypes, got {dtype}"
        )

    input_type = input.type
    if not isinstance(input_type, RankedTensorType):
        raise ValueError(f"typecast expects a RankedTensorType input, got {input_type}")

    ctx = input_type.context
    input_tile = ttcore.ir.TileType.maybe_downcast(input_type.element_type)
    if input_tile is None:
        raise ValueError(
            f"typecast expects tile-typed elements, got {input_type.element_type}"
        )
    if not _is_supported_typecast_tile_type(input_tile):
        raise ValueError(
            "typecast only supports floating-point input tile dtypes, got "
            f"{input_tile}"
        )

    out_tile_type = ttcore.ir.TileType.get(
        ctx, input_tile.shape[0], input_tile.shape[1], ttcore_dtype
    )
    result_type = RankedTensorType.get(
        input_type.shape, out_tile_type, input_type.encoding
    )
    return ttl.typecast(result_type, input)


@syntax("exp")
def exp(
    input: TensorBlock,
    *,
    approx: bool = False,
    scale: Optional[float] = None,
    skip_clamp_check: bool = False,
    iterations: int = 8,
) -> TensorBlock:
    """Element-wise exponential.

    With default arguments this matches the plain hardware ``exp_tile`` (no
    approximation, no scaling, clamped). Keyword flags expose the SFPU exp
    template parameters:

    Args:
        input: Input tensor (CB-attached). Each element is a tile.
        approx: Enable the fast approximate exp.
        scale: Optional scale factor ``s``; when set the op computes
            ``exp(s * x)``. ``None`` (default) disables scaling.
        skip_clamp_check: When ``True``, disables clamping of very negative
            inputs (``InputClamping::None``): faster, but inputs below ~-88.5
            produce incorrect (guaranteed-negative) outputs. Only meaningful
            with ``approx=True``. Defaults to ``False`` (``ClampToNegative``).
        iterations: Number of SFPU lane iterations (default 8).

    Returns:
        Result tensor with the same shape and dtype as ``input``.
    """
    from ttl.ir import BoolAttr, IntegerType

    ctx = input.type.context
    i32 = IntegerType.get_signless(32, ctx)

    # Flag literals passed inside a compute body arrive as arith.constant
    # values, so resolve them through the constant helpers.
    approx_b = _get_constant_bool(approx)
    skip_clamp_b = _get_constant_bool(skip_clamp_check)
    iterations_i = _get_constant_int(iterations)
    scale_f = None if scale is None else _get_constant_float(scale)

    # Pass None for any flag left at its default so the op keeps its plain
    # spelling (the ODS default applies). input_clamping is an integer-backed
    # enum attribute (built like ttl.reduce's reduce_type); None=0,
    # ClampToNegative=1.
    approx_attr = BoolAttr.get(True) if approx_b else None
    iterations_attr = IntegerAttr.get(i32, iterations_i) if iterations_i != 8 else None
    clamping_attr = IntegerAttr.get(i32, 0) if skip_clamp_b else None
    scale_attr = None if scale_f is None else FloatAttr.get(F32Type.get(ctx), scale_f)

    return ttl.exp(
        input,
        approx=approx_attr,
        scale=scale_attr,
        input_clamping=clamping_attr,
        iterations=iterations_attr,
    )


def _get_block_scalar_type(block):
    """Extract the scalar MLIR type from a block's tensor element type.

    For tiled blocks (!ttcore.tile<H, W, dtype>), returns the corresponding
    scalar type (f32 for Float32, bf16 for BFloat16).
    For row-major blocks, returns the element type directly.
    """
    from ttl.dialects import ttcore
    from ttl.ir import BF16Type

    block_type = block.type
    if not isinstance(block_type, RankedTensorType):
        raise ValueError(f"Expected RankedTensorType block, got {block_type}")

    elem_type = block_type.element_type
    tile_type = ttcore.ir.TileType.maybe_downcast(elem_type)
    if tile_type is not None:
        dtype = ttcore.DataType(tile_type.data_type_as_int)
        ctx = block_type.context
        if dtype == ttcore.DataType.Float32:
            return F32Type.get(ctx)
        if dtype == ttcore.DataType.BFloat16:
            return BF16Type.get(ctx)
        raise ValueError(
            f"raw element access only supports f32 and bf16, got tile dtype {dtype}"
        )
    if elem_type == F32Type.get(block_type.context):
        return elem_type
    if elem_type == BF16Type.get(block_type.context):
        return elem_type
    raise ValueError(
        f"raw element access only supports f32 and bf16, got element type {elem_type}"
    )


def _as_index_values(block, coords):
    context = block.type.context
    index_type = IndexType.get(context)
    index_values = []
    for coord in coords:
        if isinstance(coord, int):
            index_values.append(arith.ConstantOp(index_type, coord))
        elif hasattr(coord, "type") and isinstance(coord.type, IndexType):
            index_values.append(coord)
        else:
            index_values.append(arith.IndexCastOp(index_type, coord))
    return index_values


@syntax("raw_element_read")
def raw_element_read(block, *coords):
    """Read a scalar element from a block at flat coordinates.

    Coordinates are scalar-element positions within the block. The number
    of coordinates must match the block tensor rank.

    For tiled blocks, lowering decomposes them into tile + intra-tile offsets.

    Only supported in data movement (noc) threads.

    Args:
        block: Block tensor (from cb.reserve() or cb.wait())
        *coords: Index values matching the block tensor rank

    Returns:
        Scalar value matching the block's element dtype
    """
    if len(coords) < 1:
        raise ValueError("raw_element_read requires at least one coordinate")
    scalar_type = _get_block_scalar_type(block)
    return ttl.raw_element_read(scalar_type, block, _as_index_values(block, coords))


@syntax("read_index")
def read_index(block, *coords):
    """Read a nonnegative scalar element as an index.

    Coordinates follow ``raw_element_read``. Fractional values truncate
    toward zero. The source value must be finite, nonnegative, and no greater
    than INT32_MAX; behavior is undefined otherwise.

    Only supported in data movement (noc) threads.
    """
    if len(coords) < 1:
        raise ValueError("read_index requires at least one coordinate")
    # Validate before op construction so unsupported dtypes raise in Python.
    _get_block_scalar_type(block)
    return ttl.read_index(block, _as_index_values(block, coords))


@syntax("raw_element_write")
def raw_element_write(block, *args):
    """Write a scalar value to a block at flat coordinates.

    Coordinates are scalar-element positions within the block. The number
    of coordinates must match the block tensor rank. The last argument
    is the value to write; all preceding arguments are coordinates.

    For tiled blocks, lowering decomposes them into tile + intra-tile offsets.
    If the value is f32 and the block element type is bf16, the value is
    implicitly truncated (precision loss is expected).

    Only supported in data movement (noc) threads.

    Args:
        block: Block tensor (from cb.reserve() or cb.wait())
        *args: N index values followed by the scalar value to write.

    Example:
        ttl.raw_element_write(block, row, col, val)
    """

    if len(args) < 2:
        raise ValueError(
            "raw_element_write requires at least one coordinate and a value"
        )
    coord_args = args[:-1]
    val = args[-1]
    ctx = block.type.context
    index_vals = _as_index_values(block, coord_args)

    block_scalar_type = _get_block_scalar_type(block)
    if hasattr(val, "type") and val.type != block_scalar_type:
        if val.type == F32Type.get(ctx) and block_scalar_type == BF16Type.get(ctx):
            val = arith.TruncFOp(block_scalar_type, val)

    ttl.raw_element_write(block, index_vals, val)


__all__ = [
    "TensorBlock",
    "CopyTransferHandler",
    "ReceiveRequest",
    "ReadyReceive",
    "copy",
    "wait_any",
    "core",
    "grid_size",
    "signpost",
    "matmul",
    "fill",
    "typecast",
    "exp",
    "raw_element_read",
    "raw_element_write",
    "read_index",
    "call_extern_func",
    "DFBEffect",
    "DFBAccess",
    "dfb_descriptor",
    "get_dfb_id",
    "raw_addr",
    *_generated_all,
]
