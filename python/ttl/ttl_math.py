# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL math operations namespace (ttl.math).

Re-exports elementwise operations from the generated module.
"""

# Re-export all generated elementwise operations
from ._generated_elementwise import *  # noqa: F401,F403
from ._generated_elementwise import __all__ as _generated_all
from ._src.ttl_ast import syntax
from .operators import (
    broadcast,
    fill,
    get_constant_float_value,
    get_constant_int_value,
    reduce_max,
    reduce_sum,
    transpose,
    typecast,
)
from ttl.dialects import ttl
from ttl.ir import F32Type, FloatAttr, IntegerAttr


def _get_constant_bool(val) -> bool:
    """Resolve a boolean flag from Python or MLIR constant values."""
    if isinstance(val, bool):
        return val
    iv = get_constant_int_value(val)
    if iv is None:
        raise ValueError(f"Expected constant bool, got {type(val).__name__}")
    return bool(iv)


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


@syntax("exp")
def exp(
    input,
    *,
    approx: bool = False,
    scale: float | None = None,
    skip_clamp_check: bool = False,
    iterations: int = 8,
):
    """Element-wise exponential.

    With default arguments this matches the plain hardware ``exp_tile`` (no
    approximation, no scaling, clamped). Keyword flags expose the SFPU exp
    template parameters.

    Args:
        input: Input block expression.
        approx: Enables approximate exponential evaluation.
        scale: Optional compile-time input multiplier. Computes
            ``exp(scale * input)`` when set.
        skip_clamp_check: Disables clamping of very negative inputs in
            approximate mode. Inputs below approximately -88.5 can produce
            incorrect negative outputs when enabled.
        iterations: Number of SFPU lane iterations. Defaults to 8.

    Returns:
        The element-wise exponential block expression.
    """
    from ttl.ir import BoolAttr, IntegerType

    ctx = input.type.context
    i32 = IntegerType.get_signless(32, ctx)

    approx_b = _get_constant_bool(approx)
    skip_clamp_b = _get_constant_bool(skip_clamp_check)
    iterations_i = _get_constant_int(iterations)
    scale_f = None if scale is None else _get_constant_float(scale)

    approx_attr = BoolAttr.get(True) if approx_b else None
    iterations_attr = IntegerAttr.get(i32, iterations_i) if iterations_i != 8 else None
    clamping_attr = IntegerAttr.get(i32, 0) if skip_clamp_b else None
    scale_attr = None if scale_f is None else FloatAttr.get(F32Type.get(ctx), scale_f)

    return ttl.exp(
        input.type,
        input,
        approx=approx_attr,
        scale=scale_attr,
        input_clamping=clamping_attr,
        iterations=iterations_attr,
    )


__all__ = [
    "broadcast",
    "exp",
    "fill",
    "reduce_max",
    "reduce_sum",
    "transpose",
    "typecast",
    *_generated_all,
]
