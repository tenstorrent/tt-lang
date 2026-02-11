# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pure Python implementations of TTNN golden functions.

This module provides fallback implementations for TTNN operations when the
actual TTNN library is not available. These implementations use PyTorch to
provide functionally equivalent operations for testing purposes.

All functions accept and return Tensor objects (not torch.Tensor directly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .ttnnsim import Tensor


def multiply(a: "Tensor", b: "Tensor") -> "Tensor":
    """Element-wise multiplication of two tensors.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Element-wise product of a and b
    """
    from .ttnnsim import Tensor

    result = a.to_torch() * b.to_torch()
    return Tensor(result)


def add(a: "Tensor", b: "Tensor") -> "Tensor":
    """Element-wise addition of two tensors.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Element-wise sum of a and b
    """
    from .ttnnsim import Tensor

    result = a.to_torch() + b.to_torch()
    return Tensor(result)


def subtract(a: "Tensor", b: "Tensor") -> "Tensor":
    """Element-wise subtraction of two tensors.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Element-wise difference (a - b)
    """
    from .ttnnsim import Tensor

    result = a.to_torch() - b.to_torch()
    return Tensor(result)


def isclose(
    a: "Tensor",
    b: "Tensor",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> "Tensor":
    """Element-wise comparison of two tensors within tolerances.

    Args:
        a: First input tensor
        b: Second input tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        equal_nan: Whether to consider NaN values as equal

    Returns:
        Boolean tensor indicating which elements are close
    """
    from .ttnnsim import Tensor

    result = torch.isclose(
        a.to_torch(), b.to_torch(), rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    return Tensor(result)


def repeat(input_tensor: "Tensor", shape: tuple[int, ...]) -> "Tensor":
    """Repeat tensor elements according to the given shape.

    Args:
        input_tensor: Input tensor to repeat
        shape: Number of repetitions along each dimension

    Returns:
        Tensor with elements repeated according to shape
    """
    from .ttnnsim import Tensor

    result = input_tensor.to_torch().repeat(shape)
    return Tensor(result)


def eq(a: "Tensor", b: "Tensor") -> "Tensor":
    """Element-wise equality comparison.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Boolean tensor where True indicates elements are equal
    """
    from .ttnnsim import Tensor

    result = torch.eq(a.to_torch(), b.to_torch())
    return Tensor(result)


def ne(a: "Tensor", b: "Tensor") -> "Tensor":
    """Element-wise inequality comparison.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Boolean tensor where True indicates elements are not equal
    """
    from .ttnnsim import Tensor

    result = torch.ne(a.to_torch(), b.to_torch())
    return Tensor(result)


def gt(a: "Tensor", b: "Tensor") -> "Tensor":
    """Element-wise greater-than comparison.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Boolean tensor where True indicates a > b
    """
    from .ttnnsim import Tensor

    result = torch.gt(a.to_torch(), b.to_torch())
    return Tensor(result)


def lt(a: "Tensor", b: "Tensor") -> "Tensor":
    """Element-wise less-than comparison.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Boolean tensor where True indicates a < b
    """
    from .ttnnsim import Tensor

    result = torch.lt(a.to_torch(), b.to_torch())
    return Tensor(result)


def sqrt(input_tensor: "Tensor") -> "Tensor":
    """Element-wise square root.

    Args:
        input_tensor: Input tensor

    Returns:
        Square root of each element
    """
    from .ttnnsim import Tensor

    result = torch.sqrt(input_tensor.to_torch())
    return Tensor(result)


def abs(input_tensor: "Tensor") -> "Tensor":
    """Element-wise absolute value.

    Args:
        input_tensor: Input tensor

    Returns:
        Absolute value of each element
    """
    from .ttnnsim import Tensor

    result = torch.abs(input_tensor.to_torch())
    return Tensor(result)


def exp(input_tensor: "Tensor") -> "Tensor":
    """Element-wise exponential.

    Args:
        input_tensor: Input tensor

    Returns:
        Exponential of each element
    """
    from .ttnnsim import Tensor

    result = torch.exp(input_tensor.to_torch())
    return Tensor(result)


def sin(input_tensor: "Tensor") -> "Tensor":
    """Element-wise sine.

    Args:
        input_tensor: Input tensor (in radians)

    Returns:
        Sine of each element
    """
    from .ttnnsim import Tensor

    result = torch.sin(input_tensor.to_torch())
    return Tensor(result)


def cos(input_tensor: "Tensor") -> "Tensor":
    """Element-wise cosine.

    Args:
        input_tensor: Input tensor (in radians)

    Returns:
        Cosine of each element
    """
    from .ttnnsim import Tensor

    result = torch.cos(input_tensor.to_torch())
    return Tensor(result)


def tan(input_tensor: "Tensor") -> "Tensor":
    """Element-wise tangent.

    Args:
        input_tensor: Input tensor (in radians)

    Returns:
        Tangent of each element
    """
    from .ttnnsim import Tensor

    result = torch.tan(input_tensor.to_torch())
    return Tensor(result)


def relu(input_tensor: "Tensor") -> "Tensor":
    """Element-wise ReLU activation.

    Args:
        input_tensor: Input tensor

    Returns:
        max(0, x) for each element x
    """
    from .ttnnsim import Tensor

    result = torch.relu(input_tensor.to_torch())
    return Tensor(result)


def sigmoid(input_tensor: "Tensor") -> "Tensor":
    """Element-wise sigmoid activation.

    Args:
        input_tensor: Input tensor

    Returns:
        1 / (1 + exp(-x)) for each element x
    """
    from .ttnnsim import Tensor

    result = torch.sigmoid(input_tensor.to_torch())
    return Tensor(result)


def gelu(input_tensor: "Tensor") -> "Tensor":
    """Element-wise GELU activation.

    Args:
        input_tensor: Input tensor

    Returns:
        GELU(x) for each element x
    """
    from .ttnnsim import Tensor

    result = torch.nn.functional.gelu(input_tensor.to_torch())
    return Tensor(result)


def logical_and(a: "Tensor", b: "Tensor") -> "Tensor":
    """Element-wise logical AND.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Boolean tensor with element-wise logical AND
    """
    from .ttnnsim import Tensor

    result = torch.logical_and(a.to_torch(), b.to_torch())
    return Tensor(result)


def logical_or(a: "Tensor", b: "Tensor") -> "Tensor":
    """Element-wise logical OR.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Boolean tensor with element-wise logical OR
    """
    from .ttnnsim import Tensor

    result = torch.logical_or(a.to_torch(), b.to_torch())
    return Tensor(result)
