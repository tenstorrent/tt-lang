# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for comparing tensor outputs in tests."""

import torch
import math

def ulp(x: torch.Tensor) -> torch.Tensor:
    "Return Unit of Least Precision for each element of a given tensor"
    # Notes:
    # - This should be identical to the definition of ULP by Goldberg
    #   "What every computer scientist should know about floating-point arithmetic"
    #   https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
    # - We use torch.abs(x) to ensure symmetry ULP(-x) == ULP(x)
    # - For x powers of 2, x + ULP(x) is not closest number but second closest (previous number is 2x closer)
    #   However, this avoids rounding-to-nearest-tie-to-even issues on addition (i.e. x + ULP(x) != x)
    abs_x = torch.abs(x)
    next = torch.nextafter(
        abs_x, torch.tensor(math.inf, dtype=x.dtype)
    )  # 1 ULP ~ Difference between two consecutive floating point numbers
    ulp_value = next - abs_x

    # Special case: if abs_x == torch.finfo(x.dtype).max, then next == math.inf, which leads to ULP(x) == inf rather than finite number
    # We fix this problem by manually calculating ULP at max value, and masking tensor when input == max
    dtype_max = torch.finfo(x.dtype).max
    max_epsilon = dtype_max - torch.nextafter(
        torch.tensor(dtype_max, dtype=x.dtype), torch.tensor(-math.inf, dtype=x.dtype)
    )
    ulp_value = torch.where(abs_x == dtype_max, max_epsilon, ulp_value)

    return ulp_value


def comp_ulp(golden, calculated, ulp_threshold, allow_nonfinite=False):
    """
    Compute absolute error between two tensors in Units of Least Precision (ULP)
    """

    # If both tensors are empty, then we can return True
    if torch.numel(golden) == 0 and torch.numel(calculated) == 0:
        return True, "Both tensors are empty"

    # hitting this oops
    # if not allow_nonfinite and not torch.all(torch.isfinite(calculated)):
    #     return False, "Calculated tensor contains non-finite values"

    # if not _comp_nonfinite(golden, calculated):
    #     return False, "Tensors are not finite at the same positions"
    # nonfinite elements can interfere with ULP error calculation
    # To avoid this, replace nan, +inf, -inf with 0
    # (we have already checked that both tensors have the same nonfinite elements)
    mask_finite = ~torch.isfinite(golden)
    golden = golden.clone()
    calculated = calculated.clone()
    golden[mask_finite] = 0
    calculated[mask_finite] = 0

    # ULP is measured according to the golden tensor
    # In most cases, data type of golden tensor should be the same as calculated tensor.
    # However, in some cases, we may want to measure < 1 ULP differences, which requires golden tensor
    # to have higher precision than calculated tensor.
    # If we passed golden tensor to ulp() as is, we would get ULP of higher precision.
    # e.g. ulp of float32 rather bfloat16 calculation, which would give us a wrong value.
    ulp_value = ulp(golden.type(calculated.dtype))

    if (
        golden.dtype != calculated.dtype
    ):  # Note: assumes that golden has higher precision than calculated tensor
        calculated = calculated.type(golden.dtype)
        ulp_value = ulp_value.type(
            golden.dtype
        )  # Convert ULP to higher precision (for sub-1 ULP measurements)

    ulp_delta = torch.max(torch.abs(calculated - golden) / ulp_value)

    return (ulp_delta <= ulp_threshold, f"Max ULP Delta: {ulp_delta}")


# TODO: add support for ttnn.Tensor inputs when ttnn module is part of tt-lang dependencies
def assert_with_ulp(
    expected_result: torch.Tensor,
    actual_result: torch.Tensor,
    ulp_threshold=10,
    allow_nonfinite=False,
):
    """
    Assert that two tensors are similar within a given distance expressed in Units of Least Precision (ULP)

    The error is measured using the following formula:
    ``
        | expected - actual | / ULP(expected)
    ``

    Where ULP(expected) returns, for each element, the length of a single Unit of Least Precision (ULP).


    Args:
        expected_result (Union[ttnn.Tensor, torch.Tensor]): The expected reference tensor
        actual_result (Union[ttnn.Tensor, torch.Tensor]): The actual tensor to compare against the reference
        ulp_threshold (float, optional): Maximum tolerated ULP distance. Defaults to 10.
        allow_nonfinite (bool, optional): If disabled, any non-finite value (NaN, +inf, -inf) will trigger an assertion. If enabled, differences between non-finite values at the same positions will trigger an assertion.

    Notes:
        The length of a single ULP is measured using the difference between two consecutive floating point numbers.

        ULP should be preferred when errors between `calculated` and `golden` outputs are known to be small (difference < 10s of ULPs).
        This is typically the case for element-wise operations that approximate common numerical functions (e.g. exp, pow, log, ...).

        For more significant differences, where `calculated` and `golden` differ by orders of magnitude, ULPs may be harder to compare
        Indeed, with current definition, on bfloat16:
        - ULP-Delta(4, 0) = 128
        - ULP-Delta(0, 4) = 4.36e+40

        Generally, if the ULP error exceeds the 2**(#mantissa bits) (128-ULP for bfloat16, 8388608 for float32), then it means that both outputs are different by more than an order of magnitude.
        For these cases, functions such as `assert_allclose(golden, calculated, rtol, atol)` should be used instead.

        To measure the accuracy in ULP of operations on bfloat8_b data type, the ttnn bfloat8_b tensor should be either passed directly to the
        function, or converted to bfloat16 beforehand (bfloat16 has the 'same' resolution as bfloat8_b).
        Indeed, ttnn.to_torch() converts bfloat8_b to float32 by default, which would lead to assert_with_ulp() measuring ULP error as if
        data type was computed as float32.

        This should be identical to the definition of ULP by Goldberg
        "What every computer scientist should know about floating-point arithmetic"
        https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html

    Returns:
        tuple: A tuple containing:
            - ulp_passed (bool): True if ulp check passed, False otherwise
            - ulp_message (str): A message describing comparison result

    Raises:
        AssertionError: If the tensor shapes don't match or if tensor difference is greater than ulp_threshold.
    """

    assert list(expected_result.shape) == list(
        actual_result.shape
    ), f"list(expected_result.shape)={list(expected_result.shape)} vs list(actual_result.shape)={list(actual_result.shape)}"

    maximum_meaningful_ulp_thresholds = {
        torch.float64: 2**52,
        torch.float32: 2**23,
        torch.float16: 2**10,
        torch.bfloat16: 2**7,
    }
    maximum_meaningful_ulp_threshold = (
        maximum_meaningful_ulp_thresholds[torch.float32]
        if expected_result.dtype in maximum_meaningful_ulp_thresholds
        else maximum_meaningful_ulp_thresholds[expected_result.dtype]
    )

    if ulp_threshold > maximum_meaningful_ulp_threshold:
        print(
            f"ULP threshold {ulp_threshold} is greater than the maximum meaningful ULP threshold of {maximum_meaningful_ulp_threshold} for dtype {expected_result.dtype}"
        )

    ulp_passed, ulp_message = comp_ulp(
        expected_result, actual_result, ulp_threshold, allow_nonfinite
    )
    assert ulp_passed, ulp_message
    return ulp_passed, ulp_message


    """
Python implementation of get_large_matmul_params from bmm_op.hpp

This module provides functions to compute optimal matrix multiplication parameters
for distributing work across multiple cores on a Tenstorrent device.
"""

from typing import List, Tuple


def get_prime_factors(n: int) -> List[int]:
    i = 2
    prime_factors = []
    
    while i * i <= n:
        if n % i != 0:
            i += 1
        else:
            n //= i
            prime_factors.append(i)
    
    if n > 1:
        prime_factors.append(n)
    
    return prime_factors


def get_possible_products(factors: List[int]) -> List[int]:
    """
    Generate all possible products from a list of factors.
    
    This function computes all unique products that can be formed by taking
    subsets of the factors (including taking factors multiple times if they
    appear multiple times in the input).
    
    Args:
        factors: A list of prime factors
        
    Returns:
        A sorted list of all unique products that can be formed from the factors
    """
    if not factors:
        return [1]
    
    products = []
    
    for fac in factors:
        new_products = []
        
        # Add the factor itself if not already in products
        if fac not in products:
            new_products.append(fac)
        
        # Multiply factor with all existing products
        for prod in products:
            new_prod = fac * prod
            if new_prod not in products:
                new_products.append(new_prod)
        
        # Add all new products to the products list
        products.extend(new_products)
    
    # Sort products
    products.sort()
    
    return products


def get_maximum_block_dim(block_dim: int, in0_block_w: int) -> int:
    other_dim = (400 - 2 * in0_block_w * block_dim) // (2 * in0_block_w + block_dim)
    
    if other_dim > 0:
        return other_dim
    
    return 0


# Subblock choices in priority order
SUBBLOCK_HW_CHOICES = [
    (4, 2), (2, 4), (8, 1), (1, 8), (7, 1), (1, 7), (3, 2), (2, 3), (6, 1), (1, 6),
    (5, 1), (1, 5), (2, 2), (4, 1), (1, 4), (3, 1), (1, 3), (2, 1), (1, 2), (1, 1),
]


def get_large_matmul_params(
    Mt: int,
    Nt: int,
    num_cores_y: int,
    num_cores_x: int,
    in0_block_w: int
) -> Tuple[int, int, int, int]:
    """
    Compute optimal matrix multiplication parameters for multi-core execution.
    
    This function determines the per-core block sizes (Mpc, Npc) and subblock
    dimensions (subblock_h, subblock_w) for distributing a matrix multiplication
    across multiple cores while respecting memory and compute constraints.
    
    Args:
        Mt: Total number of tiles in M dimension (output rows)
        Nt: Total number of tiles in N dimension (output columns)
        num_cores_y: Number of available cores in Y dimension
        num_cores_x: Number of available cores in X dimension
        in0_block_w: Inner dimension block width (K dimension in tiles)
        
    Returns:
        A tuple (Mpc, Npc, subblock_h, subblock_w) where:
        - Mpc: Number of M tiles per core
        - Npc: Number of N tiles per core
        - subblock_h: Subblock height for compute kernel
        - subblock_w: Subblock width for compute kernel
        Returns (0, 0, 0, 0) if no valid configuration is found.
    """
    # Get prime factorizations
    Nt_fac = get_prime_factors(Nt)
    Mt_fac = get_prime_factors(Mt)
    
    Npc_min = 1
    Mpc_min = 1
    
    # Remove factors larger than available cores from Nt_fac
    # These must be handled per-core (Npc_min)
    i = 0
    while i < len(Nt_fac):
        if Nt_fac[i] > num_cores_x:
            Npc_min *= Nt_fac[i]
            Nt_fac.pop(i)
        else:
            i += 1
    
    # Remove factors larger than available cores from Mt_fac
    # These must be handled per-core (Mpc_min)
    i = 0
    while i < len(Mt_fac):
        if Mt_fac[i] > num_cores_y:
            Mpc_min *= Mt_fac[i]
            Mt_fac.pop(i)
        else:
            i += 1
    
    # Check if minimum Npc violates memory constraints
    if Npc_min > get_maximum_block_dim(Mpc_min, in0_block_w):
        return (0, 0, 0, 0)
    
    Mpc = Mpc_min
    Npc = Npc_min
    
    # Case 1: Mpc_min > 1 (M dimension has large prime factors)
    if Mpc_min > 1:
        Npc_choices = get_possible_products(Nt_fac)
        Npc_max = get_maximum_block_dim(Mpc_min, in0_block_w)
        
        # Maximize Npc within memory constraints
        for ele in Npc_choices:
            if ele * Npc_min <= Npc_max:
                Npc = ele * Npc_min
            else:
                break
        
        # Check if this fits within the core grid
        if Mt // Mpc > num_cores_y or Nt // Npc > num_cores_x:
            return (0, 0, 0, 0)
        
        # Find compatible subblock dimensions
        for subblock_h, subblock_w in SUBBLOCK_HW_CHOICES:
            if Mpc % subblock_h == 0 and Npc % subblock_w == 0:
                return (Mpc, Npc, subblock_h, subblock_w)
    
    # Case 2: Npc_min > 1 (N dimension has large prime factors)
    elif Npc_min > 1:
        Mpc_choices = get_possible_products(Mt_fac)
        Mpc_max = get_maximum_block_dim(Npc_min, in0_block_w)
        
        # Maximize Mpc within memory constraints
        for ele in Mpc_choices:
            if ele * Mpc_min <= Mpc_max:
                Mpc = ele * Mpc_min
            else:
                break
        
        # Check if this fits within the core grid
        if Mt // Mpc > num_cores_y or Nt // Npc > num_cores_x:
            return (0, 0, 0, 0)
        
        # Find compatible subblock dimensions
        for subblock_h, subblock_w in SUBBLOCK_HW_CHOICES:
            if Mpc % subblock_h == 0 and Npc % subblock_w == 0:
                return (Mpc, Npc, subblock_h, subblock_w)
    
    # Case 3: Both Mpc_min == 1 and Npc_min == 1
    else:
        Mpc_choices = get_possible_products(Mt_fac)
        Npc_choices = get_possible_products(Nt_fac)
        
        # Try different Npc values and find the largest compatible Mpc
        for Npc in Npc_choices:
            Mpc_max = get_maximum_block_dim(Npc, in0_block_w)
            
            # Find largest Mpc that fits in memory
            for ele in Mpc_choices:
                if ele <= Mpc_max:
                    Mpc = ele
            
            # Check if this configuration fits within the core grid
            if Mt // Mpc > num_cores_y or Nt // Npc > num_cores_x:
                continue
            
            # Find compatible subblock dimensions
            for subblock_h, subblock_w in SUBBLOCK_HW_CHOICES:
                if Mpc % subblock_h == 0 and Npc % subblock_w == 0:
                    return (Mpc, Npc, subblock_h, subblock_w)
    
    # No valid configuration found
    return (0, 0, 0, 0)

