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
