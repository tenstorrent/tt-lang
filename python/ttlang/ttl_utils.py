# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for tt-lang."""

from typing import Union

# Mapping from kernel type strings to thread type strings
_KERNEL_TYPE_TO_THREAD_TYPE = {
    "compute": "compute",
    "datamovement": "noc",
    "ethernet": "ethernet",
}


def get_thread_type_string(input: Union[str, object]) -> str:
    """Map kernel type to thread type string.

    Handles both string kernel types and MLIR ThreadTypeAttr.

    Args:
        input: Either a string kernel type ("compute", "datamovement", "ethernet")
               or a ttkernel.ThreadTypeAttr from MLIR IR

    Returns:
        Thread type string: "compute", "noc", "ethernet"

    Raises:
        ValueError: If input is a string that's not a valid kernel type
    """
    # If it's already a string, use the dict lookup
    if isinstance(input, str):
        if input in _KERNEL_TYPE_TO_THREAD_TYPE:
            return _KERNEL_TYPE_TO_THREAD_TYPE[input]
        raise ValueError(f"Unknown kernel type: {input}")

    # For ThreadTypeAttr objects, parse the string representation
    # ThreadTypeAttr prints as #ttkernel.thread<compute> or #ttkernel.thread<noc>
    input_str = str(input)
    for thread_type in ["compute", "noc", "ethernet"]:
        if thread_type in input_str:
            return thread_type

    raise ValueError(f"Unknown thread type in attribute: {input_str}")
