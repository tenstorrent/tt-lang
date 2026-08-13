# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN GlobalSemaphore type and address validation."""


def is_ttnn_global_semaphore(value: object) -> bool:
    """Require the exact TTNN type so local and lookalike objects are rejected."""
    try:
        from ttnn._ttnn.global_semaphore import global_semaphore
    except ImportError:
        return False
    return isinstance(value, global_semaphore)


def get_ttnn_global_semaphore_address(value: object) -> int:
    """Return one GlobalSemaphore address without suppressing TTNN failures."""
    if not is_ttnn_global_semaphore(value):
        raise TypeError(f"expected ttnn GlobalSemaphore, got {type(value)}")
    import ttnn

    address = ttnn.get_global_semaphore_address(value)
    if type(address) is not int:
        raise TypeError(
            "ttnn.get_global_semaphore_address() must return one integer "
            f"address, got {type(address)}"
        )
    if not 0 <= address < (1 << 32):
        raise ValueError(
            "ttnn.get_global_semaphore_address() result must fit in uint32_t, "
            f"got {address}"
        )
    return address
