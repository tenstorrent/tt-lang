# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mock kernel code for testing the scheduler with user code frames.

This module provides simple wrapper functions that call simulator APIs.
Since this file is NOT in /python/sim/, stack frames from these functions
will be visible to find_user_code_location() and won't be filtered out.
"""

from python.sim.greenlet_scheduler import block_if_needed


def do_wait(obj):
    """Wait on an object - wrapper that provides user code frame."""
    block_if_needed(obj, "wait")


def do_reserve(obj):
    """Reserve an object - wrapper that provides user code frame."""
    block_if_needed(obj, "reserve")
