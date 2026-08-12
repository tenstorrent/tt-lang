# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Control-flow constructs interpreted by the TT-Lang frontend."""


def static_range(*range_arguments: int) -> range:
    """Return an integer range that the frontend unrolls during compilation.

    Every argument must resolve to a compile-time integer. The supported
    signatures match :class:`range`: stop, start/stop, and start/stop/step.
    """

    return range(*range_arguments)


__all__ = ["static_range"]
