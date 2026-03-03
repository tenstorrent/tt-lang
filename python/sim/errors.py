# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Custom exception classes for dfbsim.
"""


class DFBError(RuntimeError):
    pass


class DFBContractError(DFBError):
    pass


class DFBOutOfRange(DFBError):
    pass


class DFBTimeoutError(DFBError):
    pass
