# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Custom exception classes for cbsim.
"""


class CBError(RuntimeError):
    pass


class CBContractError(CBError):
    pass


class CBNotConfigured(CBError):
    pass


class CBOutOfRange(CBError):
    pass


class CBTimeoutError(CBError):
    pass
