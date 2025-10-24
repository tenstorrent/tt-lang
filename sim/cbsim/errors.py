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
