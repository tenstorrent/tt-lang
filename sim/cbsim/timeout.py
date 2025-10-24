"""
Timeout management for cbsim.
"""
from typing import Optional, Annotated
from pydantic import validate_call, Field

# Global timeout (seconds) used internally by blocking waits; set to None to block indefinitely.
# Useful in simulator context to avoid infinite loops.
GLOBAL_WAIT_TIMEOUT: Optional[float] = 5.0

@validate_call
def set_global_timeout(seconds: Optional[Annotated[float, Field(gt=0)]] ) -> None:
    """Set the module-wide timeout for cb_wait_front / cb_reserve_back."""
    global GLOBAL_WAIT_TIMEOUT
    GLOBAL_WAIT_TIMEOUT = seconds


def get_global_timeout() -> Optional[float]:
    """Return the current module-wide timeout used for waits."""
    return GLOBAL_WAIT_TIMEOUT
