# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
System descriptor utilities for E2E tests.

Provides shared logic for finding or generating system descriptor files.
"""

import os
from typing import Optional


def get_system_desc_path(cli_path: Optional[str] = None) -> str:
    """
    Get the system descriptor path.

    Searches in order:
    1. Explicit path provided as argument
    2. SYSTEM_DESC_PATH environment variable
    3. Auto-generate from current device

    Args:
        cli_path: Optional explicit path from CLI.

    Returns:
        Path to system descriptor file.

    Raises:
        RuntimeError: If no system descriptor can be found or generated.
    """
    # Check explicit path first.
    if cli_path and os.path.exists(cli_path):
        return cli_path

    # Check environment variable.
    env_path = os.environ.get("SYSTEM_DESC_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Generate system descriptor if not provided.
    try:
        from _ttmlir_runtime import runtime

        system_desc = runtime.get_current_system_desc()
        generated_path = "/tmp/ttlang_e2e_system.ttsys"
        system_desc.store(generated_path)
        return generated_path
    except (ImportError, Exception) as e:
        raise RuntimeError(f"Cannot get system descriptor: {e}")







