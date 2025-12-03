# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Constants for the cbsim module.
"""
from typing import Any, cast
from .typedefs import Shape, CBID

from pydantic.fields import FieldInfo
from annotated_types import Lt  # type that holds the 'lt' constraint


def _extract_max_cbs_from_cbid() -> int:
    fi: FieldInfo = FieldInfo.from_annotation(
        cast(Any, CBID)
    )  # Cast required for type checkers

    for meta in fi.metadata:
        match meta:
            case Lt():
                value = meta.lt
                match value:
                    case int():
                        return value
                    case _:
                        raise RuntimeError(
                            f"Lt constraint value must be int, got {type(value).__name__}"
                        )
            case _:
                # Skip non-Lt metadata
                continue

    raise RuntimeError("No Lt constraint found on CBID")


MAX_CBS = _extract_max_cbs_from_cbid()

# Private tile size - use TILE_SHAPE in external code
_TILE_SIZE = 32  # Standard tile dimensions (32x32)
# TODO: Should this be a user defined option?
TILE_SHAPE: Shape = (_TILE_SIZE, _TILE_SIZE)  # Standard tile shape (32x32)

# Timeout constants for simulation operations (in seconds)
CB_DEFAULT_TIMEOUT = 1.0  # Default timeout for circular buffer operations
COPY_PIPE_TIMEOUT = 2.0  # Timeout for pipe copy operations
