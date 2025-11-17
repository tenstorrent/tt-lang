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
        if isinstance(meta, Lt):
            value = meta.lt
            assert isinstance(
                value, int
            )  # This assertion constrains the type for type checkers
            return value

    raise RuntimeError("No Lt constraint found on CBID")


MAX_CBS = _extract_max_cbs_from_cbid()

# Private tile size - use TILE_SHAPE in external code
_TILE_SIZE = 32  # Standard tile dimensions (32x32)
# TODO: Should this be a user defined option?
TILE_SHAPE: Shape = (_TILE_SIZE, _TILE_SIZE)  # Standard tile shape (32x32)
