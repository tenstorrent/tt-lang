# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pykernel type diagnostics."""

from ttl.ir import Context, F32Type, Location, RankedTensorType
from ttl.pykernel._src.utils import (
    _as_ranked_tensor_type,
    _tensor_type_mismatch_message,
)


def test_scalar_type_is_not_downcast_to_ranked_tensor():
    """Scalar diagnostics do not access RankedTensorType-only properties."""
    with Context(), Location.unknown():
        scalar_type = F32Type.get()

        assert _as_ranked_tensor_type(scalar_type) is None
        assert (
            _tensor_type_mismatch_message(scalar_type, scalar_type)
            == "Unhandled cast from f32 to f32"
        )


def test_ranked_tensor_type_is_preserved():
    """A ranked tensor remains available to tensor-specific diagnostics."""
    with Context(), Location.unknown():
        tensor_type = RankedTensorType.get((1, 2), F32Type.get())

        assert _as_ranked_tensor_type(tensor_type) == tensor_type
