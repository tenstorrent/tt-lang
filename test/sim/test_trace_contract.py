# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Schema-conformance contract test for the simulator trace.

The cycle estimator (``sim_stats``) reads events from the trace but never imports
the simulator -- the trace file is the only contract, so nothing catches a
producer-side rename at import time. These tests fail the moment the estimator's
expected events (``parse.CONSUMED_EVENTS``) drift from the registry the producer
defines in ``sim/trace.py``.
"""

from sim.trace import ALL_CATEGORIES, _EVENT_CATEGORY
from python.sim_stats.cycles.parse import CONSUMED_EVENTS


def test_consumed_events_exist_in_producer_registry() -> None:
    """Every event the estimator reads must be defined by sim/trace.py."""
    unknown = CONSUMED_EVENTS - set(_EVENT_CATEGORY)
    assert not unknown, (
        f"cycle estimator reads events absent from sim/trace.py: {sorted(unknown)}. "
        "The producer renamed/removed an event, or parse.CONSUMED_EVENTS drifted."
    )


def test_consumed_events_recorded_under_all_categories() -> None:
    """A full-tracing run must actually record every event the estimator needs."""
    missing = {
        e for e in CONSUMED_EVENTS if _EVENT_CATEGORY.get(e) not in ALL_CATEGORIES
    }
    assert not missing, (
        f"events {sorted(missing)} are consumed by the estimator but their category "
        "is not enabled by ALL_CATEGORIES; a full-tracing run would omit them."
    )
