# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Example source for docs/sphinx/specs/TTLangSpecification.md.
#
# The lines between the "spec:begin" and "spec:end" markers below are included
# verbatim in the specification. Regenerate the specification after editing:
#
#     python docs/sphinx/specs/build_spec.py

import ttl


def select_completed_receive(
    slow_landing_dfb,
    fast_landing_dfb,
    slow_pipe,
    fast_pipe,
    next_index,
):
    # spec:begin
    # Both receives are posted before either one is awaited. Waiting on
    # slow_request directly would block this thread even if fast_request had
    # already completed.
    slow_block = slow_landing_dfb.reserve()
    fast_block = fast_landing_dfb.reserve()
    slow_request = ttl.copy(slow_pipe, slow_block)
    fast_request = ttl.copy(fast_pipe, fast_block)

    completed = ttl.wait_any((slow_request, fast_request), start=next_index)
    selected = completed.index()

    if selected == 0:
        slow_block.push()
        with slow_landing_dfb.wait() as slow_result:
            consume(slow_result)

    if selected == 1:
        fast_block.push()
        with fast_landing_dfb.wait() as fast_result:
            consume(fast_result)

    # Begin the next selection after this candidate so a candidate earlier in
    # the tuple does not retain permanent priority when both have completed.
    next_index = (selected + 1) % 2

    # The nonselected request and its reserved block remain pending and can be
    # included in a later ttl.wait_any call or awaited directly.
    # spec:end
