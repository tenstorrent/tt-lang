# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin that turns a dispatch timeout into a handled hang.

tt-lang catches the timeout at its own launch site, but dispatch is asynchronous:
the throw can just as easily land in the next ttnn call, a blaze program run, or
a fixture teardown. This hook catches it wherever it lands in a test.

Enable per repository with, in conftest.py:

    pytest_plugins = ["ttl.hang_pytest"]
"""

from . import hang


def pytest_exception_interact(node, call, report):
    error = getattr(call.excinfo, "value", None)
    if error is not None and hang.is_dispatch_timeout(error):
        hang.handle_hang(error)
