#!/bin/bash
# Run pytest and suppress exit code 5 (no tests collected), which is expected on macOS
# when TTNN is unavailable. All tests are properly skipped via pytest.importorskip.

"$@"
EXIT_CODE=$?

# Exit code 5 means "no tests collected" - this is OK on macOS without TTNN
if [ "$EXIT_CODE" -eq 5 ]; then
    exit 0
fi

exit "$EXIT_CODE"
