#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Force requirements-runtime.txt's numpy pin to the values on tt-lang main.
# Older tagged refs may carry a numpy constraint that no longer resolves against
# current torch/ttnn; rebuilding such a ref with this patch replaces the numpy
# requirement line(s) so the resulting wheel installs cleanly. Runs from the
# repository root (see apply-wheel-patches.sh).

set -euo pipefail

req="requirements-runtime.txt"

if [ ! -f "$req" ]; then
    echo "error: $req not found; run from the repository root." >&2
    exit 1
fi

# Drop existing numpy requirement lines -- "numpy" followed by a version
# operator, a ';' marker, whitespace, or end of line (so numpydoc / numpy-foo
# are left alone) -- then append the pins used on tt-lang main.
sed -i -E '/^[[:space:]]*numpy([[:space:]<>=!~;]|$)/d' "$req"
cat >> "$req" <<'EOF'
numpy>=1.20.0
numpy<2; platform_system == "Darwin" and platform_machine == "x86_64"
EOF

echo "Patched numpy requirement in $req:"
grep -n '^numpy' "$req"
