#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Force requirements-runtime.txt's numpy pin to the values on tt-lang main.
# Older tagged refs may carry a numpy constraint that no longer resolves against
# current torch/ttnn; rebuilding such a ref with this patch replaces just the
# numpy requirement line(s) so the resulting wheel installs cleanly. Runs from
# the repository root (see apply-wheel-patches.sh).

set -euo pipefail

req="requirements-runtime.txt"

if [ ! -f "$req" ]; then
    echo "error: $req not found; run from the repository root." >&2
    exit 1
fi

# numpy requirement lines as they currently appear on tt-lang main.
tmp="$(mktemp)"
awk '
function is_numpy(line,   s, name) {
    s = line
    sub(/^[ \t]+/, "", s)
    if (s == "" || substr(s, 1, 1) == "#") return 0
    sub(/;.*$/, "", s)                 # drop environment marker
    name = s
    sub(/[<>=!~[ \t].*$/, "", name)    # keep the requirement name only
    return (tolower(name) == "numpy")
}
{
    if (is_numpy($0)) {
        if (!inserted) {
            print "numpy>=1.20.0"
            print "numpy<2; platform_system == \"Darwin\" and platform_machine == \"x86_64\""
            inserted = 1
        }
        next                           # drop the pre-existing numpy line
    }
    print
}
END {
    if (!inserted) {
        print "numpy>=1.20.0"
        print "numpy<2; platform_system == \"Darwin\" and platform_machine == \"x86_64\""
    }
}
' "$req" > "$tmp"

mv "$tmp" "$req"

echo "Patched numpy requirement in $req:"
grep -n '^numpy' "$req"
