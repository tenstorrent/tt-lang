#!/usr/bin/env bash
# verify-sha.sh - Compare two git SHAs with prefix-aware matching.
#
# Usage: verify-sha.sh <expected_sha> <actual_sha>
#
# Exits 0 if one SHA is a prefix of the other (i.e. they refer to the same
# commit).  Exits 1 on mismatch.  Exits 2 on usage error.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: verify-sha.sh <expected_sha> <actual_sha>" >&2
  exit 2
fi

expected="$1"
actual="$2"

# Truncate both to the shorter length, then compare.
len_e=${#expected}
len_a=${#actual}

if (( len_e <= len_a )); then
  actual="${actual:0:$len_e}"
else
  expected="${expected:0:$len_a}"
fi

if [[ "$expected" == "$actual" ]]; then
  exit 0
else
  exit 1
fi
