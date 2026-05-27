#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Append a Markdown install summary to $GITHUB_STEP_SUMMARY for the S3 PyPI
# publish workflow. With --dry-run, record that no upload occurred. With no
# $GITHUB_STEP_SUMMARY set, output goes to stdout for local invocations/tests.
#
# Usage: publish-s3-summary.sh [--dry-run] <ttnn_dep_mode> <version_override>

set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 [--dry-run] <ttnn_dep_mode> <version_override>" >&2
    exit 2
fi

mode="$1"
version="$2"
index_url="https://pypi.eng.aws.tenstorrent.com/"
pytorch_url="https://download.pytorch.org/whl/cpu"
summary_title="### Published wheels"
if [[ "$dry_run" -eq 1 ]]; then
    summary_title="### Wheel publish dry run"
fi

emit() {
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        cat >> "$GITHUB_STEP_SUMMARY"
    else
        cat
    fi
}

emit_header() {
    emit <<EOF
$summary_title

EOF
    if [[ "$dry_run" -eq 1 ]]; then
        emit <<EOF
No wheels were uploaded.

EOF
    fi
    emit <<EOF
Package index: $index_url

EOF
}

emit_header

if [[ "$mode" == "external" ]]; then
    emit <<EOF
Light install:

\`\`\`bash
pip install \\
  --extra-index-url $index_url \\
  --extra-index-url $pytorch_url \\
  tt-lang-light==$version
\`\`\`

Underlying no-ttnn tt-lang wheel:

\`\`\`bash
pip install \\
  --extra-index-url $index_url \\
  --extra-index-url $pytorch_url \\
  tt-lang==$version+light
\`\`\`
EOF
else
    emit <<EOF
\`\`\`bash
pip install \\
  --extra-index-url $index_url \\
  --extra-index-url $pytorch_url \\
  tt-lang==$version
\`\`\`
EOF
fi
