#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Append a Markdown install summary to $GITHUB_STEP_SUMMARY for the S3 PyPI
# publish workflow. With no $GITHUB_STEP_SUMMARY set the output goes to stdout
# (useful for local invocations and tests).
#
# Usage: publish-s3-summary.sh <ttnn_dep_mode> <pretend_version>

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <ttnn_dep_mode> <pretend_version>" >&2
    exit 2
fi

mode="$1"
version="$2"
index_url="https://pypi.eng.aws.tenstorrent.com/"
pytorch_url="https://download.pytorch.org/whl/cpu"

emit() {
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        cat >> "$GITHUB_STEP_SUMMARY"
    else
        cat
    fi
}

if [[ "$mode" == "external" ]]; then
    emit <<EOF
### Published wheels

Package index: $index_url

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
### Published wheels

Package index: $index_url

\`\`\`bash
pip install \\
  --extra-index-url $index_url \\
  --extra-index-url $pytorch_url \\
  tt-lang==$version
\`\`\`
EOF
fi
