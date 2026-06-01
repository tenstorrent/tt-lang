#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Append a Markdown install summary to $GITHUB_STEP_SUMMARY for the S3 PyPI
# publish workflow. With --dry-run, record that no upload occurred. With no
# $GITHUB_STEP_SUMMARY set, output goes to stdout for local invocations/tests.
#
# Usage: publish-s3-summary.sh [--dry-run] <wheel_variant> <version_override>

set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 [--dry-run] <wheel_variant> <version_override>" >&2
    exit 2
fi

variant="$1"
version="$2"
index_url="https://pypi.eng.aws.tenstorrent.com/"
pytorch_url="https://download.pytorch.org/whl/cpu"
summary_title="### Published wheels"
if [[ "$dry_run" -eq 1 ]]; then
    summary_title="### Wheel publish dry run"
fi

case "$variant" in
    light | bundled-and-light | bundled | pypi) ;;
    *)
        echo "Unknown S3 wheel variant: $variant" >&2
        exit 2
        ;;
esac

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

emit_ttlang_install() {
    local heading="$1"
    local package_spec="$2"
    if [[ -n "$heading" ]]; then
        emit <<EOF
$heading

EOF
    fi
    emit <<EOF
\`\`\`bash
pip install \\
  --extra-index-url $index_url \\
  --extra-index-url $pytorch_url \\
  $package_spec
\`\`\`
EOF
}

emit_light_install() {
    emit <<EOF
Light install:

\`\`\`bash
pip install \\
  --extra-index-url $index_url \\
  --extra-index-url $pytorch_url \\
  tt-lang-light==$version
\`\`\`

Underlying light tt-lang wheel:

\`\`\`bash
pip install \\
  --extra-index-url $index_url \\
  --extra-index-url $pytorch_url \\
  tt-lang==$version+light
\`\`\`
EOF
}

case "$variant" in
    light)
        emit_light_install
        ;;
    bundled-and-light)
        emit_ttlang_install "Bundled install:" "tt-lang==$version"
        emit <<EOF

EOF
        emit_light_install
        ;;
    bundled | pypi)
        emit_ttlang_install "" "tt-lang==$version"
        ;;
esac
