#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Append a Markdown install summary to $GITHUB_STEP_SUMMARY for the S3 PyPI
# publish workflow. With --dry-run, record that no upload occurred. With
# --index-subdir <subdir> points install commands at a simple index. Use
# --find-links-subdir <subdir> for generated wheel views consumed with
# pip --find-links, including `tt-lang/<YYYY-MM>/` and `tt-lang/releases/`.
# With no $GITHUB_STEP_SUMMARY set, output goes to stdout for local
# invocations/tests.
#
# Usage: publish-s3-summary.sh [--dry-run] [--dry-run-if true|false] [--index-subdir <subdir>] [--find-links-subdir <subdir>] <wheel_variant> <version_override>

set -euo pipefail

usage() {
    echo "Usage: $0 [--dry-run] [--dry-run-if true|false] [--index-subdir <subdir>] [--find-links-subdir <subdir>] <wheel_variant> <version_override>" >&2
    exit 2
}

dry_run=0
index_subdir=""
find_links_subdir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --dry-run-if)
            [[ $# -ge 2 ]] || usage
            case "$2" in
                true) dry_run=1 ;;
                false) dry_run=0 ;;
                *) usage ;;
            esac
            shift 2
            ;;
        --index-subdir)
            [[ $# -ge 2 ]] || usage
            index_subdir="$2"
            shift 2
            ;;
        --find-links-subdir)
            [[ $# -ge 2 ]] || usage
            find_links_subdir="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -ne 2 ]]; then
    usage
fi
if [[ -n "$index_subdir" && -n "$find_links_subdir" ]]; then
    echo "--index-subdir and --find-links-subdir are mutually exclusive" >&2
    exit 2
fi

variant="$1"
version="$2"
index_url="https://pypi.eng.aws.tenstorrent.com/"
index_label="Package index"
if [[ -n "$index_subdir" ]]; then
    index_url="https://pypi.eng.aws.tenstorrent.com/${index_subdir}/"
fi
find_links_url=""
if [[ -n "$find_links_subdir" ]]; then
    find_links_url="https://pypi.eng.aws.tenstorrent.com/${find_links_subdir}/"
    index_url="$find_links_url"
    index_label="Wheel directory"
fi
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
$index_label: $index_url

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
    if [[ -n "$find_links_url" ]]; then
        emit <<EOF
\`\`\`bash
pip install \\
  --find-links $find_links_url \\
  --extra-index-url $pytorch_url \\
  $package_spec
\`\`\`
EOF
        return
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
    if [[ -n "$find_links_url" ]]; then
        emit <<EOF
Light install:

\`\`\`bash
pip install \\
  --find-links $find_links_url \\
  --extra-index-url $pytorch_url \\
  tt-lang-light==$version
\`\`\`

Underlying light tt-lang wheel:

\`\`\`bash
pip install \\
  --find-links $find_links_url \\
  --extra-index-url $pytorch_url \\
  tt-lang==$version+light
\`\`\`
EOF
        return
    fi
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
