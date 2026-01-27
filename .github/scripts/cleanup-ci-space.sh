#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Clean up unnecessary files from CI runner to free up disk space
# Removes browsers, unused tools, and caches that aren't needed for Docker builds

set -e

# Remove a directory if it exists
remove_dir() {
    local dir="$1"
    local name="${2:-$(basename "$dir")}"
    if [ -d "$dir" ]; then
        echo "Removing $name..."
        sudo rm -rf "$dir"
    fi
}

# Remove a toolcache entry if it exists
remove_toolcache() {
    local tool="$1"
    local toolcache_dir="/opt/hostedtoolcache/$tool"
    if [ -d "$toolcache_dir" ]; then
        echo "Removing $tool toolcache..."
        sudo rm -rf "$toolcache_dir"
    fi
}

echo "Cleaning up CI runner space..."

# Remove browsers (not needed for Docker builds)
remove_dir "/opt/google/chrome" "Google Chrome"
remove_dir "/opt/microsoft/msedge" "Microsoft Edge"

# Remove PowerShell (not needed for Docker builds)
remove_dir "/opt/microsoft/powershell" "PowerShell"

# Remove Azure CLI (not needed for Docker builds)
remove_dir "/opt/az" "Azure CLI"

# Remove pipx (not needed if not using pipx tools)
remove_dir "/opt/pipx" "pipx"

# Remove unused hosted toolcache entries
# Keep Python as it might be needed, but remove others if not used
if [ -d "/opt/hostedtoolcache" ]; then
    echo "Cleaning up hosted toolcache..."
    remove_toolcache "CodeQL"
    remove_toolcache "Ruby"
    remove_toolcache "PyPy"
    remove_toolcache "go"
    remove_toolcache "node"
fi

# Clean up action archive cache
remove_dir "/opt/actionarchivecache" "action archive cache"

# Clean up Docker system (prune unused images, containers, volumes)
if command -v docker &> /dev/null; then
    echo "Cleaning up Docker system..."
    docker system prune -af --volumes 2>/dev/null || echo "Docker cleanup skipped (may not be running)"
fi

echo "CI space cleanup complete"
echo "Disk usage after cleanup:"
df -h
