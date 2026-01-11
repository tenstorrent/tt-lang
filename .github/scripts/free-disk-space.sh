#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Free disk space on GitHub Actions ubuntu-latest runners.
# This removes preinstalled software that is not needed for tt-lang CI.

set -e

echo "========================================"
echo "Disk usage before cleanup"
echo "========================================"
df -h /

echo ""
echo "Removing hosted tool cache (Python, Node, Go, Ruby, CodeQL, PyPy, Java)..."
sudo rm -rf /__t/* || true
sudo rm -rf /opt/hostedtoolcache || true

echo "Removing .NET SDK..."
sudo rm -rf /usr/share/dotnet || true

echo "Removing Android SDK..."
sudo rm -rf /usr/local/lib/android || true

echo "Removing Swift..."
sudo rm -rf /usr/share/swift || true

echo "Removing Haskell (GHC, ghcup)..."
sudo rm -rf /usr/local/.ghcup || true
sudo rm -rf /opt/ghc || true

echo "Removing Google Chrome..."
sudo rm -rf /opt/google/chrome || true

echo "Removing Microsoft tools (PowerShell, etc)..."
sudo rm -rf /opt/microsoft || true
sudo rm -rf /usr/local/share/powershell || true

echo "Removing Azure CLI..."
sudo rm -rf /opt/az || true

echo "Removing vcpkg..."
sudo rm -rf /usr/local/share/vcpkg || true

echo "Removing Chromium..."
sudo rm -rf /usr/local/share/chromium || true

echo "Cleaning apt cache..."
sudo apt-get clean || true

echo ""
echo "========================================"
echo "Disk usage after cleanup"
echo "========================================"
df -h /
