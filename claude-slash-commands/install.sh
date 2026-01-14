#!/bin/bash
# Install TT-Lang slash commands for Claude Code
# This script copies command files to ~/.claude/commands/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$HOME/.claude/commands"

# List of command files to install
COMMANDS=(
    "ttl-help.md"
    "ttl-import.md"
    "ttl-export.md"
    "ttl-optimize.md"
    "ttl-profile.md"
    "ttl-simulate.md"
    "ttl-test.md"
)

echo "TT-Lang Slash Commands Installer"
echo "================================="
echo ""
echo "This will install the following commands to $TARGET_DIR:"
echo ""
for cmd in "${COMMANDS[@]}"; do
    echo "  - /${cmd%.md}"
done
echo ""

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Target directory does not exist."
    echo ""
    echo "Command to run:"
    echo "  mkdir -p $TARGET_DIR"
    echo ""
    read -p "Create directory and continue? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
    mkdir -p "$TARGET_DIR"
    echo "Directory created."
    echo ""
fi

# Show what will be copied
echo "Commands to run:"
for cmd in "${COMMANDS[@]}"; do
    echo "  cp $SCRIPT_DIR/$cmd $TARGET_DIR/$cmd"
done
echo ""

# Confirm installation
read -p "Proceed with installation? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

# Copy files
echo ""
echo "Installing commands..."
for cmd in "${COMMANDS[@]}"; do
    cp "$SCRIPT_DIR/$cmd" "$TARGET_DIR/$cmd"
    echo "  Installed: /${cmd%.md}"
done

echo ""
echo "Installation complete!"
echo ""
echo "Available commands:"
echo "  /ttl-help     - List all available TT-Lang commands"
echo "  /ttl-import   - Import CUDA/Triton/PyTorch kernel to TT-Lang"
echo "  /ttl-export   - Export TT-Lang kernel to TT-Metal C++"
echo "  /ttl-optimize - Profile and optimize kernel performance"
echo "  /ttl-profile  - Run profiler and show per-line cycle counts"
echo "  /ttl-simulate - Run simulator and suggest improvements"
echo "  /ttl-test     - Generate tests for a kernel"
echo ""
echo "Use these commands in Claude Code by typing the command name,"
echo "e.g., '/ttl-import my_cuda_kernel.cu'"
