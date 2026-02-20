#!/bin/bash
# Install TT-Lang slash commands and tools for Claude Code
# This script copies command files to ~/.claude/commands/ and tools to ~/.claude/commands/tools/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMANDS_DIR="$HOME/.claude/commands"
TOOLS_DIR="$COMMANDS_DIR/tools"

# List of command files to install
COMMANDS=(
    "ttl-help.md"
    "ttl-import.md"
    "ttl-export.md"
    "ttl-optimize.md"
    "ttl-profile.md"
    "ttl-simulate.md"
    "ttl-test.md"
    "ttl-bug.md"
)

# List of tool scripts to install
TOOLS=(
    "_lib.sh"
    "run-test.sh"
    "copy-file.sh"
    "copy-from-remote.sh"
    "remote-run.sh"
    "smoke-test.sh"
)

echo "TT-Lang Slash Commands Installer"
echo "================================="
echo ""
echo "This will install:"
echo ""
echo "  Commands (to $COMMANDS_DIR):"
for cmd in "${COMMANDS[@]}"; do
    echo "    /${cmd%.md}"
done
echo ""
echo "  Tools (to $TOOLS_DIR):"
for tool in "${TOOLS[@]}"; do
    echo "    $tool"
done
echo ""

# Create directories if needed
for dir in "$COMMANDS_DIR" "$TOOLS_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Creating $dir"
        mkdir -p "$dir"
    fi
done

# Confirm installation
read -p "Proceed with installation? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

# Install commands
echo ""
echo "Installing commands..."
for cmd in "${COMMANDS[@]}"; do
    cp "$SCRIPT_DIR/$cmd" "$COMMANDS_DIR/$cmd"
    echo "  Installed: /${cmd%.md}"
done

# Install tools
echo ""
echo "Installing tools..."
for tool in "${TOOLS[@]}"; do
    cp "$SCRIPT_DIR/tools/$tool" "$TOOLS_DIR/$tool"
    chmod +x "$TOOLS_DIR/$tool"
    echo "  Installed: $tool"
done

# Copy config example if no config exists
if [ ! -f "$TOOLS_DIR/remote.conf" ]; then
    cp "$SCRIPT_DIR/tools/remote.conf.example" "$TOOLS_DIR/remote.conf.example"
    echo ""
    echo "================================================"
    echo "SETUP REQUIRED: Configure your remote connection"
    echo "================================================"
    echo ""
    echo "  cp $TOOLS_DIR/remote.conf.example $TOOLS_DIR/remote.conf"
    echo "  \$EDITOR $TOOLS_DIR/remote.conf"
    echo ""
    echo "Set REMOTE_SHELL to your remote execution command, e.g.:"
    echo "  SSH + Docker:  REMOTE_SHELL=\"ssh user@server docker exec -i container\""
    echo "  Direct SSH:    REMOTE_SHELL=\"ssh user@server\""
    echo "  Lima VM:       REMOTE_SHELL=\"limactl shell ttsim --\""
    echo ""
    echo "Then verify with: $TOOLS_DIR/smoke-test.sh"
else
    echo ""
    echo "remote.conf already exists, skipping config setup."
fi

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
echo "  /ttl-bug      - File a bug report with reproducer"
echo ""
echo "Use these commands in Claude Code by typing the command name,"
echo "e.g., '/ttl-import my_cuda_kernel.cu'"
