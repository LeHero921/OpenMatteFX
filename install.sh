#!/bin/bash

# Install plugin to DaVinci Resolve
echo "Installing MagicMask plugin..."

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
PLUGIN_FILE="$BUILD_DIR/MagicMask.ofx"

if [ ! -f "$PLUGIN_FILE" ]; then
    echo "Error: Plugin not found at $PLUGIN_FILE"
    echo "Please build the plugin first with ./build.sh"
    exit 1
fi

# Check if DaVinci Resolve plugins directory exists
DAVINCI_PLUGINS_DIR="$HOME/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Plugins"

if [ ! -d "$DAVINCI_PLUGINS_DIR" ]; then
    echo "Warning: DaVinci Resolve plugins directory not found at $DAVINCI_PLUGINS_DIR"
    echo "Creating directory..."
    mkdir -p "$DAVINCI_PLUGINS_DIR"
fi

# Copy plugin
echo "Copying plugin to DaVinci Resolve..."
cp -r "$PLUGIN_FILE" "$DAVINCI_PLUGINS_DIR/"

if [ $? -eq 0 ]; then
    echo "Plugin installed successfully!"
    echo "Please restart DaVinci Resolve to use MagicMask."
else
    echo "Error: Failed to install plugin"
    exit 1
fi
