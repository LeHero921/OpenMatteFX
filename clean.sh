#!/bin/bash

# Clean build files script
echo "Cleaning build files..."

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
    echo "Build directory cleaned."
else
    echo "No build directory to clean."
fi

echo "Clean complete!"
