#!/bin/bash

# MagicMask Plugin Build Script
# This script compiles Metal shaders, builds the final OFX plugin, and logs the output

set -e  # Exit on any error

# 1. Pfade definieren
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
LOG_DIR="$BUILD_DIR/logs"
RESOURCE_DIR="$PROJECT_DIR/Resources"
SHADER_DIR="$PROJECT_DIR/shaders"

# 2. Log-Verzeichnis erstellen und Logging aktivieren
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/build_$(date +%Y%m%d_%H%M%S).log"

# Ab hier wird der gesamte Output ins Terminal UND in die Logdatei geschrieben
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================="
echo "Building MagicMask Plugin"
echo "=================================="
echo "Project Directory: $PROJECT_DIR"
echo "Build Directory: $BUILD_DIR"
echo "Log File: $LOG_FILE"
echo "=================================="

# 3. Vorherige Build-Artefakte aufräumen
echo "Cleaning previous build artifacts..."
rm -f "$BUILD_DIR/MetalTensorConverter.air"
rm -f "$BUILD_DIR/Resources/shaders/MetalTensorConverter.metallib"

# Verzeichnisse erstellen
mkdir -p "$BUILD_DIR/Resources/shaders"

# 4. Abhängigkeiten prüfen
echo "Checking OpenFX SDK..."
if [ ! -d "$PROJECT_DIR/include/openfx" ]; then
    echo "Error: OpenFX SDK not found at $PROJECT_DIR/include/openfx"
    exit 1
fi

echo "Checking ONNX Runtime..."
if ! command -v onnxruntime-config &> /dev/null; then
    echo "ONNX Runtime not found. Installing via Homebrew..."
    brew install onnxruntime || true
fi

# 5. Metal Shader manuell kompilieren (als Backup)
echo "=================================="
echo "Compiling Metal Shaders"
echo "=================================="

if [ ! -f "$SHADER_DIR/MetalTensorConverter.metal" ]; then
    echo "Error: Metal shader source not found!"
    exit 1
fi

xcrun metal -c "$SHADER_DIR/MetalTensorConverter.metal" \
    -o "$BUILD_DIR/MetalTensorConverter.air" \
    -std=metal3.0 \
    -Wno-unused-function

xcrun metallib "$BUILD_DIR/MetalTensorConverter.air" \
    -o "$BUILD_DIR/Resources/shaders/MetalTensorConverter.metallib"

if [ ! -f "$BUILD_DIR/Resources/shaders/MetalTensorConverter.metallib" ]; then
    echo "Error: Failed to create Metal library"
    exit 1
fi
echo "Metal shaders compiled successfully!"

# 6. CMake Build
echo "=================================="
echo "Building OFX Plugin via CMake"
echo "=================================="
cd "$BUILD_DIR"

if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake."
    exit 1
fi

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      "$PROJECT_DIR" || {
    echo "CMake configuration failed!"
    exit 1
}

make -j$(sysctl -n hw.logicalcpu) || {
    echo "Build failed!"
    exit 1
}

# 7. Erfolgsprüfung
if [ -d "MagicMaskPlugin.ofx.bundle" ]; then
    echo ""
    echo "=================================="
    echo "BUILD SUCCESSFUL!"
    echo "=================================="
    echo "Plugin location: $BUILD_DIR/MagicMaskPlugin.ofx.bundle"
    echo "Log saved to: $LOG_FILE"
    echo ""
    echo "To install in DaVinci Resolve:"
    echo "1. Run this command to copy the plugin:"
    echo "   sudo cp -r MagicMaskPlugin.ofx.bundle /Library/OFX/Plugins/"
    echo "2. Restart DaVinci Resolve"
    echo ""
else
    echo "Error: Plugin bundle not created! Check the log file at $LOG_FILE."
    exit 1
fi