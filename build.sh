#!/bin/bash

# MagicMask Plugin Build Script
# This script compiles Metal shaders and builds the final OFX plugin

set -e  # Exit on any error

echo "=================================="
echo "Building MagicMask Plugin"
echo "=================================="

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
RESOURCE_DIR="$PROJECT_DIR/Resources"
SHADER_DIR="$PROJECT_DIR/shaders"

echo "Project Directory: $PROJECT_DIR"
echo "Build Directory: $BUILD_DIR"

# Clean previous build artifacts (only shader-related files)
echo "Cleaning previous build artifacts..."
rm -f "$BUILD_DIR/MetalTensorConverter.air"
rm -f "$BUILD_DIR/Resources/shaders/MetalTensorConverter.metallib"

# Create necessary directories
echo "Creating required directories..."
mkdir -p "$BUILD_DIR/Resources/shaders"

# Verify OpenFX SDK exists
echo "Checking OpenFX SDK..."
if [ ! -d "$PROJECT_DIR/include/openfx" ]; then
    echo "Error: OpenFX SDK not found at $PROJECT_DIR/include/openfx"
    echo "Please ensure you have the OpenFX SDK in include/openfx/"
    exit 1
fi

# Verify ONNX Runtime is installed
echo "Checking ONNX Runtime..."
if ! command -v onnxruntime-config &> /dev/null; then
    echo "ONNX Runtime not found. Installing via Homebrew..."
    brew install onnxruntime || true
fi

echo "=================================="
echo "Building MagicMask Plugin - Metal Shaders"
echo "=================================="

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
RESOURCE_DIR="$PROJECT_DIR/Resources"
SHADER_DIR="$PROJECT_DIR/shaders"

echo "Project Directory: $PROJECT_DIR"
echo "Build Directory: $BUILD_DIR"

# Create necessary directories
echo "Creating required directories..."
mkdir -p "$BUILD_DIR/Resources/shaders"

# Verify Metal shader source exists
if [ ! -f "$SHADER_DIR/MetalTensorConverter.metal" ]; then
    echo "Creating default Metal shader..."
    mkdir -p "$SHADER_DIR"
    cat > "$SHADER_DIR/MetalTensorConverter.metal" << 'EOF'
#include <metal_stdlib>
using namespace metal;

// Convert interleaved RGBA buffer to planar NCHW tensor format
kernel void convertToTensor(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    device float* outputTensor [[buffer(0)]],
    uint2 threadPosition [[thread_position_in_grid]],
    uint2 gridSize [[grid_size]]
) {
    const int width = gridSize.x;
    const int height = gridSize.y;
    const int x = threadPosition.x;
    const int y = threadPosition.y;
    
    if (x >= width || y >= height) return;
    
    float4 pixel = inputTexture.read(threadPosition);
    
    const int batch = 1;
    const int channels = 3;
    const int channelSize = width * height;
    const int baseIndex = (batch * channels * width * height);
    
    outputTensor[baseIndex + 0 * channelSize + y * width + x] = pixel.x; // R
    outputTensor[baseIndex + 1 * channelSize + y * width + x] = pixel.y; // G  
    outputTensor[baseIndex + 2 * channelSize + y * width + x] = pixel.z; // B
}
EOF
fi

# Compile Metal shader to .air (Intermediate representation)
echo "Compiling Metal shader to intermediate format..."
xcrun metal -c "$SHADER_DIR/MetalTensorConverter.metal" \
    -o "$BUILD_DIR/MetalTensorConverter.air" \
    -std=metal3.0 \
    -Wno-unused-function

# Compile .air to .metallib (Final library format)
echo "Compiling intermediate format to Metal library..."
xcrun metallib "$BUILD_DIR/MetalTensorConverter.air" \
    -o "$BUILD_DIR/Resources/shaders/MetalTensorConverter.metallib"

# Verify compilation success
if [ ! -f "$BUILD_DIR/Resources/shaders/MetalTensorConverter.metallib" ]; then
    echo "Error: Failed to create Metal library"
    exit 1
fi

echo "Metal shaders compiled successfully!"
echo "Output file: $BUILD_DIR/Resources/shaders/MetalTensorConverter.metallib"

# Show file details
ls -la "$BUILD_DIR/Resources/shaders/"

echo "=================================="
echo "Metal Shader Build Complete!"
echo "=================================="

# Build the plugin using existing source files
echo "Building OFX plugin..."
cd "$BUILD_DIR"

# Check if we have CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake."
    exit 1
fi

# Configure and build the plugin (this will use existing source files)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      "$PROJECT_DIR" || {
    echo "CMake configuration failed!"
    exit 1
}

# Build the plugin
echo "Building plugin..."
make || {
    echo "Build failed!"
    exit 1
}

# Verify build success
if [ -f "MagicMask.ofx" ]; then
    echo ""
    echo "=================================="
    echo "BUILD SUCCESSFUL!"
    echo "=================================="
    echo "Plugin location: $BUILD_DIR/MagicMask.ofx"
    echo ""
    echo "To install in DaVinci Resolve:"
    echo "1. Copy MagicMask.ofx to:"
    echo "   ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Plugins/"
    echo "2. Restart DaVinci Resolve"
    echo ""
    echo "Plugin structure:"
    ls -la MagicMask.ofx/
    echo ""
    echo "To clean build files: ./clean.sh"
else
    echo "Error: Plugin not created!"
    exit 1
fi

echo "=================================="
echo "Build Complete!"
echo "=================================="
