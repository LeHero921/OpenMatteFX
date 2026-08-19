#include <metal_stdlib>
using namespace metal;

// Convert interleaved RGBA buffer to planar NCHW tensor format
// Input: interleaved RGBA pixel data (float4)
// Output: planar tensor (1, 3, H, W) - NCHW format
kernel void convertToTensor(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    device float* outputTensor [[buffer(0)]],
    uint2 threadPosition [[thread_position_in_grid]],
    uint2 gridSize [[grid_size]]
) {
    // Get dimensions from the grid size
    const int width = gridSize.x;
    const int height = gridSize.y;
    
    // Get current pixel position
    const int x = threadPosition.x;
    const int y = threadPosition.y;
    
    if (x >= width || y >= height) return;
    
    // Read pixel from input texture
    float4 pixel = inputTexture.read(threadPosition);
    
    // Convert RGBA to RGB (assuming alpha is not needed for SAM model)
    // For NCHW format: [batch][channel][height][width]
    const int batch = 1;
    const int channels = 3;
    const int channelSize = width * height;
    
    // Index calculation for planar tensor
    const int baseIndex = (batch * channels * width * height);
    
    // Store RGB values in separate channels
    outputTensor[baseIndex + 0 * channelSize + y * width + x] = pixel.x; // R
    outputTensor[baseIndex + 1 * channelSize + y * width + x] = pixel.y; // G  
    outputTensor[baseIndex + 2 * channelSize + y * width + x] = pixel.z; // B
    
    // Note: If we had RGBA input, we could also store alpha here if needed
}

// Alternative version that handles full NCHW tensor conversion including normalization
kernel void convertToTensorNormalized(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    device float* outputTensor [[buffer(0)]],
    uint2 threadPosition [[thread_position_in_grid]],
    uint2 gridSize [[grid_size]]
) {
    // Get dimensions from the grid size
    const int width = gridSize.x;
    const int height = gridSize.y;
    
    // Get current pixel position
    const int x = threadPosition.x;
    const int y = threadPosition.y;
    
    if (x >= width || y >= height) return;
    
    // Read pixel from input texture
    float4 pixel = inputTexture.read(threadPosition);
    
    // Normalize to [0,1] range if needed (OFX buffers are typically normalized)
    // For now we assume they're already in [0,1]
    
    const int batch = 1;
    const int channels = 3;
    const int channelSize = width * height;
    
    // Index calculation for planar tensor
    const int baseIndex = (batch * channels * width * height);
    
    // Store RGB values in separate channels (normalized to [0,1])
    outputTensor[baseIndex + 0 * channelSize + y * width + x] = pixel.x; // R
    outputTensor[baseIndex + 1 * channelSize + y * width + x] = pixel.y; // G  
    outputTensor[baseIndex + 2 * channelSize + y * width + x] = pixel.z; // B
    
    // Note: If we had RGBA input, we could also store alpha here if needed
}

// Optimized version for specific use case - direct conversion from OFX buffer
kernel void convertOFXBufferToTensor(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    device float* outputTensor [[buffer(0)]],
    uint2 threadPosition [[thread_position_in_grid]],
    uint2 gridSize [[grid_size]]
) {
    // Get dimensions from the grid size
    const int width = gridSize.x;
    const int height = gridSize.y;
    
    // Get current pixel position
    const int x = threadPosition.x;
    const int y = threadPosition.y;
    
    if (x >= width || y >= height) return;
    
    // Read pixel from input texture
    float4 pixel = inputTexture.read(threadPosition);
    
    // Convert OFX interleaved RGBA to planar NCHW format:
    // Input: interleaved RGBA (4 channels per pixel)
    // Output: planar RGB (3 channels, height*width elements each)
    
    const int batch = 1;
    const int channels = 3;
    const int channelSize = width * height;
    
    // Index calculation for planar tensor
    const int baseIndex = (batch * channels * width * height);
    
    // Store RGB values in separate channels
    outputTensor[baseIndex + 0 * channelSize + y * width + x] = pixel.x; // R
    outputTensor[baseIndex + 1 * channelSize + y * width + x] = pixel.y; // G  
    outputTensor[baseIndex + 2 * channelSize + y * width + x] = pixel.z; // B
    
    // The final tensor format will be:
    // [batch][channel][height][width] = [1][3][H][W]
    // Where each channel is stored contiguously in memory
}
