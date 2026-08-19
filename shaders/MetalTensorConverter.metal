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
