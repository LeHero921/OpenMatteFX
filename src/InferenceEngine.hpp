#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <Metal/Metal.h>

// Forward declarations for ONNX Runtime types
struct OrtSession;
struct OrtEnv;
struct OrtValue;

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // Initialize engine with model paths
    void initialize(const std::string& sam2ModelPath, const std::string& refinementModelPath);
    
    // Run inference on image and point prompt
    std::vector<float> runInference(
        OFX::Clip* inputClip,
        int width,
        int height,
        double pointX,
        double pointY,
        int mode = 2 // Default to two-pass mode
    );

    bool isInitialized() const { return m_initialized; }
    bool isMetalAvailable() const { return m_metalInitialized; }

private:
    // Internal session management
    std::unique_ptr<OrtEnv, decltype(&OrtReleaseEnv)> m_env;
    std::unique_ptr<OrtSession, decltype(&OrtReleaseSession)> m_sam2Session;
    std::unique_ptr<OrtSession, decltype(&OrtReleaseSession)> m_refinementSession;

    // Metal GPU context for tensor conversion
    id<MTLDevice> m_metalDevice;
    id<MTLCommandQueue> m_commandQueue;
    id<MTLLibrary> m_metalLibrary;
    NSCache* m_bufferPool;
    
    // Model paths
    std::string m_sam2ModelPath;
    std::string m_refinementModelPath;
    
    bool m_initialized;
    bool m_metalInitialized;

    // Helper functions
    void loadModel(const std::string& modelPath, OrtSession** session);
    std::vector<float> runSAM2Inference(
        const float* imageTensor,
        int width,
        int height,
        double pointX,
        double pointY
    );
    
    std::vector<float> runRefinementInference(
        const float* imageTensor,
        const float* maskTensor,
        int width,
        int height
    );

    // Metal specific functions
    void initializeMetal();
    std::vector<float> convertOFXBufferToTensorMetal(OFX::Clip* clip, int width, int height);
    
    // Utility functions
    void checkStatus(OrtStatus* status);
};

#endif // INFERENCE_ENGINE_HPP
