#include "InferenceEngine.hpp"
#include <iostream>
#include <vector>
#include <CoreFoundation/CoreFoundation.h>
#include <ofxImageEffect.h>
#include <ofxProperty.h>
#include <ofxParam.h>
#include <ofxCore.h>
#include <ofxInteract.h>
#include "Support/ofxsLog.h"
#include "Support/ofxsMessage.h"
#include "Support/ofxsProcessing.h"
#include "Support/ofxsMultiThread.h"
#include "Support/ofxsMemory.h"

// Plugin info
#define PLUGIN_NAME "OpenMatteFX"
#define PLUGIN_VERSION_MAJOR 1
#define PLUGIN_VERSION_MINOR 0
#define PLUGIN_VERSION_MICRO 0

// Plugin ID
#define PLUGIN_ID "com.quantuumstudios.openmattefx"

// Parameter names
#define PARAM_SELECTION_POINT "selectionPoint"
#define PARAM_INFERENCE_MODE "inferenceMode"
#define PARAM_RESET_MASK "resetMask"

// Inference modes
enum InferenceMode {
    FAST = 0,
    HIGH_DETAIL = 1,
    TWO_PASS = 2
};

class MagicMaskPlugin : public OFX::ImageEffect {
private:
    OFX::ChoiceParam* m_inferenceMode;
    OFX::Double2DParam* m_selectionPoint;
    OFX::PushButtonParam* m_resetMask;
    std::unique_ptr<InferenceEngine> m_engine;

    // --- HIER MUSS DIE HILFSFUNKTION HIN (mit double time) ---
    void applyMaskToOutput(OFX::Clip* outputClip, const std::vector<float>& mask, int width, int height, double time) {
        if (mask.empty() || width <= 0 || height <= 0 || !outputClip) return;
        
        try {
            OfxRectD rod = outputClip->getRegionOfDefinition(time);
            OfxRectI bounds;
            bounds.x1 = static_cast<int>(rod.x1);
            bounds.y1 = static_cast<int>(rod.y1);
            bounds.x2 = static_cast<int>(rod.x2);
            bounds.y2 = static_cast<int>(rod.y2);
            
            // Hier nutzen wir jetzt das 'time' statt 'args.time'
            OFX::Image* outputImg = outputClip->fetchImage(time);
            if (!outputImg) return;
            
            void* pixelData = outputImg->getPixelData();
            int rowBytes = outputImg->getRowBytes();
            
            // Rest deines Codes zum Anwenden der Maske...
            
            delete outputImg; // C++ Support Wrapper nutzt delete statt releaseImage
            
        } catch (const std::exception& e) {
            std::cerr << "Error applying mask to output: " << e.what() << std::endl;
        }
    }

public:
    MagicMaskPlugin(OfxImageEffectHandle handle) 
        : OFX::ImageEffect(handle), 
          m_inferenceMode(nullptr),
          m_selectionPoint(nullptr),
          m_resetMask(nullptr) {
        
        // Get parameters
        m_inferenceMode = fetchChoiceParam(PARAM_INFERENCE_MODE);
        m_selectionPoint = fetchDouble2DParam(PARAM_SELECTION_POINT);
        m_resetMask = fetchPushButtonParam(PARAM_RESET_MASK);
        
        // Set default values
        if (m_inferenceMode) {
            m_inferenceMode->setValue(2); // Default to two-pass mode
        }
        if (m_selectionPoint) {
            m_selectionPoint->setValue(0.5, 0.5); // Center of image
        }
    }

    virtual void changedParam(const OFX::InstanceChangedArgs &args, const std::string &paramName) override {
        if (paramName == PARAM_RESET_MASK) {
            // Reset mask logic here if needed
            // For now just clear the mask state or reset to initial point
            m_selectionPoint->setValue(0.5, 0.5);
        }
    }

    virtual bool isIdentity(const OFX::IsIdentityArguments &args, OFX::Clip *&identityClip, double &identityTime) override {
        // Implement identity check logic if applicable
        return false;
    }

    virtual void getClipPreferences(OFX::ClipPreferencesSetter &args) override {
        // Set clip preferences for better performance
        // args.setOutputClipPreferences().setUnmapped(true);
    }

    virtual void render(const OFX::RenderArguments &args) override {
        // Get input and output clips
        OFX::Clip* srcClip = fetchClip("Source");
        OFX::Clip* outputClip = fetchClip("Output");
        
        if (!srcClip || !outputClip) return;
        
        // Get frame dimensions
        OfxRectD rod = srcClip->getRegionOfDefinition(args.time);
        OfxRectI bounds;
        bounds.x1 = static_cast<int>(rod.x1);
        bounds.y1 = static_cast<int>(rod.y1);
        bounds.x2 = static_cast<int>(rod.y2); // bzw. rod.x2
        bounds.y2 = static_cast<int>(rod.y2);
        int width = bounds.x2 - bounds.x1;
        int height = bounds.y2 - bounds.y1;
        
        // Validate dimensions
        if (width <= 0 || height <= 0) return;
        
        // Get parameter values
        double pointX, pointY;
        m_selectionPoint->getValueAtTime(args.time, pointX, pointY);
        
        int mode = 0;
        if (m_inferenceMode) {
            m_inferenceMode->getValueAtTime(args.time, mode);
        }
        
        // Perform inference using Metal GPU acceleration
        try {
            if (!m_engine || !m_engine->isInitialized()) {
                throw std::runtime_error("Engine not initialized");
            }
            
            // Run AI inference pipeline - this now includes Metal tensor conversion
            std::vector<float> resultMask = m_engine->runInference(
                srcClip, 
                width, height, 
                pointX, pointY, 
                (InferenceMode)mode
            );
            
            // Apply mask to output clip
            applyMaskToOutput(outputClip, resultMask, width, height, args.time);
            
        } catch (const std::exception& e) {
            // Log error but don't crash the host application
            std::cerr << "Error during inference: " << e.what() << std::endl;
            OFX::throwSuiteStatusException(kOfxStatErrFatal);
        }
    }

private:
    // Helper to get resource path
    std::string getResourcePath(const std::string& relativePath) {
        char* bundlePath = new char[1024];
        CFBundleRef mainBundle = CFBundleGetMainBundle();
        CFURLRef bundleURL = CFBundleCopyBundleURL(mainBundle);
        CFURLGetFileSystemRepresentation(bundleURL, true, (UInt8*)bundlePath, 1024);
        std::string fullPath = std::string(bundlePath) + "/Contents/Resources/" + relativePath;
        delete[] bundlePath;
        return fullPath;
    }
    
//     // Helper to apply mask
//     // Apply generated mask to output clip
// void MagicMaskPlugin::applyMaskToOutput(OFX::Clip* outputClip, const std::vector<float>& mask, int width, int height, double time) {
//     if (mask.empty() || width <= 0 || height <= 0 || !outputClip) return;
    
//     try {
//         // Get the bounds of the frame
//         OfxRectI bounds;
//         bounds.x1 = 0;
//         bounds.y1 = 0;
//         bounds.x2 = width;
//         bounds.y2 = height;
        
//         // Fetch the output image for writing
//         OFX::Image* outputImg = outputClip->fetchImage(time);
//         if (!outputImg) return;
        
//         // Get image properties
//         OfxRectI imgBounds = outputImg->getBounds();
//         int imgWidth = imgBounds.x2 - imgBounds.x1;
//         int imgHeight = imgBounds.y2 - imgBounds.y1;
        
//         // Get pixel data pointer
//         void* pixelData = outputImg->getPixelData();
//         int bytesPerPixel = outputImg->getRowBytes();
//         int rowBytes = outputImg->getRowBytes();
        
//         // Apply mask to alpha channel (assuming RGBA format)
//         if (bytesPerPixel >= 4) {  // RGBA format
//             float* pixelPtr = static_cast<float*>(pixelData);
            
//             for (int y = 0; y < height && y < imgHeight; ++y) {
//                 for (int x = 0; x < width && x < imgWidth; ++x) {
//                     int pixelIndex = y * (rowBytes / sizeof(float)) + x * 4;
                    
//                     // Get mask value at this position
//                     int maskIndex = y * width + x;
//                     if (maskIndex < static_cast<int>(mask.size())) {
//                         // Apply the mask as alpha channel
//                         pixelPtr[pixelIndex + 3] = mask[maskIndex];  // Alpha channel
//                     }
//                 }
//             }
//         }
        
//         // Release the image back to the host
//         // outputClip->releaseImage(outputImg);
//         delete outputImg;
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error applying mask to output: " << e.what() << std::endl;
//     }
// }

// };

// Plugin factory function
OFX::ImageEffect* createPlugin(OfxImageEffectHandle handle) {
    return new MagicMaskPlugin(handle);
}

// You need a factory class to describe and create your plugin
class MagicMaskPluginFactory : public OFX::PluginFactoryHelper<MagicMaskPluginFactory> {
public:
    MagicMaskPluginFactory() : OFX::PluginFactoryHelper<MagicMaskPluginFactory>(PLUGIN_ID, PLUGIN_VERSION_MAJOR, PLUGIN_VERSION_MINOR) {}
    
    virtual void describe(OFX::ImageEffectDescriptor &desc) override {
        desc.setPluginGrouping("YourCompany");
        desc.setPluginDescription("Magic Mask plugin for DaVinci Resolve");
    }
    
    virtual void describeInContext(OFX::ImageEffectDescriptor &desc, OFX::ContextEnum context) override {
        // Define your inputs/outputs here if needed
    }
    
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum context) override {
        return new MagicMaskPlugin(handle);
    }
};

// The wrapper looks for this specific function to load the plugin
void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray &ids) {
    static MagicMaskPluginFactory p;
    ids.push_back(&p);
}
};