# create download_models.sh
#!/bin/bash
echo "Downloading required models..."

# Create models directory
mkdir -p Resources/models/

# Download SAM2 model (if available)
if curl -f -s "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.onnx" -o Resources/models/sam2.onnx; then
    echo "SAM2 model downloaded successfully"
else
    echo "Warning: Could not download SAM2 ONNX, using MobileSAM fallback"
    curl -f -s "https://github.com/ChaoningZhang/Mobile-SAM/raw/main/weights/mobile_sam.onnx" -o Resources/models/sam2.onnx
fi

# Download ViTMatte model  
curl -f -s "https://github.com/hkchengrex/ViTMatte/releases/download/v1.0/vitmatte.onnx" -o Resources/models/vitmatte.onnx

echo "Models downloaded to Resources/models/"
