#!/usr/bin/env python3
"""
Download various YOLO models for performance comparison.
This script downloads different model sizes, formats, and optimizations.
"""

import os
import sys
import urllib.request
from pathlib import Path
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model directory
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model configurations
MODELS_CONFIG = {
    # Standard YOLOv8 models (different sizes)
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt", 
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
    
    # YOLOv5 for comparison
    "yolov5n.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
    "yolov5s.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    
    # YOLOv10 (newer architecture)
    "yolov10n.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
    "yolov10s.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
}

def download_file(url: str, filepath: Path) -> bool:
    """Download a file from URL to filepath."""
    try:
        logger.info(f"Downloading {filepath.name}...")
        urllib.request.urlretrieve(url, filepath)
        logger.info(f"✓ Downloaded {filepath.name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {filepath.name}: {e}")
        return False

def export_onnx_models():
    """Export PyTorch models to ONNX format for comparison."""
    pt_models = list(MODELS_DIR.glob("*.pt"))
    
    for pt_model in pt_models:
        try:
            logger.info(f"Exporting {pt_model.name} to ONNX...")
            model = YOLO(pt_model)
            onnx_path = pt_model.with_suffix('.onnx')
            
            # Export to ONNX with optimization for inference
            model.export(
                format='onnx',
                dynamic=False,
                simplify=True,
                opset=11,
                imgsz=640
            )
            logger.info(f"✓ Exported {onnx_path.name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to export {pt_model.name} to ONNX: {e}")

def create_quantized_models():
    """Create INT8 quantized versions of models."""
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        
        # For now, we'll create placeholder files and document the process
        # Actual quantization requires specific datasets and tools
        quantized_info = MODELS_DIR / "quantized_models_info.txt"
        
        with open(quantized_info, 'w') as f:
            f.write("""Quantized Model Information
============================

To create quantized models for better performance on Raspberry Pi:

1. INT8 Quantization using ONNX Runtime:
   - Requires calibration dataset
   - Can improve inference speed by 2-4x
   - Slight accuracy trade-off

2. TensorRT Optimization (if using Jetson):
   - model.export(format='engine', half=True)
   - Optimized for NVIDIA hardware

3. OpenVINO Format (Intel hardware):
   - model.export(format='openvino', half=True)
   - Optimized for Intel CPUs/GPUs

4. TensorFlow Lite:
   - model.export(format='tflite', int8=True)
   - Good for mobile/edge devices

Commands to create quantized models:
-----------------------------------
# TensorFlow Lite INT8
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='tflite', int8=True)"

# ONNX with dynamic quantization  
python -c "import onnx; from onnxruntime.quantization import quantize_dynamic; quantize_dynamic('yolov8n.onnx', 'yolov8n_quantized.onnx')"

# OpenVINO (requires Intel toolkit)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino', half=True)"
""")
        
        logger.info("✓ Created quantized models information file")
        
    except Exception as e:
        logger.error(f"✗ Failed to create quantized model info: {e}")

def create_model_comparison_info():
    """Create a file with model comparison information."""
    info_file = MODELS_DIR / "models_info.md"
    
    content = """# Model Comparison Information

## Available Models

### YOLOv8 Series (Ultralytics)
- **yolov8n.pt**: Nano model (~6MB, fastest)
- **yolov8s.pt**: Small model (~22MB, balanced)
- **yolov8m.pt**: Medium model (~52MB, higher accuracy)

### YOLOv5 Series (For comparison)
- **yolov5n.pt**: Nano model (~4MB)
- **yolov5s.pt**: Small model (~14MB)

### YOLOv10 Series (Latest architecture)
- **yolov10n.pt**: Nano model (~5MB, improved efficiency)
- **yolov10s.pt**: Small model (~16MB)

### ONNX Models
- ***.onnx**: Optimized for cross-platform inference
- Generally faster inference than PyTorch models
- Smaller memory footprint

### Performance Expectations (Raspberry Pi 5)

| Model | Size | FPS (Est.) | mAP@0.5 | Use Case |
|-------|------|------------|---------|----------|
| yolov8n | 6MB | 8-12 | 37.3 | Real-time detection |
| yolov8s | 22MB | 4-8 | 44.9 | Balanced performance |
| yolov8m | 52MB | 2-5 | 50.2 | High accuracy |
| yolov5n | 4MB | 10-15 | 28.0 | Ultra-fast detection |
| yolov10n | 5MB | 10-14 | 38.5 | Latest architecture |

### Optimization Strategies
1. **Model Pruning**: Remove unnecessary weights
2. **Quantization**: INT8 instead of FP32
3. **ONNX Runtime**: Optimized inference engine
4. **TensorFlow Lite**: Mobile-optimized format
5. **OpenVINO**: Intel CPU optimization

### Usage Notes
- Start with nano models for real-time applications
- Use larger models for accuracy-critical applications
- ONNX models typically provide 10-30% speed improvement
- Quantized models can be 2-4x faster with minimal accuracy loss
"""
    
    with open(info_file, 'w') as f:
        f.write(content)
    
    logger.info("✓ Created model comparison information")

def main():
    """Main function to download and prepare models."""
    logger.info("Starting model download and preparation...")
    logger.info(f"Models will be saved to: {MODELS_DIR}")
    
    # Download standard models
    successful_downloads = 0
    total_models = len(MODELS_CONFIG)
    
    for model_name, url in MODELS_CONFIG.items():
        model_path = MODELS_DIR / model_name
        
        if model_path.exists():
            logger.info(f"✓ {model_name} already exists, skipping...")
            successful_downloads += 1
            continue
            
        if download_file(url, model_path):
            successful_downloads += 1
    
    logger.info(f"Downloaded {successful_downloads}/{total_models} models successfully")
    
    # Export ONNX models
    logger.info("\nExporting PyTorch models to ONNX format...")
    export_onnx_models()
    
    # Create quantized model information
    logger.info("\nCreating quantized model information...")
    create_quantized_models()
    
    # Create model comparison info
    logger.info("\nCreating model comparison documentation...")
    create_model_comparison_info()
    
    # List all downloaded models
    logger.info("\nFinal model inventory:")
    for model_file in sorted(MODELS_DIR.glob("*")):
        if model_file.is_file():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            logger.info(f"  {model_file.name}: {size_mb:.1f} MB")
    
    logger.info("\n✓ Model download and preparation complete!")
    logger.info("Run 'python scripts/benchmark.sh' to compare model performance")

if __name__ == "__main__":
    main()
