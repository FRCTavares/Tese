# Model Comparison Information

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
