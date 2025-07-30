# Tese - UAV Object Detection Research Project

This repository contains the complete research project focused on UAV (Unmanned Aerial Vehicle) object detection using YOLO-based deep learning models.

## 📁 Project Structure

```
├── Code/
│   └── AllYolosPython/          # Main Python implementation
│       ├── src/                 # Source code
│       ├── models/              # YOLO model files
│       ├── config/              # Configuration files
│       ├── scripts/             # Utility scripts
│       ├── notebooks/           # Jupyter notebooks
│       └── docs/                # Technical documentation
├── Development/                 # Development documentation and diagrams
├── Reports/                     # Academic reports and documentation
└── Research/                    # Research papers and references
```

## 🎯 Project Overview

This research project focuses on developing and optimizing object detection algorithms for UAV applications using various YOLO (You Only Look Once) model variants. The project explores lightweight models suitable for onboard processing on unmanned aerial vehicles.

### Key Components

#### Code/AllYolosPython
The main implementation containing:
- **Multi-YOLO Support**: YOLOv5, YOLOv8, YOLOv10 implementations
- **Real-time Processing**: Optimized for real-time object detection
- **Camera Integration**: Support for various camera systems including RealSense
- **Performance Benchmarking**: Tools for measuring FPS and accuracy
- **Model Export**: ONNX model export capabilities

#### Research Collection
Comprehensive collection of research papers covering:
- Lightweight UAV object detection algorithms
- YOLO model benchmarking on embedded systems
- Real-time wildlife tracking applications
- LiDAR-based flying object detection
- Small object detection optimization

#### Development Documentation
- System architecture diagrams
- Setup instructions and configurations
- Performance analysis reports

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Camera hardware (optional, for live detection)

### Quick Start
1. Navigate to the main implementation:
   ```bash
   cd Code/AllYolosPython
   ```

2. Follow the installation guide:
   ```bash
   # See Code/AllYolosPython/docs/installation_guide.md for detailed instructions
   pip install -r requirements.txt
   ```

3. Run the main application:
   ```bash
   python src/main.py
   ```

## 📊 Research Focus Areas

1. **Model Optimization**: Comparing performance of different YOLO variants
2. **Edge Computing**: Optimizing models for embedded systems
3. **Real-time Processing**: Achieving high FPS for UAV applications
4. **Small Object Detection**: Enhancing detection of distant or small targets
5. **Hardware Integration**: Camera and sensor integration for UAV platforms

## 📝 Documentation

- **Technical Documentation**: See `Code/AllYolosPython/docs/`
- **Research Papers**: See `Research/` directory
- **Development Reports**: See `Reports/` directory
- **Setup Instructions**: See `Development/` directory

## 🔬 Research Papers Included

The `Research/` directory contains relevant academic papers covering:
- Lightweight YOLO implementations for UAVs
- Benchmarking studies on embedded systems
- Real-time object detection optimization
- Edge computing for aerial platforms

## 📈 Performance Metrics

The project includes comprehensive benchmarking tools to measure:
- **FPS (Frames Per Second)**: Real-time processing capability
- **Accuracy**: Detection precision and recall
- **Memory Usage**: Resource consumption analysis
- **Latency**: End-to-end processing time

## 🛠️ Model Support

Currently supported YOLO variants:
- **YOLOv5** (n, s variants)
- **YOLOv8** (n, s, m variants)
- **YOLOv10** (n, s variants)

All models support both PyTorch (.pt) and ONNX (.onnx) formats for optimized inference.

## 📋 Requirements

For detailed requirements and dependencies, see:
- `Code/AllYolosPython/requirements.txt` - Main dependencies
- `Code/AllYolosPython/requirements/` - Specialized requirement files

## 🤝 Contributing

This is a research project. For questions or collaboration opportunities, please refer to the documentation in the respective directories.

## 📄 License

See `Code/AllYolosPython/LICENSE` for license information.

## 🔗 Related Work

This project builds upon extensive research in:
- YOLO object detection algorithms
- UAV-based computer vision
- Edge computing optimization
- Real-time embedded systems

---

*This repository represents ongoing research in UAV object detection and optimization for aerial platforms.*
