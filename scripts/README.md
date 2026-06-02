# AI Engine Scripts

Utility scripts and tools for model management and examples.

## Scripts

### 1. clip_model_manager.py

Download and export Chinese-CLIP models to ONNX format.

**Usage**:
```bash
# List supported models
python scripts/clip_model_manager.py --list

# Download and export model
python scripts/clip_model_manager.py --download --export --model ViT-B-16 --mirror

# Verify exported models
python scripts/clip_model_manager.py --verify --model ViT-B-16
```

**Documentation**: See `CLIP_MODEL_MANAGER_README.md`

### 2. clip_onnx_examples.py

Example code for using ONNX models.

**Usage**:
```bash
python scripts/clip_onnx_examples.py
```

### 3. unified_clip_onnx.py

Unified interface for Chinese-CLIP ONNX models.

**Features**:
- Separate model storage (memory efficient)
- Unified interface (easy to use)
- Flexible loading modes

**Usage**:
```python
from scripts.unified_clip_onnx import create_clip_model

# Vision only
model = create_clip_model('ViT-B-16', 'vision')

# Both encoders
model = create_clip_model('ViT-B-16', 'both')
```

## Documentation

### CLIP_MODEL_MANAGER_README.md

Complete guide for model management:
- Installation
- Download models
- Export to ONNX
- Verify models
- Troubleshooting

### MODEL_CONFIG_GUIDE.md

Guide for model configuration:
- Supported models
- Configuration methods
- Performance comparison
- Best practices

## Quick Start

```bash
# From project root
cd d:\work\56-time-tidy-prj\ai-engine

# Download and export model (first time)
python scripts/clip_model_manager.py --download --export --model ViT-B-16 --mirror

# Verify
python scripts/clip_model_manager.py --verify --model ViT-B-16

# Run examples
python scripts/clip_onnx_examples.py
```

## Notes

These scripts are utilities for:
- Model preparation (download, export)
- Examples and demonstrations
- Documentation and guides

They are not part of the core AI engine functionality.