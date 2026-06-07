# Time Tidy AI Engine

AI engine for image classification and tagging using Chinese-CLIP model.

## Project Structure

```
ai-engine/
├── src/                    # Core AI engine code
│   ├── engines/           # Classification engines
│   │   ├── search_engine.py         # Image/text search engine
│   │   ├── hierarchical_tag_engine.py  # Hierarchical classification
│   │   └── tag_engine_dynamic.py    # Dynamic tagging
│   ├── config/            # Configuration files
│   │   └ hierarchical_tags.json     # Hierarchical tag config
│   ├── models/            # ONNX models (downloaded)
│   ├── utils/             # Utility functions
│   ├── config.py          # Model configuration
│   └── main.py            # Main entry point
│
├── tests/                  # Test scripts with markdown reports
│   ├── test_hierarchical_tag.py     # Test hierarchical classification
│   ├── test_search_engine.py        # Test search engine
│   └ README.md            # Test documentation
│
├── scripts/                # Utility scripts and tools
│   ├── clip_model_manager.py        # Model download & export tool
│   ├── clip_onnx_examples.py        # ONNX usage examples
│   ├── unified_clip_onnx.py         # Unified CLIP interface
│   ├── CLIP_MODEL_MANAGER_README.md # Model management guide
│   ├── MODEL_CONFIG_GUIDE.md        # Model configuration guide
│   └ README.md            # Scripts documentation
│
├── test_results/           # Test output (markdown reports)
└── README.md               # This file
```

## Features

- **Hierarchical Classification**: Three-level classification (category → subcategory → attributes)
- **Dynamic Search**: Open vocabulary text-to-image search
- **Configurable Tags**: User-defined hierarchical tag configuration
- **Chinese-CLIP**: Optimized for Chinese text and images
- **ONNX Runtime**: Fast inference, low memory usage

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and Export Model

```bash
# Download and export model (first time)
python scripts/clip_model_manager.py --download --export --model ViT-B-16 --mirror

# Verify model
python scripts/clip_model_manager.py --verify --model ViT-B-16
```

### 3. Run Tests

```bash
# Test hierarchical classification (default: 5 images)
python tests/test_hierarchical_tag.py

# Test with more images
python tests/test_hierarchical_tag.py --num-images 10

# Test search engine
python tests/test_search_engine.py --num-images 5
```

### 4. View Test Results

Test results are saved in `test_results/` as markdown files with image previews:
- `hierarchical_tag_test_YYYYMMDD_HHMMSS.md`
- `search_engine_test_YYYYMMDD_HHMMSS.md`

## Core Components

### Engines

1. **SearchEngine** (`src/engines/search_engine.py`)
   - Image feature extraction
   - Text feature extraction
   - Similarity search

2. **HierarchicalTagEngine** (`src/engines/hierarchical_tag_engine.py`)
   - Three-level classification
   - Top-K candidate strategy
   - Multiple prompts per category

3. **TagEngineDynamic** (`src/engines/tag_engine_dynamic.py`)
   - Dynamic tagging with custom tags
   - Open vocabulary support

### Configuration

- **Model Config** (`src/config.py`): Select CLIP model (ViT-B-16, ViT-L-14, etc.)
- **Tag Config** (`src/config/hierarchical_tags.json`): Define hierarchical categories and tags

## Usage Examples

### Text-to-Image Search

```python
from engines.search_engine_faiss import SearchEngine

engine = SearchEngine()
result = engine.process({
    "cmd": "search",
    "query": "可爱的小猫",
    "top_k": 10
})
```

## Customization

### Modify Hierarchical Tags

Edit `src/config/hierarchical_tags.json` to customize categories:

```json
{
  "level1_categories": [
    {
      "id": "travel",
      "name": "旅游出行",
      "prompts": ["一张旅游的照片", "旅行时拍的照片"]
    }
  ],
  "level2_categories": {
    "travel": [
      {
        "id": "domestic_travel",
        "name": "国内游",
        "prompts": ["一张国内旅游的照片"]
      }
    ]
  }
}
```

### Switch Model

Edit `src/config.py`:

```python
CLIP_MODEL_NAME = "ViT-L-14"  # Higher accuracy
```

Then download the model:

```bash
python scripts/clip_model_manager.py --download --export --model ViT-L-14 --mirror
```

## Documentation

- **Scripts Documentation**: `scripts/README.md`
- **Model Management Guide**: `scripts/CLIP_MODEL_MANAGER_README.md`
- **Model Configuration Guide**: `scripts/MODEL_CONFIG_GUIDE.md`
- **Test Documentation**: `tests/README.md`

## Requirements

- Python >= 3.10
- ONNX Runtime
- Chinese-CLIP
- Pillow
- NumPy

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python tests/test_hierarchical_tag.py --num-images 10

# Format code
black src/
```

## License

MIT License