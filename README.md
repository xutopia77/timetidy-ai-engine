# Time Tidy AI Engine

AI engine for image tagging using CLIP model.

## Features

- CLIP-based image tagging
- ONNX Runtime inference
- Chinese label support

## Requirements

- Python >= 3.10
- ONNX Runtime
- Pillow
- NumPy

## Usage

```bash
# Run AI engine
python src/main.py

# Or with specific Python
/path/to/python src/main.py
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
```

## Models

Models are stored in `src/models/`:

- `clip_vision.onnx` - CLIP vision model
- `tag_embeddings.npz` - Pre-computed text embeddings
- `tag_vocab.json` - Tag vocabulary
