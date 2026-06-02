# AI Engine - FAISS-based Search Implementation

## Overview

This implementation follows the design specification in `docs/plan-discuss.md` (L2659-4338).

## Key Differences from Old Implementation

### 1. Search Function

**Old Implementation** (`search_engine.py`):
- Action: `search_by_text`
- Parameters: `text`, `image_paths`, `top_k`
- Returns: `image_path`, `similarity`
- Performance: Slow (extracts features for all images each time)
- No vector database

**New Implementation** (`search_engine_faiss.py`):
- Action: `search`
- Parameters: `query`, `top_k`, `filters`
- Returns: `record_id`, `similarity`, `rank`
- Performance: ~2-5ms (uses FAISS index)
- Uses FAISS IndexIDMap2(IndexIVFFlat)

### 2. Architecture

**Old Implementation**:
- Single engine: SearchEngine
- Direct feature extraction
- No index management
- No data synchronization

**New Implementation**:
- Multiple engines:
  - `SearchEngine`: Text-to-image search
  - `IndexInitEngine`: Index initialization and building
  - `SyncEngine`: Data synchronization (IMAGE_ADDED, IMAGE_DELETED, etc.)
  - `TagEngine`: Tag detection
  - `PingEngine`: Health check

### 3. Data Model

**Old Implementation**:
- Uses `image_path` as identifier
- No concept of `record_id`

**New Implementation**:
- Uses `record_id` as identifier (matches SQLite database)
- Supports `image_hash` for deduplication
- Metadata file: `ai_metadata.json`

## Module Structure

```
ai-engine/src/
├── main_faiss.py                 # Main entry point (stdio communication)
├── engines/
│   ├── clip_encoder.py           # CLIP feature extraction
│   ├── faiss_index_manager.py    # FAISS index management
│   ├── search_engine_faiss.py    # Text-to-image search
│   ├── index_init_engine.py      # Index initialization
│   ├── sync_engine.py            # Data synchronization
│   └── tag_engine_dynamic.py     # Tag detection
├── data/
│   └── index/
│       ├── faiss_index.bin       # FAISS index file
│       ├── ai_metadata.json      # Metadata file
│       └── index_config.json     # Index configuration
└── models/                       # CLIP ONNX models
```

## Usage

### 1. Index Initialization

```json
{
  "id": "1234567890",
  "action": "index_init",
  "params": {
    "action": "init",
    "rebuild": false,
    "image_records": [
      {
        "record_id": 1001,
        "image_path": "/path/to/image1.jpg",
        "image_hash": "abc123"
      },
      {
        "record_id": 1002,
        "image_path": "/path/to/image2.jpg",
        "image_hash": "def456"
      }
    ]
  }
}
```

**Response**:
```json
{
  "id": "1234567890",
  "success": true,
  "data": {
    "success": true,
    "action": "build_new",
    "total_images": 2,
    "failed_records": 0,
    "build_time_seconds": 5.2,
    "index_file": "data/index/faiss_index.bin",
    "metadata_file": "data/index/ai_metadata.json"
  }
}
```

### 2. Text-to-Image Search

```json
{
  "id": "1234567891",
  "action": "search",
  "params": {
    "action": "search",
    "query": "风景",
    "top_k": 10,
    "filters": {
      "min_similarity": 0.5
    }
  }
}
```

**Response**:
```json
{
  "id": "1234567891",
  "success": true,
  "data": {
    "query": "风景",
    "results": [
      {
        "record_id": 1001,
        "similarity": 0.95,
        "rank": 1
      },
      {
        "record_id": 1002,
        "similarity": 0.89,
        "rank": 2
      }
    ],
    "query_time_ms": 3.5,
    "total_count": 2,
    "filters_applied": true
  }
}
```

### 3. Data Synchronization

#### IMAGE_ADDED Event

```json
{
  "id": "1234567892",
  "action": "sync",
  "params": {
    "event_type": "IMAGE_ADDED",
    "record_id": 1003,
    "image_path": "/path/to/new_image.jpg",
    "image_hash": "ghi789"
  }
}
```

**Response**:
```json
{
  "id": "1234567892",
  "success": true,
  "data": {
    "success": true,
    "event_type": "IMAGE_ADDED",
    "record_id": 1003,
    "vector_id": 1003,
    "process_time_ms": 150.5,
    "message": "Image added to index successfully"
  }
}
```

#### IMAGE_DELETED Event

```json
{
  "id": "1234567893",
  "action": "sync",
  "params": {
    "event_type": "IMAGE_DELETED",
    "record_id": 1001
  }
}
```

#### BATCH_IMPORTED Event

```json
{
  "id": "1234567894",
  "action": "sync",
  "params": {
    "event_type": "BATCH_IMPORTED",
    "records": [
      {
        "record_id": 1004,
        "image_path": "/path/to/image4.jpg",
        "image_hash": "jkl012"
      },
      {
        "record_id": 1005,
        "image_path": "/path/to/image5.jpg",
        "image_hash": "mno345"
      }
    ]
  }
}
```

## Testing

### Test Tool

```bash
cd ai-engine/tests
python test_faiss_search.py --test-dir <path> --max-images <num> --search-query "风景" --top-k 5
```

### Test Results

Test results are saved to `ai-engine/test_results/faiss_search_test_<timestamp>.md`.

The report includes:
- Index initialization results
- Search performance (query_time_ms)
- Search results (record_id, similarity)
- Tag detection results
- Sync event results

## Performance

### Index Initialization
- Build time: ~5-10s for 1000 images
- Depends on CLIP model size and CPU performance

### Search Performance
- Query time: ~2-5ms (10,000-100,000 images)
- Text feature caching improves repeated queries
- FAISS IndexIVFFlat provides fast approximate search

### Data Synchronization
- IMAGE_ADDED: ~150ms (single image)
- IMAGE_DELETED: ~5ms
- BATCH_IMPORTED: ~150ms per image

## Dependencies

Required packages:
- `faiss-cpu` or `faiss-gpu` (for vector search)
- `onnxruntime` (for CLIP model inference)
- `cn-clip` (for Chinese text tokenization)
- `numpy`, `pillow` (for image processing)

Install dependencies:
```bash
pip install faiss-cpu onnxruntime cn-clip numpy pillow
```

## File Locations

- **Index file**: `ai-engine/data/index/faiss_index.bin`
- **Metadata file**: `ai-engine/data/index/ai_metadata.json`
- **Config file**: `ai-engine/data/index/index_config.json`
- **CLIP models**: `ai-engine/src/models/`

## Design Compliance

This implementation follows the design specification in `docs/plan-discuss.md`:

✓ Uses FAISS IndexIDMap2(IndexIVFFlat)
✓ Returns record_id instead of image_path
✓ Supports filters (min_similarity)
✓ Query time ~2-5ms for 10,000-100,000 images
✓ Event-driven data synchronization
✓ stdio communication protocol
✓ JSON request-response format
✓ Text feature caching for performance optimization