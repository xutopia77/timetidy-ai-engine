# AI 引擎测试

AI 引擎功能测试脚本，生成 Markdown 格式的测试报告。

## 核心引擎说明

### SearchEngine（搜索引擎）

`src/engines/search_engine.py` 提供基于 Chinese-CLIP ONNX 模型的搜索功能：

**主要功能**：
1. **图像特征提取** (`encode_image`)
   - 输入：图像路径
   - 输出：归一化的特征向量（用于相似度计算）

2. **文本特征提取** (`encode_text`)
   - 输入：中文文本
   - 输出：归一化的特征向量

3. **文本到图像搜索** (`search_by_text`)
   - 输入：查询文本 + 图像列表
   - 输出：按相似度排序的图像结果

4. **图像相似度比较** (`compare_images`)
   - 输入：两张图像路径
   - 输出：相似度分数（0-1）

**技术特点**：
- 使用 ONNX Runtime 进行高效推理
- 支持 CPU 执行
- 自动下载模型（可配置）
- 支持多种 CLIP 模型（ViT-B-16、ViT-L-14、ViT-H-14、RN50）

## 测试脚本

### 1. test_hierarchical_tag.py

测试分层标签分类引擎，使用真实图像进行分类测试。

**使用方法**：
```bash
# 默认：处理 5 张图像
python tests/test_hierarchical_tag.py

# 处理 10 张图像
python tests/test_hierarchical_tag.py --num-images 10

# 指定测试目录
python tests/test_hierarchical_tag.py --test-dir path/to/images --num-images 20
```

**输出结果**：
- Markdown 报告：`test_results/hierarchical_tag_test_YYYYMMDD_HHMMSS.md`
- 包含：分类统计数据、详细结果及图像预览

### 2. test_search_engine.py

测试搜索引擎的文搜图功能，使用 JSON 配置文件定义搜索查询。

**配置文件**：`tests/search_queries.json`

用户可以通过修改配置文件自定义搜索文本：
```json
{
  "queries": [
    {
      "text": "风景",
      "description": "搜索风景相关的图片",
      "top_k": 5
    },
    {
      "text": "人物",
      "description": "搜索人物相关的图片",
      "top_k": 5
    }
  ],
  "settings": {
    "max_images": 100
  }
}
```

**配置说明**：
- `queries`: 搜索查询列表
  - `text`: 搜索文本（中文）
  - `description`: 查询描述（可选）
  - `top_k`: 返回结果数量（默认 5）
- `settings.max_images`: 最大搜索图像数量

**使用方法**：
```bash
# 使用默认配置文件 (tests/search_queries.json)
python tests/test_search_engine.py

# 指定自定义配置文件
python tests/test_search_engine.py --config path/to/custom_queries.json

# 指定测试图像目录
python tests/test_search_engine.py --test-dir path/to/images

# 覆盖最大图像数量
python tests/test_search_engine.py --max-images 50
```

**输出结果**：
- Markdown 报告：`test_results/search_engine_test_YYYYMMDD_HHMMSS.md`
- 包含：
  - 搜索配置概览
  - 每个查询的搜索结果（排名、图像、相似度）
  - 图像预览
  - 详细结果展示

## 测试结果

所有测试结果保存在 `test_results/` 目录：
- Markdown 格式报告，包含图像预览
- 使用正斜杠路径，确保跨平台兼容性

## 环境要求

- Python 3.8+
- ONNX 模型已导出（先运行 `scripts/clip_model_manager.py`）
- 测试图像位于 `test-data/data-min-real/` 或自定义目录

## 运行测试

从项目根目录运行：
```bash
cd d:\work\56-time-tidy-prj\ai-engine

# 测试分层标签分类
python tests/test_hierarchical_tag.py --num-images 5

# 测试文搜图功能（使用默认配置）
python tests/test_search_engine.py
```

或从测试目录运行：
```bash
cd d:\work\56-time-tidy-prj\ai-engine\tests

# 测试分层标签分类
python test_hierarchical_tag.py --num-images 5

# 测试文搜图功能
python test_search_engine.py
```

## 执行流程分析

### test_search_engine.py 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 加载配置文件 (search_queries.json)                           │
│     - 读取搜索查询列表                                            │
│     - 读取设置参数 (max_images)                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. 查找测试图像文件                                              │
│     - 遍历测试目录                                                │
│     - 收集所有图像文件路径                                        │
│     - 限制最大图像数量                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. 创建 SearchEngine 实例                                       │
│     - 初始化模型路径（不加载模型）                                 │
│     - 懒加载模式：首次使用时才加载                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. 执行搜索查询（循环处理每个查询）                               │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  4.1 文本特征提取 (encode_text)                          │ │
│     │      - cn_clip 分词 → token序列                          │ │
│     │      - ONNX Text Encoder 推理                            │ │
│     │      - 特征归一化                                        │ │
│     └─────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  4.2 图像特征提取 (encode_image) - 循环N次               │ │
│     │      - 图像预处理 (resize, normalize)                    │ │
│     │      - ONNX Vision Encoder 推理                          │ │
│     │      - 特征归一化                                        │ │
│     └─────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  4.3 相似度计算 (compute_similarity) - 循环N次           │ │
│     │      - 文本特征 · 图像特征                                │ │
│     │      - 得到相似度分数                                     │ │
│     └─────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  4.4 排序并返回 Top-K 结果                                │ │
│     └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. 生成 Markdown 测试报告                                        │
│     - 搜索配置概览                                                │
│     - 搜索结果表格                                                │
│     - 图像预览                                                    │
│     - 详细结果展示                                                │
└─────────────────────────────────────────────────────────────────┘
```

### SearchEngine 核心执行流程

#### 1. 模型初始化（懒加载）

```python
# __init__: 仅初始化路径，不加载模型
SearchEngine.__init__()
  → 设置模型路径 (vision_model_path, text_model_path)
  → _vision_session = None  # 未加载
  → _text_session = None    # 未加载

# _ensure_model: 首次调用时加载模型
_ensure_model()
  → 检查模型文件是否存在
  → 创建 ONNX InferenceSession
  → 加载 Vision Encoder (clip_vision_*.onnx)
  → 加载 Text Encoder (clip_text_*.onnx)
```

**特点**：懒加载设计，避免不必要的模型加载开销。

#### 2. 文本特征提取流程

```python
encode_text(text)
  → _tokenize_text(text)           # cn_clip 分词
    → clip.tokenize([text])        # 得到 token 序列
    → padding/truncate to 77 tokens
  → ONNX Text Encoder 推理
    → _text_session.run(None, {input_name: tokens})
  → 特征归一化
    → features / np.linalg.norm(features)
```

**耗时分析**：
- 分词：~1-5ms
- ONNX 推理：~10-50ms（取决于模型大小）
- 归一化：~1ms

#### 3. 图像特征提取流程

```python
encode_image(image_path)
  → _preprocess_image(image_path)  # 图像预处理
    → Image.open().convert("RGB")
    → resize to 224x224 (BICUBIC)
    → normalize (mean, std)
    → transpose (HWC → CHW)
  → ONNX Vision Encoder 推理
    → _vision_session.run(None, {input_name: input_data})
  → 特征归一化
    → features / np.linalg.norm(features)
```

**耗时分析**：
- 图像加载+预处理：~5-20ms
- ONNX 推理：~20-100ms（取决于模型大小）
- 归一化：~1ms

#### 4. 文搜图流程 (search_by_text)

```python
search_by_text(text, image_paths, top_k)
  → encode_text(text)              # 1次，提取文本特征
  → for img_path in image_paths:   # N次循环
      → encode_image(img_path)     # 图像特征提取（主要耗时）
      → compute_similarity()       # 相似度计算
  → sort by similarity             # 排序
  → return top_k results           # 返回结果
```

**性能瓶颈**：
- 如果有 N 张图像，需要执行 N 次 `encode_image`
- 多个查询时，图像编码会**重复执行**
- 例如：5 个查询 × 100 张图像 = 500 次 encode_image

## 性能优化建议

### 优化方案对比

| 方案 | 实现难度 | 性能提升 | 适用场景 |
|------|----------|----------|----------|
| 特征缓存（内存） | 低 | 高 | 多次查询同一图像集 |
| 特征缓存（文件） | 中 | 高 | 跨会话复用特征 |
| 批量编码 | 中 | 中 | 大规模图像处理 |
| 并行处理 | 中 | 中 | CPU 多核环境 |
| GPU 加速 | 高 | 很高 | 有 GPU 环境 |

### 1. 特征缓存（推荐）

**问题**：当前实现中，每个查询都要重新编码所有图像。

**解决方案**：预先编码所有图像，缓存特征向量。

```python
# 优化后的流程
def search_with_cache(text, image_paths, cached_features, top_k):
    # 1. 文本特征提取（1次）
    text_features = encode_text(text)
    
    # 2. 使用缓存的特征，无需重新编码
    results = []
    for img_path, img_features in cached_features.items():
        similarity = compute_similarity(text_features, img_features)
        results.append({"image_path": img_path, "similarity": similarity})
    
    # 3. 排序返回
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# 预先编码所有图像（只执行一次）
cached_features = {}
for img_path in image_paths:
    cached_features[img_path] = encode_image(img_path)

# 多次查询使用缓存
for query in queries:
    results = search_with_cache(query["text"], image_paths, cached_features, top_k)
```

**性能提升**：
- 原方案：M 个查询 × N 张图像 = M×N 次 encode_image
- 优化后：N 次 encode_image（预先编码）+ M 次 encode_text
- 提升：~M 倍（查询数量越多，提升越大）

### 2. 特征持久化缓存

**解决方案**：将图像特征保存到文件，跨会话复用。

```python
import pickle
import hashlib

def get_image_hash(image_path):
    """Calculate image file hash for cache key"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_cached_features(cache_file):
    """Load cached features from file"""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cached_features(cache_file, features):
    """Save cached features to file"""
    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)

def encode_image_with_cache(image_path, cache_file):
    """Encode image with cache"""
    cached = load_cached_features(cache_file)
    image_hash = get_image_hash(image_path)
    
    if image_hash in cached:
        return cached[image_hash]  # 使用缓存
    
    features = encode_image(image_path)
    cached[image_hash] = features
    save_cached_features(cache_file, cached)
    return features
```

**适用场景**：
- 图像集固定，多次测试
- 避免重复编码开销
- 需要管理缓存更新策略

### 3. 批量编码

**解决方案**：使用批量推理减少 ONNX 调用次数。

```python
def encode_images_batch(image_paths, batch_size=32):
    """Batch encode images for better performance"""
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # 批量预处理
        batch_inputs = []
        for img_path in batch_paths:
            batch_inputs.append(_preprocess_image(img_path))
        
        # 批量推理（需要模型支持 batch dimension）
        batch_features = _vision_session.run(
            None, 
            {input_name: np.concatenate(batch_inputs)}
        )[0]
        
        # 批量归一化
        batch_features = batch_features / np.linalg.norm(
            batch_features, axis=-1, keepdims=True
        )
        
        all_features.extend(batch_features)
    
    return all_features
```

**注意事项**：
- 需要模型支持 batch dimension
- 内存占用增加（batch_size × feature_dim）
- 需要调整 batch_size 平衡性能和内存

### 4. 并行处理

**解决方案**：使用多线程/多进程并行编码图像。

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def encode_images_parallel(image_paths, max_workers=4):
    """Parallel encode images using multiple threads"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        features = list(executor.map(encode_image, image_paths))
    return dict(zip(image_paths, features))
```

**注意事项**：
- ONNX Runtime 在多线程环境下需要正确配置
- ThreadPoolExecutor：适合 I/O 密集型（图像加载）
- ProcessPoolExecutor：适合 CPU 密集型（ONNX 推理）

### 5. GPU 加速

**解决方案**：使用 CUDAExecutionProvider 加速推理。

```python
# 在 _ensure_model 中修改
self._vision_session = ort.InferenceSession(
    self._vision_model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

self._text_session = ort.InferenceSession(
    self._text_model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

**性能提升**：
- GPU 推理速度比 CPU 快 5-10 倍
- 需要安装 CUDA 和 cuDNN
- 需要安装 onnxruntime-gpu

## 推荐优化策略

### 测试场景优化（当前需求）

针对 `test_search_engine.py` 的测试场景，推荐使用**特征缓存（内存）**方案：

```python
# test_search_engine.py 优化示例
def main():
    # ... 加载配置和图像列表 ...
    
    engine = SearchEngine()
    
    # 优化：预先编码所有图像（只执行一次）
    print("预先编码所有图像...")
    cached_features = {}
    for idx, img_path in enumerate(image_files, 1):
        print(f"  [{idx}/{len(image_files)}] 编码: {Path(img_path).name}")
        cached_features[img_path] = engine.encode_image(img_path)
    
    # 使用缓存执行搜索
    all_results = {}
    for query in queries:
        query_text = query["text"]
        top_k = query.get("top_k", 5)
        
        print(f"\n查询: \"{query_text}\"")
        
        # 文本特征提取
        text_features = engine.encode_text(query_text)
        
        # 使用缓存计算相似度
        results = []
        for img_path, img_features in cached_features.items():
            similarity = engine.compute_similarity(text_features, img_features)
            results.append({"image_path": img_path, "similarity": round(similarity, 4)})
        
        # 排序返回 top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        all_results[query_text] = results[:top_k]
        
        print(f"  找到 {len(results[:top_k])} 个结果")
    
    # ... 生成报告 ...
```

**性能对比**：
- 原方案：5 查询 × 100 图像 = 500 次 encode_image (~25-50 秒)
- 优化后：100 次 encode_image + 5 次 encode_text (~5-10 秒)
- **提升约 5 倍**

### 生产环境优化建议

对于生产环境，推荐组合使用多种优化方案：

1. **特征持久化缓存**：避免重复编码
2. **批量编码**：大规模图像处理
3. **GPU 加速**：最大化推理速度
4. **索引结构**：使用向量数据库（如 FAISS、Milvus）加速搜索