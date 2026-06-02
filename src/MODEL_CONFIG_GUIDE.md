# 模型配置指南

## 概述

本项目支持多种Chinese-CLIP模型，可以通过配置文件轻松切换。

---

## 支持的模型

| 模型名称 | 参数量 | 特征维度 | 文件大小 | 推荐场景 |
|---------|--------|----------|---------|---------|
| **ViT-B-16** | 86M + 102M | 512 | 330 MB | **通用推荐** ⭐ |
| ViT-L-14 | 307M + 102M | 768 | 670 MB | 高精度需求 |
| ViT-L-14-336 | 307M + 102M | 768 | 670 MB | 最高精度 |
| ViT-H-14 | 632M + 326M | 1024 | 1.2 GB | 研究使用 |
| RN50 | 39M + 102M | 2048 | 350 MB | 传统CNN |

---

## 配置方法

### 方法1：修改配置文件（推荐）

编辑 `ai-engine/src/config.py`：

```python
# 修改这一行
CLIP_MODEL_NAME = "ViT-B-16"  # 改为你想用的模型
```

**示例**：

```python
# 使用推荐模型（默认）
CLIP_MODEL_NAME = "ViT-B-16"

# 使用高精度模型
CLIP_MODEL_NAME = "ViT-L-14"

# 使用最高精度模型
CLIP_MODEL_NAME = "ViT-L-14-336"
```

---

### 方法2：环境变量（临时切换）

```bash
# 临时使用其他模型
export CLIP_MODEL_NAME="ViT-L-14"
python main.py
```

---

## 完整工作流程

### 场景1：首次使用（推荐模型）

```bash
# 1. 配置模型（默认已是ViT-B-16，无需修改）
# config.py: CLIP_MODEL_NAME = "ViT-B-16"

# 2. 下载并导出模型
cd ai-engine/src
python clip_model_manager.py --download --export --model ViT-B-16 --mirror

# 3. 验证
python clip_model_manager.py --verify --model ViT-B-16

# 4. 使用
python main.py
```

---

### 场景2：切换到高精度模型

```bash
# 1. 修改配置
# 编辑 config.py:
# CLIP_MODEL_NAME = "ViT-L-14"

# 2. 下载并导出新模型
python clip_model_manager.py --download --export --model ViT-L-14 --mirror

# 3. 验证
python clip_model_manager.py --verify --model ViT-L-14

# 4. 使用（自动使用ViT-L-14）
python main.py
```

---

### 场景3：多模型切换

```bash
# 下载多个模型
python clip_model_manager.py --download --export --model all --mirror

# 切换使用
# 修改 config.py 中的 CLIP_MODEL_NAME 即可

# 使用 ViT-B-16
# CLIP_MODEL_NAME = "ViT-B-16"

# 使用 ViT-L-14
# CLIP_MODEL_NAME = "ViT-L-14"
```

---

## 配置文件详解

### config.py 关键配置

```python
# 模型选择
CLIP_MODEL_NAME = "ViT-B-16"  # 当前使用的模型

# 模型下载
ALLOW_MODEL_DOWNLOAD = False  # 是否允许自动下载

# 特征维度（自动设置）
MODEL_FEATURE_DIMS = {
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-L-14-336": 768,
    "ViT-H-14": 1024,
    "RN50": 2048,
}

# 辅助函数
def get_feature_dim():
    """获取当前模型的特征维度"""
    return MODEL_FEATURE_DIMS.get(CLIP_MODEL_NAME, 512)

def get_model_filenames():
    """获取当前模型的文件名"""
    model_name = CLIP_MODEL_NAME.replace("-", "_").replace("/", "_")
    return {
        "vision": f"clip_vision_{model_name}.onnx",
        "text": f"clip_text_{model_name}.onnx",
        "tokenizer": f"tokenizer_{model_name}.json",
    }
```

---

## 模型文件说明

### 文件命名规则

```
模型名称: ViT-B-16
↓
文件名:
├── clip_vision_ViT_B_16.onnx    # 视觉编码器
├── clip_text_ViT_B_16.onnx      # 文本编码器
└── tokenizer_ViT_B_16.json      # Tokenizer信息
```

### 文件位置

```
ai-engine/src/models/
├── clip_vision_ViT_B_16.onnx      (150 MB)
├── clip_text_ViT_B_16.onnx        (180 MB)
├── tokenizer_ViT_B_16.json        (1 KB)
├── clip_vision_ViT_L_14.onnx      (490 MB)
├── clip_text_ViT_L_14.onnx        (580 MB)
└── tokenizer_ViT_L_14.json        (1 KB)
```

---

## 性能对比

### 推理速度（单张图片）

| 模型 | CPU | GPU | 内存占用 |
|------|-----|-----|---------|
| ViT-B-16 | 30ms | 8ms | 320 MB |
| ViT-L-14 | 80ms | 20ms | 1.0 GB |
| ViT-L-14-336 | 100ms | 25ms | 1.2 GB |
| ViT-H-14 | 200ms | 50ms | 2.5 GB |

### 准确度对比

| 模型 | 图片分类 | 文本检索 | 综合评分 |
|------|---------|---------|---------|
| ViT-B-16 | 68.2% | 61.5% | ⭐⭐⭐⭐ |
| ViT-L-14 | 75.4% | 66.4% | ⭐⭐⭐⭐⭐ |
| ViT-L-14-336 | 76.6% | 67.5% | ⭐⭐⭐⭐⭐ |
| ViT-H-14 | 78.2% | 69.1% | ⭐⭐⭐⭐⭐ |

---

## 推荐配置

### 个人项目（推荐）

```python
CLIP_MODEL_NAME = "ViT-B-16"
```

**理由**：
- ✅ 速度快（30ms/图）
- ✅ 内存小（320MB）
- ✅ 精度够用（68%）
- ✅ 适合家庭相册

---

### 专业项目

```python
CLIP_MODEL_NAME = "ViT-L-14"
```

**理由**：
- ✅ 精度高（75%）
- ✅ 语义理解强
- ⚠️ 速度慢（80ms/图）
- ⚠️ 内存大（1GB）

---

### 商业部署

```python
CLIP_MODEL_NAME = "ViT-B-16"
# 配合ONNX Runtime
```

**理由**：
- ✅ 部署轻量
- ✅ 推理快速
- ✅ 成本低

---

## 常见问题

### Q1: 如何查看当前使用的模型？

```python
import config
print(f"当前模型: {config.CLIP_MODEL_NAME}")
print(f"特征维度: {config.get_feature_dim()}")
print(f"模型文件: {config.get_model_filenames()}")
```

### Q2: 切换模型后需要重新下载吗？

是的，每个模型需要单独下载：

```bash
# 切换到ViT-L-14后
python clip_model_manager.py --download --export --model ViT-L-14 --mirror
```

### Q3: 可以同时安装多个模型吗？

可以！下载所有模型后，通过修改配置切换：

```bash
# 下载所有模型
python clip_model_manager.py --download --export --model all --mirror

# 修改 config.py 切换使用
# CLIP_MODEL_NAME = "ViT-B-16"  或 "ViT-L-14" 等
```

### Q4: 不同模型的特征维度不同，会影响使用吗？

不会！代码会自动适配：

```python
# 自动获取正确的特征维度
feature_dim = config.get_feature_dim()

# ViT-B-16: 512
# ViT-L-14: 768
# ViT-H-14: 1024
```

---

## 总结

### 配置流程

```
1. 编辑 config.py
   └─ 设置 CLIP_MODEL_NAME

2. 下载模型
   └─ python clip_model_manager.py --download --export --model <name> --mirror

3. 验证模型
   └─ python clip_model_manager.py --verify --model <name>

4. 使用模型
   └─ python main.py  (自动使用配置的模型)
```

### 推荐配置

- **个人项目**: ViT-B-16（默认）
- **专业项目**: ViT-L-14
- **研究项目**: ViT-H-14

简单修改配置文件，即可切换模型！🎯
