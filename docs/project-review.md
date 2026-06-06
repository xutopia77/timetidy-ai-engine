       



# 项目设计分析

这个文档是个人记录整理文档，AI不用分析，自己整理看的

## 设计文档与代码实现对比分析

### ✅ 已一致的部分

1. **协议格式** - 完全一致
   - 请求：`{"id": "...", "action": "...", "data": {...}}`
   - 响应：`{"id": "...", "code": 0, "status": "success", "data": {...}, "data_dbg": {...}}`

2. **action和cmd使用** - 完全一致
   - 所有engine都使用`params.get("cmd", "")`来区分子操作
   - action到engine的映射正确

3. **核心功能** - 基本一致
   - search、index_init、sync、tag_detect、find_similar、ping 都已实现

### ⚠️ 存在矛盾的部分



#### 建议3：简化 `detect_similar_groups` 功能

**理由：**
- 当前实现过于复杂（并查集、分组、代表图片等）
- 可以简化为返回相似对列表，由Go端处理分组逻辑

**建议简化方案：**
```python
# 当前返回格式（复杂）
{
  "groups": [
    {
      "group_id": 1,
      "images": [...],
      "avg_similarity": 0.95,
      "pair_count": 2
    }
  ]
}

# 建议简化格式
{
  "similar_pairs": [
    {"id1": 1001, "id2": 1002, "similarity": 0.95},
    {"id1": 1001, "id2": 1003, "similarity": 0.89}
  ]
}
```

#### 建议4：统一错误处理

**当前问题：**
- 有些地方使用 `raise ValueError`
- 有些地方使用 `raise RuntimeError`
- 错误信息格式不统一

**建议：**
```python
# 统一使用自定义异常
class AIEngineError(Exception):
    def __init__(self, message: str, code: int = 1):
        self.message = message
        self.code = code
        super().__init__(message)

# 使用示例
raise AIEngineError("record_id not found in index", code=2)
```

#### 建议5：简化测试代码

**当前问题：**
- `test_full.py` 文件过大（约800行）
- 测试逻辑和报告生成混在一起

**建议：**
- 已经将报告生成独立到 `report_generator.py` ✅
- 进一步拆分测试：每个engine一个测试文件
  - `test_search.py`
  - `test_similarity.py`
  - `test_tag_detect.py`
  - `test_index_init.py`

### 📊 复杂度降低效果预估

| 项目 | 当前行数 | 优化后行数 | 减少 |
|------|---------|-----------|------|
| similarity_engine.py | ~400 | ~200 | -50% |
| test_full.py | ~800 | ~400 | -50% |
| 设计文档 | ~2200 | ~1800 | -18% |
| **总计** | ~3400 | ~2400 | **-29%** |

### 🎯 优先级建议

1. **高优先级** - 删除 `find_similar_by_image`（设计文档已明确）
2. **中优先级** - 删除 `batch_find_similar`（收益小）
3. **低优先级** - 简化 `detect_similar_groups`（需要讨论）
4. **持续优化** - 统一错误处理、拆分测试

### ✅ 总结

**协议层面：** 设计文档和代码实现已经完全一致，这部分做得很好！

**功能层面：** 存在一些冗余功能，建议按照设计文档的指导思想"降低项目复杂度"进行清理。

**下一步行动：** 建议先处理高优先级的 `find_similar_by_image` 删除，这是设计文档明确要求的。



# 代码分析



# === `_compute_category_scores` 方法详解

这个方法是整个负向 Prompt 机制的核心，负责计算每个分类的最终得分。

```python
  def _compute_category_scores(
      self, 
      image_vector: np.ndarray, 
      vectors: np.ndarray, 
      prompt_counts: List[int],
      neg_vectors: Optional[np.ndarray] = None,
      neg_prompt_counts: Optional[List[int]] = None,
  ) -> List[float]:
      """Compute similarity scores for categories with optional negative prompts
      
      When negative prompts are provided, the score is adjusted as:
          adjusted_score = positive_score - neg_weight * negative_score
      
      Args:
          image_vector: Image feature vector
          vectors: Prompt embedding matrix
          prompt_counts: Number of prompts per category
          neg_vectors: Negative prompt embedding matrix (optional)
          neg_prompt_counts: Number of negative prompts per category (optional)
      
      Returns:
          List of adjusted similarity scores per category
      """
      settings = self._config.get("settings", {})
      enable_negative = settings.get("enable_negative_prompts", True)
      neg_weight = 0.5  # Weight for negative prompt penalty
      
      similarities = vectors @ image_vector
      
      scores = []
      start_idx = 0
      for i, count in enumerate(prompt_counts):
          end_idx = start_idx + count
          avg_score = float(similarities[start_idx:end_idx].mean())
          
          # Apply negative prompt penalty
          if (enable_negative and neg_vectors is not None 
                  and neg_prompt_counts is not None 
                  and neg_prompt_counts[i] > 0):
              neg_start = sum(neg_prompt_counts[:i])
              neg_end = neg_start + neg_prompt_counts[i]
              neg_similarities = neg_vectors[neg_start:neg_end] @ image_vector
              neg_score = float(neg_similarities.mean())
              avg_score = avg_score - neg_weight * neg_score
          
          scores.append(avg_score)
          start_idx = end_idx
      
      return scores
```

### 输入参数

| 参数                  | 含义                                                    |
| ------------------- | ----------------------------------------------------- |
| `image_vector`      | 图片的特征向量 (1024维)                                       |
| `vectors`           | 所有正向 prompt 的 embedding 矩阵，shape 为 `(总prompt数, 1024)` |
| `prompt_counts`     | 每个分类的正向 prompt 数量，如 `[7, 7, 6, ...]`                  |
| `neg_vectors`       | 所有负向 prompt 的 embedding 矩阵（可选）                        |
| `neg_prompt_counts` | 每个分类的负向 prompt 数量，如 `[5, 5, 5, ...]`（可选）              |

### 核心逻辑分三步

**第一步：正向得分计算（始终执行）**

```python
similarities = vectors @ image_vector  # 矩阵乘法，批量算出所有 prompt 与图片的余弦相似度
```

然后按 `prompt_counts` 分组求平均。比如 `prompt_counts = [7, 7, 6]`，就把前 7 个相似度取平均作为第 1 个分类的正向得分，接下来 7 个作为第 2 个，以此类推。

**第二步：负向惩罚（条件执行）**

当三个条件同时满足时才生效：
- `enable_negative_prompts` 配置开启
- 传入了 `neg_vectors`
- 当前分类有负向 prompt（`neg_prompt_counts[i] > 0`）

负向得分的计算方式与正向相同——对负向 prompt 的相似度取平均，然后乘以惩罚权重 `0.5` 后从正向得分中扣除：

```
adjusted_score = positive_score - 0.5 × negative_score
```

**举个例子**：假设"亲子育儿"分类：
- 正向 prompts（如"宝宝的笑脸照片"）与图片的平均相似度 = 0.30
- 负向 prompts（如"只有大人的照片"）与图片的平均相似度 = 0.15
- 最终得分 = 0.30 - 0.5 × 0.15 = **0.225**

如果图片里确实是宝宝，正向高、负向低，惩罚小；如果图片里是大人，正向低、负向高，惩罚大，从而把该分类的得分压下去。

**第三步：返回结果**

返回每个分类调整后的得分列表，供上层 `classify_level1` / `classify_level2` / `classify_attributes` 排序选优。

### 一个细节

`neg_start` 的计算用了 `sum(neg_prompt_counts[:i])`，这是因为负向 prompt 也是按分类顺序拼接成一个矩阵的，需要用累计偏移量来定位每个分类的负向 prompt 区间。


