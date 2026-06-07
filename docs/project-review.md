
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



# =============代码分析





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

**输入参数**

| 参数                  | 含义                                                    |
| ------------------- | ----------------------------------------------------- |
| `image_vector`      | 图片的特征向量 (1024维)                                       |
| `vectors`           | 所有正向 prompt 的 embedding 矩阵，shape 为 `(总prompt数, 1024)` |
| `prompt_counts`     | 每个分类的正向 prompt 数量，如 `[7, 7, 6, ...]`                  |
| `neg_vectors`       | 所有负向 prompt 的 embedding 矩阵（可选）                        |
| `neg_prompt_counts` | 每个分类的负向 prompt 数量，如 `[5, 5, 5, ...]`（可选）              |

**核心逻辑分三步**

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

**一个细节**

`neg_start` 的计算用了 `sum(neg_prompt_counts[:i])`，这是因为负向 prompt 也是按分类顺序拼接成一个矩阵的，需要用累计偏移量来定位每个分类的负向 prompt 区间。

## 分数尺度不一致问题

确实存在**分数尺度不一致**的问题。让我详细分析：

**没有负向向量的分类**

```
最终得分 = positive_score
```

假设正向相似度在 `[0, 0.4]` 区间，最终得分也在这个区间。

**有负向向量的分类**

```
最终得分 = positive_score - 0.5 × negative_score
```

假设：
- `positive_score = 0.30`
- `negative_score = 0.20`

最终得分 = `0.30 - 0.5 × 0.20 = 0.20`

**结果**：正向匹配度更高的分类，因为被惩罚，得分反而更低。

---

## 具体例子

| 分类 | 正向得分 | 负向得分 | 最终得分 | 有负向? |
|------|---------|---------|---------|--------|
| 亲子育儿 | 0.30 | 0.20 | **0.20** | ✓ |
| 其他 | 0.25 | 无 | **0.25** | ✗ |

"亲子育儿"的正向匹配度更高（0.30 > 0.25），但最终得分反而更低。这在排序选优时会造成**系统性偏差**——没有负向 prompt 的分类会被**不公平地优待**。

**是否需要统一？**

**需要**，但要看场景：

1. **如果所有分类都有负向 prompt** → 问题不存在（当前配置基本满足）
2. **如果部分分类没有负向 prompt**（如"其他"分类）→ 问题存在

当前配置中，"其他"分类的 `negative_prompts` 是空数组 `[]`，所以确实存在这个问题。

---

## 可能的解决方案

| 方案 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **方案A：统一惩罚** | 无负向时用默认惩罚值（如0） | 简单 | 不太合理 |
| **方案B：归一化** | `score / (1 + 0.5)` | 统一尺度 | 需要预估最大惩罚 |
| **方案C：乘法惩罚** | `positive × (1 - neg_weight × neg_score)` | 始终为正，尺度统一 | 逻辑不同 |
| **方案D：分离排序** | 有负向和无负向分开排序再合并 | 公平 | 复杂 |

**方案C** 可能是最优雅的：
```
adjusted_score = positive_score × (1 - 0.5 × negative_score)
```
- 无负向时：`positive × 1 = positive`
- 有负向时：`positive × (1 - 惩罚)`，惩罚在 `[0, 0.5]` 区间

这样所有分类的得分尺度都基于 `positive_score`，只是有负向的会被**按比例缩小**，而不是**绝对值减小**。

## 结论

当前实现确实存在分数尺度不一致的问题。如果需要修复，**方案C（乘法惩罚）** 是比较推荐的，既保持尺度统一，又不需要复杂的归一化逻辑。

## 方案C的问题

```
有负向：adjusted = positive × (1 - 0.5 × neg_score) = positive × 0.9（假设neg=0.2）
无负向：adjusted = positive × 1
```

**结果**：有负向的分类得分被缩小到原来的 90%，无负向的保持 100%。尺度仍然不一致。

### 根本问题

无论用什么公式（减法、乘法），只要**有的分类有惩罚、有的没有**，尺度就不可能一致。

### 真正的解决方案

**方案一：配置层面解决** —— 确保所有分类都有负向 prompt

给"其他"分类也加上有意义的负向 prompt，比如：
```json
{
  "id": "other",
  "name": "其他",
  "negative_prompts": [
    "清晰的照片",
    "有明确主题的照片"
  ]
}
```

这样所有分类都在同一尺度下竞争。

**方案二：代码层面兜底** —— 无负向时使用默认惩罚

```python
if neg_prompt_counts[i] > 0:
    # 正常计算负向得分
    neg_score = ...
else:
    # 无负向时，使用一个默认的低惩罚值（如0）
    neg_score = 0
```

这样公式统一为 `positive - 0.5 × neg_score`，无负向的分类 `neg_score=0`，相当于不惩罚。

**方案三：重新理解设计意图**

也许这种"不公平"是**有意为之**的：
- 有负向 prompt 的分类 = 定义明确、边界清晰
- 无负向 prompt 的分类 = 兜底分类，应该**更容易被选中**

当其他分类因为负向惩罚得分降低时，"其他"分类自然浮上来，这正好符合兜底的语义。

### 我的建议

**方案一**最干净 —— 在配置中给所有分类补上负向 prompt，从源头消除不一致。


---


