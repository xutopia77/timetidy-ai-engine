
# AI相册的AI核心功能设计

这个文档主要是AI相册的AI核心功能设计，包括搜索、标签分类、图片描述生成、图片相似度计算等。设计部分可以不用放置大量详细代码。


## AI核心功能设计实现方案

### 1. 整体架构设计

#### 1.1 系统架构

**双语言协作架构（基于stdio通信）：**

```
┌─────────────────────────────────────────────────────────────┐
│  相册主体功能（Go开发）                                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  SQLite数据库                                             ││
│  │  - 图片完整信息（路径、大小、分辨率、EXIF等）                 ││
│  │  - 业务数据（相册、标签、用户信息等）                         ││
│  │  - 记录ID（record_id）                                    ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  文件管理模块                                             ││
│  │  - 图片导入、删除、移动、编辑                               ││
│  │  - 文件系统监听                                           ││
│  │  - 事件发送（通知AI模块）                                  ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  AI通信模块（PythonProcess）                              ││
│  │  - 启动Python子进程                                       ││
│  │  - stdin发送JSON请求                                      ││
│  │  - stdout接收JSON响应                                     ││
│  │  - stderr接收日志信息                                     ││
│  │  - 管理请求-响应匹配（pending map）                        ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              ↕ stdio通信（stdin/stdout）
┌─────────────────────────────────────────────────────────────┐
│  AI核心功能模块（Python开发）                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  stdio通信服务                                            ││
│  │  - 从stdin读取JSON请求                                    ││
│  │  - 向stdout写入JSON响应                                   ││
│  │  - 向stderr输出日志信息                                   ││
│  │  - 请求路由和分发                                         ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  核心引擎层                                               ││
│  │  - SearchEngine: 文搜图、图搜图                            ││
│  │  - HierarchicalTagEngine: 分层标签生成                    ││
│  │  - DescribeEngine: 图片描述生成                           ││
│  │  - SimilarityEngine: 图片相似度计算                       ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  向量数据库层（FAISS）                                     ││
│  │  - IndexIDMap2(IndexIVFFlat)                             ││
│  │  - 支持删除、自定义ID                                     ││
│  │  - 持久化索引文件                                         ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  元数据管理层                                             ││
│  │  - ai_metadata.json（最小化）                             ││
│  │  - vector_id ↔ record_id 映射                            ││
│  │  - file_hash（检测变化）                                 ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  特征提取层（CLIP ONNX）                                   ││
│  │  - Vision Encoder: 图像特征提取                           ││
│  │  - Text Encoder: 文本特征提取                             ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**关键设计原则：**

原则1: 语言边界清晰
- Go负责：文件管理、数据库、用户界面、业务逻辑
- Python负责：AI推理、特征提取、向量搜索、AI功能
- 通过stdio通信，保持边界清晰

原则2: 数据解耦
- Go维护SQLite（完整业务数据）
- Python维护最小化元数据（仅AI必需）
- 通过record_id映射，避免数据冗余

原则3: 进程常驻
- Python进程常驻后台（避免每次启动开销）
- Go启动时自动启动Python子进程
- Python进程崩溃自动重启（可选）

原则4: 容错设计
- AI模块独立进程（故障不影响相册主体）
- 自动恢复机制
- 手动重建接口

原则5: 性能优化
- 向量数据库毫秒级搜索
- 特征缓存复用
- 批量编码加速

**stdio通信协议：**

通信方式：
- Go启动Python子进程（python main.py）
- Go通过stdin发送JSON请求（每行一个JSON对象）
- Python通过stdout返回JSON响应（每行一个JSON对象）
- Python通过stderr输出日志信息

```json
请求格式：
{
  "id": "1234567890", // 请求ID（时间戳）
  "action": "search", // 操作类型，也就是对应的engine
  "data": {           // 参数，根据每一个action有所不同
    "cmd": "search",  // 子操作（可选）
    "query": "风景",
    "top_k": 10
  }
}

响应格式：
{
  "id": "1234567890",   // 请求ID（与请求匹配）
  "code": 0,            // 0 成功，1 失败，其他状态根据action不同可以自由扩展
  "status": "success",  // 状态，成功时是success，错误时是错误原因
  "data": {             // Go需要的核心数据（简洁易解析）
    "results": [...]    // 每一个请求都有自己的数据结果
  },
  "data_dbg": {           // 辅助调试数据（可选，不影响Go处理）
    "query_time_ms": 3.5, // 调试信息，如处理时间
    "results": [...]      // 其他调试信息
  }
}
```

**响应格式设计说明：**

- `data`: Go需要的核心数据，简洁易解析
  - 只包含Go业务逻辑需要的数据
  - 结构简单，减少解析复杂度
  - 示例：搜索结果、标签列表、描述文本等

- `data_dbg`: 辅助调试数据，可选字段
  - 包含调试信息，不影响Go业务逻辑
  - 记录详细处理过程，方便问题排查
  - 示例：处理时间、中间结果、详细日志等

**优势：**
- Go解析简单：只需解析`data`字段
- 调试信息完整：`data_dbg`保留所有调试信息
- 向后兼容：`data_dbg`可选，不影响现有逻辑

**支持的action列表：**

| action | 说明 | 子操作（data.cmd） |
|--------|------|----------------------------------------|
| search | 文搜图、图搜图 | search、encode_image、encode_text、get_status |
| index_init | 索引初始化 | start、stop、continue、get_progress、get_status |
| sync | 数据同步 | sync_all、get_status |
| tag_detect | 标签检测 | generate_tags |
| find_similar | 相似图搜索 | find_similar、detect_similar_groups |
| ping | 通信测试 | 无 |

优势：
- 进程常驻，避免每次启动Python的开销
- 简单的请求-响应模型
- 易于调试（stderr输出日志）


#### 1.2 技术栈选择

**Go相册主体技术栈：**

- 语言: Go 1.20+
- 数据库: SQLite（业务数据）
- AI通信: PythonProcess（管理Python子进程）
  - stdin/stdout JSON通信
  - 请求-响应匹配（pending map）
  - 超时控制（120秒）
- 文件监听: fsnotify（监听图片目录变化）
- 日志: logrus/zap
- 配置: viper

**Python AI模块技术栈：**

- 语言: Python 3.8+
- 通信方式: stdio（标准输入输出）
  - 从stdin读取JSON请求
  - 向stdout写入JSON响应
  - 向stderr输出日志信息
- 向量数据库: FAISS(IndexIDMap2(IndexIVFFlat))
- 特征提取: Chinese-CLIP ONNX
- 日志: loguru
- 配置: pydantic

### 2. AI核心功能详细设计

#### 2.1 文搜图、图搜图（SearchEngine）

**功能概述：**

```
SearchEngine 提供两种搜索模式：

┌─────────────────────────────────────────────────────────────────┐
│  文搜图（search）                                                │
│  - 输入：文本查询（如"风景"、"人物"、"美食"）                       │
│  - 流程：文本编码 → FAISS搜索 → 返回结果                          │
│  - 性能：~2-5ms                                                  │
├─────────────────────────────────────────────────────────────────┤
│  图搜图（search_by_image）                                       │
│  - 输入：参考图片 record_id（图片已在索引中）                      │
│  - 流程：从索引获取向量 → FAISS搜索 → 返回结果                     │
│  - 性能：~2-5ms                                                  │
│  - 实现：复用 SimilarityEngine.find_similar                      │
└─────────────────────────────────────────────────────────────────┘

```

**文搜图（search）：**

```
功能需求：
- 输入：搜索文本（如"风景"、"人物"、"美食"）
- 输出：图片列表（record_id、相似度分数）

核心流程：
1. Go发送搜索请求（通过stdin）
2. Python接收请求并提取文本特征（CLIP Text Encoder）
3. FAISS向量搜索（IndexIDMap2）
4. Python返回结果（通过stdout）
5. Go接收结果并处理

性能优化：
- 文本特征缓存（相同查询复用）
- 搜索时间：~2-5ms
```

**图搜图（search_by_image）：**

```
功能需求：
- 输入：参考图片 record_id（图片已在索引中）
- 输出：相似图片列表（record_id、相似度分数）

核心流程：
1. Go发送图搜图请求（通过stdin）
2. Python从FAISS索引获取参考图片向量（无需编码）
3. FAISS向量搜索
4. Python返回结果（通过stdout）
5. Go接收结果并处理

关键设计：
- 复用 SimilarityEngine.find_similar 实现
- 图片向量已在索引中，无需重新编码
- 搜索时间：~2-5ms

如果图片不在索引中（外部图片），就返回错误，不再处理，降低项目复杂度。
```

**函数接口设计：**

```
文搜图接口：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "search",
     "data": {
       "cmd": "search", // 可选，默认为search
       "query": "风景照片",
       "top_k": 10,
       "filters": {
         "min_similarity": 0.5
       }
     }
   }

2. 返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "results": [
         {"record_id": 1001, "similarity": 0.95, "rank": 1},
         {"record_id": 1002, "similarity": 0.89, "rank": 2}
       ]
     },
     "data_dbg": {
       "query_time_ms": 3.5
     }
   }

其他接口：
- encode_image: 编码图片特征
- encode_text: 编码文本特征
- get_status: 获取引擎状态

图搜图接口（使用 find_similar action）：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "find_similar",
     "data": {
       "cmd": "find_similar",
       "record_id": 1001,
       "top_k": 10,
       "min_similarity": 0.5
     }
   }

2. 返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "similar_images": [
         {"record_id": 1005, "similarity": 0.95, "rank": 1},
         {"record_id": 1020, "similarity": 0.89, "rank": 2}
       ]
     },
     "data_dbg": {
       "query_time_ms": 3.2
     }
   }
```

**引擎复用关系：**

```
SearchEngine（搜索引擎）
├── search               → 文搜图（文本编码 + FAISS搜索）
├── encode_image         → 编码图片特征
├── encode_text          → 编码文本特征
└── get_status           → 获取引擎状态

SimilarityEngine（相似度引擎）
├── find_similar           → 图搜图（支持批量）
└── detect_similar_groups  → 全相册相似检测（N-to-N）

结论：
- 图搜图 = SimilarityEngine.find_similar（默认场景）
- SearchEngine 主要用于文搜图，图搜图使用 find_similar action
```

#### 2.2 分层标签生成功能（HierarchicalTagEngine）

**功能需求：**

用户输入：
- **record_id（必需）**：图片在FAISS索引中的ID，用于获取图片向量
- file_path（可选）：图片路径，仅用于日志记录
- 标签层级选择（level1, level2, attributes, all）

返回结果：
- 分层标签结果：
  - 一级分类（旅游出行、日常生活、亲子育儿等）
  - 二级分类（国内游、出境游、周边游等）
  - 属性标签（白天、夜晚、晴天、春季等）

性能特点：
- **快速**：从FAISS获取图片向量（~1ms），不编码图片（~500ms）
- **配置调优友好**：调整配置后无需重新编码图片，适合快速迭代

**设计理念：**

合并分类和标签生成功能，使用统一的分层标签体系：

1. 技术本质相同：
   - 都是计算图片特征与文本特征的相似度
   - 都基于CLIP模型
   - 都是标签匹配过程

2. 用户体验更好：
   - 一次调用，同时得到分类和标签
   - 分类和标签是同一体系的不同层级，逻辑清晰

3. 性能更优：
   - 一次计算，避免重复提取特征
   - 减少内存占用和计算时间

4. 易于扩展：
   - 可以灵活添加新的标签层级
   - 支持用户自定义标签库（JSON配置文件）

**分层标签体系设计：**

三级标签体系：

一级分类（Level 1）- 10个大类：
- 旅游出行、日常生活、其他
- ...（详见配置文件）

二级分类（Level 2）- 每个一级分类下的细分：
- 旅游出行 → 国内游、出境游、周边游、景点打卡、旅行风景
- ...（详见配置文件）

属性标签（Attributes）- 时间、天气、季节等：
- 时间：白天、夜晚、黄昏
- ...（详见配置文件）

配置文件：
- 位置：src/config/hierarchical_tags.json
- 用户可自定义修改标签体系
- 支持多提示词（multiple prompts）提高准确度

**实现方案：**

核心流程：
1. Go发送标签生成请求（通过stdin，必须提供record_id）
2. Python接收请求并从FAISS索引获取图片向量（~1ms）
3. 计算图片向量与各层级标签文本向量的相似度
4. Python返回分层结果（通过stdout）
5. Go接收结果并处理

关键设计：
- **必须提供record_id**：从FAISS索引获取图片向量，不再编码图片
- **性能优化**：图片向量获取从~500ms降至~1ms（提升500倍）
- 从JSON配置文件加载标签体系
- 标签文本向量预计算并缓存（NPZ文件）
- 支持正向和负向提示词（negative prompts）
- 分层置信度阈值（level1: 0.22, level2: 0.18, attributes: 0.20）
- 支持用户自定义标签库

性能优化：
- **图片向量获取**：从FAISS索引获取（~1ms），不编码图片（~500ms）
- 标签文本向量预计算并缓存（一次性，~10秒）
- 支持正向和负向提示词机制
- 批量标签生成支持

**配置调优流程：**

使用FAISS向量加速配置调优：

```
第一步：生成FAISS索引（一次性）
    ↓
耗时：~83分钟（1万张图片）
    ↓
第二步：调整配置文件（hierarchical_tags.json）
    ↓
第三步：批量测试（使用FAISS向量）
    ↓
耗时：~30秒（1万张图片）
    ↓
第四步：查看结果，分析问题
    ↓
重复步骤 2-4，直到满意
```

调优优势：
- **第一次**：生成FAISS索引（编码所有图片）
- **后续**：每次调整配置只需~30秒（1万张图片）
- **节省时间**：假设迭代10次，从~14小时降至~88分钟

**函数接口设计：**

```
Python处理流程：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "tag_detect",
     "data": {
       "cmd": "generate_tags",  // 可选，默认为generate_tags
       "record_id": 1001,       // 必需，用于从FAISS获取图片向量
       "file_path": "/path/to/image.jpg",  // 可选，仅用于日志
       "tag_level": "all"       // "level1", "level2", "attributes", "all"
     }
   }

参数说明：
- record_id（必需）：图片在FAISS索引中的ID，用于获取图片向量
- file_path（可选）：图片路径，仅用于日志记录
- tag_level（可选）：标签层级，默认为"all"

错误处理：
- 如果 record_id 为 None：抛出 ValueError
- 如果 record_id 在 FAISS 中不存在：抛出 ValueError

2. 处理请求并返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "tags": [
         {"label": "旅游出行", "confidence": 0.92},
         {"label": "景点打卡", "confidence": 0.88},
         {"label": "白天", "confidence": 0.90},
         {"label": "晴天", "confidence": 0.85},
         {"label": "夏季", "confidence": 0.78}
       ]
     },
     "data_dbg": {
       "primary_category": {
         "id": "travel",
         "name": "旅游出行",
         "confidence": 0.92,
         "level": 1
       },
       "level1_tags": [
         {"id": "travel", "name": "旅游出行", "confidence": 0.92, "level": 1},
         {"id": "outdoor", "name": "户外自然", "confidence": 0.75, "level": 1}
       ],
       "level2_tags": [
         {"id": "scenic_spot", "name": "景点打卡", "confidence": 0.88, "level": 2, "parent_id": "travel"},
         {"id": "travel_scenery", "name": "旅行风景", "confidence": 0.82, "level": 2, "parent_id": "travel"}
       ],
       "attribute_tags": [
         {"id": "daytime", "name": "白天", "confidence": 0.90, "level": "attribute", "type": "time"},
         {"id": "summer", "name": "夏季", "confidence": 0.78, "level": "attribute", "type": "season"}
       ]
       "process_time_ms": 50
     }
   }
```

**配置文件说明version 2.0：**

```
配置文件路径：src/config/hierarchical_tags.json

配置文件结构：
{
  "version": "2.0",
  "description": "Hierarchical tag configuration for family photo album",
  "level1_categories": [
    {
      "id": "travel",
      "name": "旅游出行",
      "prompts": [
        "一张旅游的照片",
        "旅行时拍的照片",
        "外出旅游的照片"
      ]
    },
    // ... 其他一级分类
  ],
  "level2_categories": {
    "travel": [
      {
        "id": "domestic_travel",
        "name": "国内游",
        "prompts": [
          "一张国内旅游的照片",
          "国内景点游玩照片"
        ]
      },
      // ... 其他二级分类
    ]
  },
  "attribute_categories": {
    "time": [
      {
        "id": "daytime",
        "name": "白天",
        "prompts": ["一张白天拍摄的照片", "日光下的照片"]
      }
    ],
    "weather": [...],
    "season": [...]
  },
  "settings": {
    "level1_threshold": 0.25,
    "level2_threshold": 0.20,
    "attribute_threshold": 0.25,
    "top_k_level1": 3,
    "top_k_level2": 5,
    "top_k_attributes": 2,
    "use_multiple_prompts": true,
    "average_prompt_embeddings": true
  }
}

用户自定义：
- 可以修改标签名称和ID
- 可以添加新的标签类别
- 可以修改提示词（prompts）提高准确度
- 可以调整阈值和top_k参数
```

**配置文件说明version 3.0：**
```
{
  "version": "3.0",
  "description": "Hierarchical tag configuration for family photo album (Chinese-CLIP optimized v3.0)",
  "update_time": "2026-06-05",
  "level1_categories": [
    {
      "id": "parenting",
      "name": "亲子育儿",
      "prompts": [
        "宝宝的笑脸照片",
        "孩子在玩耍的照片",
        "孩子睡觉的照片"
      ],
      "negative_prompts": [
        "只有大人的照片",
        "纯物品照片"
      ]
    }
  ],
  "level2_categories": {
    "parenting": [
      {
        "id": "baby_growth",
        "name": "宝宝成长",
        "prompts": [
          "宝宝刚出生的照片",
          "宝宝满月的照片",
          "宝宝第一次说话的照片"
        ],
        "negative_prompts": [
          "有大人的照片",
          "聚会照片"
        ]
      },
      {
        "id": "parent_child",
        "name": "亲子互动",
        "prompts": [
          "父母和孩子一起玩耍的照片",
          "亲子游戏的照片"
        ],
        "negative_prompts": [
          "只有孩子的照片",
          "纯美食照片"
        ]
      }
    ]
  },
  "attribute_categories": {
    "time": [
      {
        "id": "early_morning",
        "name": "清晨",
        "prompts": [
          "清晨拍摄的照片",
          "天刚亮的照片"
        ]
      },
      {
        "id": "daytime",
        "name": "白天",
        "prompts": [
          "白天拍摄的照片",
          "下午的照片"
        ]
      }
    ]
  },
  "settings": {
    "level1_threshold": 0.22,
    "level2_threshold": 0.18,
    "attribute_threshold": 0.20,
    "top_k_level1": 1,
    "top_k_level2": 2,
    "top_k_attributes": 3,
    "use_multiple_prompts": true,
    "average_prompt_embeddings": true,
    "enable_negative_prompts": true,
    "hierarchical_filtering": true,
    "confidence_weighting": true,
    "conflict_resolution_priority": [
      "parenting",
      "celebration",
      "travel",
      "daily_life",
      "portrait",
      "food",
      "pet",
      "landscape",
      "hobby",
      "objects",
      "transport",
      "other"
    ]
  }
}

- 负向 Prompt 机制
- hierarchical_filtering会确保只有一级分类命中后才计算其下的二级标签，彻底解决跨分类重叠问题
- conflict_resolution_priority定义了分类优先级，当两个分类相似度接近时自动选择高优先级的分类

```



#### 2.3 图片相似度搜索功能（SimilarityEngine）

**功能概述：**

```
SimilarityEngine 提供多种相似度搜索模式：

┌─────────────────────────────────────────────────────────────────┐
│  find_similar（默认）                                            │
│  - 输入：参考图片 record_id 列表（支持批量）                       │
│  - 流程：从索引获取向量 → FAISS搜索 → 返回结果                     │
│  - 性能：~2-5ms（单个），~10-20ms（批量）                         │
├─────────────────────────────────────────────────────────────────┤
│  detect_similar_groups                                          │
│  - 输入：全相册图片                                              │
│  - 流程：N-to-N相似检测 → Union-Find分组 → 返回相似组              │
│  - 性能：~2-3s（10,000张图片）                                    │
└─────────────────────────────────────────────────────────────────┘
```

**函数接口设计：**

```
find_similar接口：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "find_similar",
     "data": {
       "cmd": "find_similar",  // 可选，默认为find_similar
       "record_ids": [1001, 1002, 1003],
       "top_k": 10,
       "min_similarity": 0.5
     }
   }

2. 返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "results": [
         {
           "reference_id": 1001,
           "similar_images": [
             {"record_id": 1005, "similarity": 0.95, "rank": 1},
             {"record_id": 1020, "similarity": 0.89, "rank": 2}
           ]
         },
         {
           "reference_id": 1002,
           "similar_images": [...]
         }
       ],
       "total_time_ms": 15.0,
       "batch_size": 3
     },
     "data_dbg": {
       "query_time_ms": 15.0
     }
   }

detect_similar_groups接口：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "find_similar",
     "data": {
       "cmd": "detect_similar_groups",
       "min_similarity": 0.9,
       "min_group_size": 2
     }
   }

2. 返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "similar_groups": [
         {
           "group_id": 1,
           "members": [1001, 1005, 1020],
           "avg_similarity": 0.95
         },
         {
           "group_id": 2,
           "members": [1002, 1008],
           "avg_similarity": 0.92
         }
       ]
     },
     "data_dbg": {
       "total_time_ms": 2500.5
     }
   }
```

#### 2.4 图片描述生成功能（DescribeEngine）

**功能需求：**

用户输入：
- 图片record_id

返回结果：
- 图片的自然语言描述（如："一张美丽的日落风景照片，远处是山脉，天空呈现橙红色"）

**实现方案：**

```
核心流程：
1. Go发送描述生成请求（通过stdin）
2. Python接收请求并提取图片特征（CLIP Vision Encoder）
3. 使用预训练模型生成描述（可选方案）
4. Python返回结果（通过stdout）
5. Go接收结果并处理

实现方案对比：

方案1: 基于标签组合（推荐）
- 先生成分层标签（HierarchicalTagEngine）
- 将标签组合成自然语言描述
- 优点：简单、快速、可控
- 缺点：描述不够丰富

方案2: 使用图像描述模型
- 使用BLIP、GIT等图像描述模型
- 直接生成自然语言描述
- 优点：描述丰富、自然
- 缺点：模型较大、推理较慢

推荐方案1（标签组合）：
tags = ["山", "日落", "风景", "夏天"]
description = "这是一张风景照片，包含山脉和日落景象，拍摄于夏天"

模板示例：
- 风景类："这是一张{category}照片，包含{tags}"
- 人物类："这是一张{category}照片，人物在{scene}"
- 美食类："这是一张{category}照片，展示了{food}"

性能优化：
- 标签生成复用HierarchicalTagEngine
- 描述模板预定义
- 批量生成支持
```

**函数接口设计：**

```
Python处理流程：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "generate_description",
     "data": {
       "record_id": 1001,
       "language": "zh"  // 中文描述
     }
   }

2. 处理请求并返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "description": "这是一张风景照片，包含山脉和日落景象，天空呈现橙红色"
     },
     "data_dbg": {
       "tags_used": ["山", "日落", "风景", "橙红色"],
       "process_time_ms": 80
     }
   }

设计说明：
- data: Go需要的核心数据
  - description: 图片描述文本
- data_dbg: 辅助调试数据
  - record_id: 记录ID
  - tags_used: 使用的标签列表
  - process_time_ms: 处理时间
```

#### 2.4 图片相似度计算功能（SimilarityEngine）

**功能概述：**

```
SimilarityEngine 提供多种相似度计算功能：

┌─────────────────────────────────────────────────────────────────┐
│  find_similar（图搜图，默认场景）                                 │
│  - 输入：参考图片 record_id（图片已在索引中）                      │
│  - 流程：从索引获取向量 → FAISS搜索 → 返回结果                     │
│  - 性能：~2-5ms                                                  │
│  - 说明：等同于 SearchEngine.search_by_image                      │
├─────────────────────────────────────────────────────────────────┤
│  detect_similar_groups（全相册相似检测）                          │
│  - 输入：相册范围（全部或部分图片）                                │
│  - 流程：批量搜索 → 并查集分组 → 返回分组结果                      │
│  - 性能：~2-3s（10,000张图片）                                    │
│  - 说明：N-to-N 相似检测，用于清理重复图片                         │
└─────────────────────────────────────────────────────────────────┘

核心设计：
- 图片特征已在FAISS索引中（无需重新提取）
- 使用向量相似度搜索（余弦相似度/内积）
- 支持相似度阈值过滤
- 排除参考图片本身
```

**find_similar（图搜图，默认场景）：**

```
功能需求：
- 输入：参考图片 record_id（图片已在索引中）
- 输出：相似图片列表（record_id、相似度分数）

核心流程：
1. Go发送相似度计算请求（通过stdin）
2. Python接收请求并从FAISS索引获取参考图片特征
3. 在FAISS索引中搜索相似向量
4. Python返回结果（通过stdout）
5. Go接收结果并处理

关键设计：
- 图片特征已在FAISS索引中（无需重新提取）
- 使用向量相似度搜索（余弦相似度）
- 支持相似度阈值过滤
- 排除参考图片本身

性能：
- 直接使用FAISS索引（无需重新编码）
- 搜索时间：~2-5ms
```

**函数接口设计：**

```
Python处理流程：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "find_similar",
     "data": {
       "record_id": 1001,
       "top_k": 10,
       "min_similarity": 0.8
     }
   }

2. 处理请求并返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "similar_images": [
         {
           "record_id": 1005,
           "similarity": 0.95,
           "rank": 1
         },
         {
           "record_id": 1020,
           "similarity": 0.89,
           "rank": 2
         }
       ]
     },
     "data_dbg": {
       "process_time_ms": 80
     }
   }
```

**全相册相似图片检测（detect_similar_groups）：**

```
功能需求：
- 检测整个相册中哪些图片是相似的
- 返回相似图片分组（而非扁平列表）
- 支持指定范围（全部或部分图片）

与 find_similar 的区别：
┌─────────────────────────────────────────────────────────────────┐
│  find_similar（单图查相似）                                      │
│  - 查询模式：1-to-N                                              │
│  - 输入：1张参考图片                                              │
│  - 输出：与该图片相似的top-K张图片                                 │
│  - 适用场景：用户查看某张图片的相似图片                             │
├─────────────────────────────────────────────────────────────────┤
│  detect_similar_groups（全相册相似检测）                          │
│  - 查询模式：N-to-N                                              │
│  - 输入：相册范围（全部或部分图片）                                 │
│  - 输出：相似图片分组列表                                          │
│  - 适用场景：用户清理重复/相似图片                                  │
└─────────────────────────────────────────────────────────────────┘

核心算法：
1. 从FAISS索引提取所有向量（或指定范围的向量）
2. 使用FAISS批量搜索：每个向量搜top-K近邻
   - 一次batch_search完成所有查询（FAISS内部优化）
3. 过滤：只保留相似度 ≥ min_similarity 的结果对
4. 去重：只保留 record_id_A < record_id_B 的对
5. 分组：用并查集（Union-Find）合并传递性相似关系
   - A≈B, B≈C → A,B,C归为同一组

性能估算：
- 10,000张图片，512维向量
- FAISS batch search：~1-2秒
- 并查集合并：~100ms
- 总计：~2-3秒（vs 逐个查询20-50秒+通信开销）
```

**函数接口设计：**

```
Python处理流程：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "find_similar",
     "data": {
       "cmd": "detect_similar_groups",
       "min_similarity": 0.9,
       "scope": "all",
       "record_ids": [],
       "top_k_per_image": 20
     }
   }

参数说明：
- min_similarity: 相似度阈值（越高越严格）
  - 0.95+：几乎完全相同的图片（连拍、备份副本）
  - 0.90-0.95：非常相似的图片（轻微裁剪、滤镜差异）
  - 0.85-0.90：相似图片（同一场景不同角度）
- scope: 范围 "all" 全部 / "partial" 指定record_ids
- record_ids: scope=partial时指定的图片ID列表
- top_k_per_image: 每张图片搜索的近邻数（默认20）

2. 处理请求并返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "groups": [
         {
           "group_id": 1,
           "images": [
             {"record_id": 1001, "representative": true},
             {"record_id": 1005, "similarity_to_representative": 0.95},
             {"record_id": 1020, "similarity_to_representative": 0.91}
           ],
           "avg_similarity": 0.93
         },
         {
           "group_id": 2,
           "images": [
             {"record_id": 2001, "representative": true},
             {"record_id": 2003, "similarity_to_representative": 0.92}
           ],
           "avg_similarity": 0.92
         }
       ]
     },
     "data_dbg": {
       "query_time_ms": 2100
     }
   }

```

**并查集（Union-Find）算法：**

```
用于合并传递性相似关系：

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

使用示例：
- A≈B → union(A, B)
- B≈C → union(B, C)
- 结果：A, B, C 在同一组（find(A) == find(B) == find(C)）
```

### 3. 数据同步机制设计

#### 3.1 统一数据同步架构

**核心设计理念：**

```
简化原则：
1. Go负责维护所有数据（SQLite数据库、文件系统、file_hash）
2. Python只负责AI功能（特征提取、向量搜索）
3. 通过process_file.json统一传递文件信息（包含file_hash）
4. Python通过file_hash对比决定是否更新索引
5. 减少实时事件同步，改为批量同步

关键设计：
- file_hash统一由Go维护（计算文件内容hash）
- Python只检查file_hash是否变化
- 如果file_hash变化，Python重新处理该文件
- 如果file_hash相同，Python跳过该文件

优势：
- 降低系统复杂度（无需复杂的元数据对比）
- 减少通信频率（批量同步而非实时同步）
- 提高可靠性（Go完全控制数据一致性）
- 易于维护（单一数据源）
- 简化Python端逻辑（只检查file_hash）
```

**同步流程：**

```
Go相册主体 → Python AI模块（通过process_file.json）

同步触发时机：
1. 用户首次使用AI功能（首次构建索引）
2. 用户手动触发同步（如导入大量新图片后）
3. 定期自动同步（如每天凌晨）
4. 用户请求重建索引

同步方式：
- Go生成process_file.json（包含所有文件信息和file_hash）
- Go触发FAISS索引构建（发送start请求）
- Python读取process_file.json
- Python检查每个文件的file_hash是否变化
- Python处理file_hash变化的文件（新增或更新）
- Python跳过file_hash未变化的文件
- Python返回同步结果
```

**同步函数接口设计：**

```
IndexInitEngine接口：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "index_init",
     "data": {
       "cmd": "start",  // start、stop、continue、get_progress（默认）、get_status
       "rebuild": false
     }
   }

2. 读取process_file.json：
   {
     "version": "1.0.0",
     "batch": 10,
     "files": [
       {
         "record_id": 1001,
         "file_path": "/path/to/image.jpg",
         "file_hash": "abc123",  // Go计算的文件hash
         "status": "pending"
       },
       // ... 所有文件列表
     ]
   }

3. 检查file_hash是否变化：
   - 加载已有元数据（ai_metadata.json）
   - 对比每个文件的file_hash
   - 如果file_hash变化（或不存在），重新处理
   - 如果file_hash相同，跳过

4. 处理文件并更新索引：
   - 删除旧向量（如果存在）
   - 提取新特征
   - 添加新向量
   - 更新元数据（保存file_hash）

5. 返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "total_files": 10000,
       "processed_files": 50,
       "skipped_files": 9950,
       "failed_files": 0
     },
     "data_dbg": {
       "query_time_ms": 2100
     }
   }
```

#### 3.2 file_hash对比同步逻辑

**核心设计理念：**

```
关键问题：如何识别需要处理的文件？

解决方案：通过file_hash对比
- Go生成process_file.json时，包含每个文件的file_hash
- Python读取process_file.json，加载已有元数据
- Python对比每个文件的file_hash
- 如果file_hash变化（或不存在），重新处理
- 如果file_hash相同，跳过

处理逻辑：
1. 新增文件：file_hash在process_file.json中存在，但ai_metadata.json中不存在该record_id
   - 处理方式：直接添加向量
   
2. 更新文件：file_hash发生变化（与ai_metadata.json中的file_hash不同）
   - 处理方式：先删除旧向量，再添加新向量
   
3. 无变化文件：file_hash相同（与ai_metadata.json中的file_hash相同）
   - 处理方式：跳过（不处理）

优势：
- 简单直观（只检查file_hash）
- 降低Python端复杂度
- 提高可靠性（Go完全控制file_hash）
- 高效（只处理需要同步的文件）
```


**file_hash对比流程图：**

```
┌─────────────────────────────────────────────────────────────┐
│  Go生成process_file.json                                     │
│  - 从SQLite获取所有图片record_id、file_path                   │
│  - Go计算每个文件的file_hash（文件内容hash）                    │
│  - 所有文件status设为pending                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Python读取process_file.json                                  │
│  - 获取所有文件列表（包含file_hash）                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Python加载已有元数据（ai_metadata.json）                      │
│  - 获取已有record_id和file_hash                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  file_hash对比                                                │
│  - 对比每个文件的file_hash                                     │
│  - file_hash变化：需要处理                                     │
│  - file_hash相同：跳过                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  处理文件                                                      │
│  1. 如果record_id已存在，删除旧向量                            │
│  2. 提取新特征                                                 │
│  3. 添加新向量                                                 │
│  4. 更新元数据（保存file_hash）                                │
│  5. 更新进度文件                                               │
│  6. 保存索引                                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  返回同步结果                                                  │
│  - total_files: 10000                                         │
│  - processed_files: 50                                        │
│  - skipped_files: 9950                                        │
└─────────────────────────────────────────────────────────────┘
```

#### 3.3 同步触发时机

**同步触发场景：**

```
场景1：首次使用AI功能
- Go检测到FAISS索引不存在
- Go从SQLite获取所有图片信息
- Go计算每个文件的file_hash
- Go生成process_file.json
- Go发送start请求（rebuild=false）
- Python构建索引（所有文件file_hash都不存在，全部处理）

场景2：用户手动触发同步
- 用户导入大量新图片（如100张）
- 用户点击"同步AI索引"按钮
- Go从SQLite获取所有图片信息
- Go计算每个文件的file_hash
- Go生成process_file.json
- Go发送start请求（rebuild=false）
- Python通过file_hash对比识别新增文件
- Python只处理新增文件（高效）

场景3：定期自动同步
- Go每天凌晨自动触发同步
- Go从SQLite获取所有图片信息
- Go计算每个文件的file_hash
- Go生成process_file.json
- Go发送start请求（rebuild=false）
- Python通过file_hash对比识别变化
- Python只处理变化的文件

场景4：用户请求重建索引
- 用户点击"重建AI索引"按钮
- Go从SQLite获取所有图片信息
- Go计算每个文件的file_hash
- Go生成process_file.json
- Go发送start请求（rebuild=true）
- Python删除现有索引
- Python重新处理所有文件
```

**同步频率建议：**

```
实时同步：不推荐（复杂度高）
- 每次文件变化都触发同步
- 通信频率高，性能开销大

批量同步：推荐（复杂度低）
- 用户手动触发（导入大量图片后）
- 定期自动同步（每天凌晨）
- 按需同步（用户请求）

优势：
- 减少通信频率
- 提高系统稳定性
- 降低维护成本
- 用户体验更好（批量处理更快）
```

#### 3.4 异常处理和恢复

**异常场景处理：**

```
场景1：Python进程崩溃
- Go检测到进程退出
- Go重新启动Python进程
- Go发送continue请求
- Python从进度文件恢复
- Python继续处理未完成的文件

场景2：索引文件损坏
- Python检测到索引加载失败
- Python返回错误信息
- Go发送rebuild请求
- Python重建索引

场景3：进度文件损坏
- Python检测到进度文件格式错误
- Python返回错误信息
- Go重新生成进度文件
- Go发送start请求

场景4：处理失败过多
- Python检测到failed_files超过阈值（如10%）
- Python返回警告信息
- Go决定是否继续或停止
- Go记录失败文件，供用户查看
```

**数据一致性保证：**

```
Go端保证：
- SQLite数据库是唯一数据源
- 所有文件操作都先更新SQLite
- Go计算并维护file_hash
- process_file.json从SQLite生成
- 确保数据一致性

Python端保证：
- 通过file_hash对比识别变化
- 只处理file_hash变化的文件
- 每处理batch个文件保存进度
- 确保索引与元数据一致

恢复机制：
- Python进程崩溃：从进度文件恢复
- 索引损坏：重建索引
- 元数据损坏：重建索引
- 进度文件损坏：重新生成
```

### 4. FAISS索引管理设计

#### 4.1 索引初始化和构建（基于进度文件）

**首次构建流程：**

```
场景：用户首次使用AI功能，FAISS索引不存在

流程：
1. Go扫描图片目录，整理所有待处理文件
2. Go将待处理文件列表写入进度文件（process_file.json）
3. Go触发AI FAISS索引构建（发送start请求）
4. Python读取进度文件，获取待处理文件列表
5. Python批量处理文件（batch_size控制）
6. Python处理一定量文件后，更新进度文件
7. Python构建FAISS索引并保存
8. Go可以随时触发停止或继续请求

进度文件设计：
位置：ai-engine/src/config/process_file.json
格式：
{
  "version": "1.0.0",
  "batch": 10,  // 每处理10个文件更新一次进度文件
  "files": [
    {
      "record_id": 5001,
      "file_path": "/path/to/image.jpg",
      "file_hash": "abc123",
      "status": "pending"  // "pending", "processing", "completed", "failed"
    },
    {
      "record_id": 5002,
      "file_path": "/path/to/image2.jpg",
      "file_hash": "def456",
      "status": "processing"
    },
    // ... 所有图片列表
  ],
  "start_time": "2024-01-01 10:00:00",
  "last_update_time": "2024-01-01 12:00:00"
}

状态说明：
- pending: 待处理（未构建索引）
- processing: 正在处理
- completed: 已完成（已构建索引）
- failed: 处理失败

重要说明：
- Go生成process_file.json时，所有文件status统一设为"pending"
- Python通过元数据对比自动识别文件类型（新增、删除、更新）
- 统计信息（total_files, processed_files等）实时计算，避免数据不一致
```

**Go触发请求接口：**

```
Python处理流程：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "index_init",
     "data": {
       "cmd": "start",  // "start", "stop", "continue", "get_progress"
       "rebuild": false  // 是否重建索引
     }
   }

2. 处理请求并返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success"
   }
```

**索引配置：**

```
FAISS索引配置（中规模1万-10万张）：
- 索引类型: IndexIDMap2(IndexIVFFlat)
- feature_dim: 1024（Chinese-CLIP ViT-H-14）
- nlist: 100-300（根据图片数量调整）
- nprobe: 10（搜索时查询10个聚类）
- 训练样本: 1000-5000张（用于聚类训练）

进度文件配置：
- 文件格式: JSON
- 存储内容: version, batch, files列表, 时间信息
- 文件大小: ~1-2MB（10万张图片）
- 更新频率: 每处理batch个文件更新一次
- 统计信息: 实时计算（total_files, processed_files等），避免数据不一致

元数据文件配置：
- 文件格式: JSON
- 存储内容: vector_id, record_id, deleted
- 文件大小: ~1-2MB（10万张图片）
```

#### 4.2 索引持久化和加载

**索引文件管理：**

```
文件列表：
- faiss_index.bin: FAISS索引文件（~400MB，10万张）
- ai_metadata.json: 元数据文件（~1-2MB）
- index_config.json: 索引配置文件
- process_file.json: 进度文件（~1-2MB）

存储位置：
- 默认: ai-engine/src/data/index/
- 进度文件: ai-engine/src/config/process_file.json
- 可配置: 通过环境变量指定

加载流程：
1. Python启动时检查索引文件是否存在
2. 如果存在，加载索引和元数据
3. 如果不存在，等待Go调用start请求
4. 加载时间: ~1-2秒（10万张图片）
```

#### 4.3 索引重建和维护

**手动重建API：**

```
Python处理流程：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "index_init",
     "data": {
       "cmd": "start",
       "rebuild": true
     }
   }

2. 处理请求并返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success"
   }

重建流程：
1. Go重新生成进度文件（所有文件状态设为idle）
2. Go发送start请求（rebuild=true）
3. Python删除现有索引
4. Python读取进度文件
5. Python重新处理所有文件
6. Python构建新索引
7. Python保存索引和进度文件
```


**进度监控：**

```
Go监控进度：
1. 调用get_progress请求
2. 获取当前处理进度
3. 显示进度条给用户
4. 检测是否完成或失败

进度响应示例：
{
  "code": 0,
  "status": "success",
  "data": {
    "total_files": 10000,
    "processed_files": 5000,
    "failed_files": 10,
    "status": "processing",  // "pending", "processing", "completed", "stopped"
    "progress_percent": 50.0,
    "estimated_time_remaining": "2 hours",
    "last_update_time": "2024-01-01 12:00:00"
  }
}
```

**异常处理：**

```
异常场景：
1. Python进程崩溃：
   - Go检测到进程退出
   - Go重新启动Python进程
   - Go发送continue请求
   - Python从进度文件恢复

2. 索引文件损坏：
   - Python检测到索引加载失败
   - Python返回错误信息
   - Go发送rebuild请求
   - Python重建索引

3. 进度文件损坏：
   - Python检测到进度文件格式错误
   - Python返回错误信息
   - Go重新生成进度文件
   - Go发送start请求

4. 处理失败过多：
   - Python检测到failed_files超过阈值
   - Python返回警告信息
   - Go决定是否继续或停止
```

### 5. 部署和运维设计

#### 5.1 部署架构

**部署方案对比：**

| 部署方案 | 优点 | 缺点 | 适用场景 |
|----------|------|------|----------|
| **方案1: 本地部署（推荐）** | 简单、无需额外服务器、stdio通信更快 | 占用相册主机资源 | 个人/家庭相册 |
| **方案2: 独立服务部署** | 资源隔离、故障不影响相册 | 需要额外服务器、HTTP通信开销 | 企业/团队相册 |
| **方案3: Docker容器部署** | 易于管理、环境一致 | 需要Docker环境、stdio通信受限 | 推荐：企业/团队相册 |

**推荐：本地部署（stdio通信）**

```
本地部署架构（stdio通信）：
┌─────────────────────────────────────────────────────────────┐
│  主机（运行Go相册主体）                                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Go相册主体程序                                            ││
│  │  - 监听端口: 8080                                         ││
│  │  - SQLite数据库                                           ││
│  │  - 图片文件存储                                            ││
│  │  - PythonProcess（管理Python子进程）                      ││
│  │    - stdin/stdout通信                                     ││
│  │    - stderr日志输出                                       ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Python AI子进程                                          ││
│  │  - FAISS索引文件                                          ││
│  │  - CLIP ONNX模型                                          ││
│  │  - 通过stdin接收请求                                      ││
│  │  - 通过stdout返回响应                                     ││
│  │  - 通过stderr输出日志                                     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

优势：
- 无需HTTP服务器，减少资源占用
- 进程常驻，避免每次启动Python的开销
- stdio通信更快，无HTTP协议开销
- 更适合桌面应用（无需端口管理）
- 更简单，易于部署和维护

启动流程：
1. Go启动时查找Python可执行文件
2. Go启动Python子进程（python main.py）
3. Python加载模型和索引
4. Go通过stdin发送ping请求验证通信
5. Python通过stdout返回响应
6. 通信建立成功，开始提供服务
```

**备选：Docker容器部署（HTTP通信）**

```
如果需要独立部署或团队协作，可使用Docker容器部署：

Docker部署架构：
┌─────────────────────────────────────────────────────────────┐
│  主机（运行Go相册主体）                                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Go相册主体程序                                            ││
│  │  - 监听端口: 8080                                         ││
│  │  - SQLite数据库                                           ││
│  │  - 图片文件存储                                            ││
│  │  - HTTP客户端（调用AI API）                                ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Docker容器（Python AI模块）                               ││
│  │  - 监听端口: 8000                                         ││
│  │  - FAISS索引文件                                          ││
│  │  - CLIP ONNX模型                                          ││
│  │  - HTTP API服务                                           ││
│  │  - 与Go通过localhost:8000通信                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

Dockerfile示例：
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py"]

docker-compose.yml示例：
version: '3'
services:
  ai-engine:
    build: ./ai-engine
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ../images:/images:ro  // 只读访问图片目录
    environment:
      - INDEX_DIR=/app/data/index
      - MODEL_DIR=/app/models
```

#### 5.2 配置管理

**Python AI模块配置：**

```
配置文件: config.yaml

process:
  mode: "stdio"  // stdio或http
  log_level: "INFO"
  
index:
  type: "IndexIDMap2_IndexIVFFlat"
  feature_dim: 1024
  nlist: 100
  nprobe: 10
  index_file: "data/index/faiss_index.bin"
  metadata_file: "data/index/ai_metadata.json"
  
model:
  name: "Chinese-CLIP-ViT-L-14"
  vision_model: "models/vision.onnx"
  text_model: "models/text.onnx"
  
sync:
  batch_size: 32
  max_concurrent_tasks: 3
  
logging:
  level: "INFO"
  stderr_output: true  // 输出到stderr
```

**Go相册主体配置：**

```
配置文件: config.yaml

ai_module:
  enabled: true
  python_path: "python"  // 或指定完整路径
  script_path: "ai-engine/src/main.py"
  timeout_seconds: 120
  retry_count: 3
  
sync:
  auto_sync: true  // 自动同步图片变化
  batch_threshold: 10  // 超过10张图片批量同步
```

#### 5.3 监控和日志

**监控指标：**

```
关键监控指标：
- Python进程状态（运行、崩溃、重启）
- 请求响应时间（搜索<5ms，其他<100ms）
- 索引大小（图片数量、索引文件大小）
- 内存占用（FAISS索引、模型加载）
- CPU使用率（特征提取、搜索）
- 任务队列长度（异步任务数量）

监控工具：
- Python: 通过stderr输出监控信息
- Go: 监控Python进程状态和响应时间
- 日志: stderr输出 + Go日志系统

监控函数接口：
Go调用接口：
GetAIStatus() (AIStatus, error)

Python处理流程：
1. 接收JSON请求：
   {
     "id": "1234567890",
     "action": "get_status",
     "data": {}
   }

2. 处理请求并返回JSON响应：
   {
     "id": "1234567890",
     "code": 0,
     "status": "success",
     "data": {
       "status": "healthy",
       "index_loaded": true,
       "total_images": 50000,
       "memory_usage_mb": 450,
       "uptime_seconds": 3600
     }
   }
```

**日志设计：**

```
日志级别：
- INFO: 正常操作（搜索、分类、同步等）
- WARNING: 异常情况（索引未加载、同步失败等）
- ERROR: 错误情况（特征提取失败、索引损坏等）
- DEBUG: 详细调试信息

日志输出方式：
- Python通过stderr输出日志（每行一个JSON对象）
- Go通过stderr pipe接收并处理日志
- Go日志系统统一管理所有日志

stderr日志格式：
{
  "timestamp": "2024-01-01T10:00:00Z",
  "level": "INFO",
  "module": "SearchEngine",
  "action": "search",
  "query": "风景",
  "result_count": 10,
  "process_time_ms": 3.5,
  "message": "Search completed successfully"
}

Go日志处理：
func (p *PythonProcess) readStderr(stderr io.Reader) {
    scanner := bufio.NewScanner(stderr)
    for scanner.Scan() {
        line := scanner.Text()
        if line != "" {
            log.Infof("ai-log: %s", line)
        }
    }
}
```

#### 5.4 错误处理和恢复

**常见错误场景：**

| 错误场景 | 处理策略 | 恢复方案 |
|----------|----------|----------|
| **Python进程崩溃** | Go检测到进程退出，自动重启 | 自动重启Python进程 |
| **索引文件损坏** | Python检测并通过stderr报警 | 从备份恢复或重建索引 |
| **模型加载失败** | Python返回错误响应 | 检查模型文件，重新下载 |
| **特征提取失败** | 记录错误，跳过图片 | 提供手动重试接口 |
| **同步失败** | 自动重试（最多3次） | 提供手动同步接口 |
| **内存不足** | 降级处理（减少batch_size） | 增加内存或优化索引 |

**错误响应格式：**

```
{
  "code": 1,
  "status": "INDEX_NOT_LOADED",
  "data": {
    "error_message": "FAISS index not loaded, please initialize first",
    "suggestion": "Call /api/v1/index/init to build index"
  },
  "timestamp": "2024-01-01T10:00:00Z"
}
```

**自动恢复机制：**

```
恢复策略：
1. 索引文件损坏：
   - 检测: 启动时验证索引文件完整性
   - 恢复: 从最近备份恢复
   - 备选: 调用重建API
   
2. 元数据不一致：
   - 检测: 定期校验元数据与SQLite一致性
   - 恢复: 从SQLite重建元数据
   
3. 模型加载失败：
   - 检测: 启动时验证模型文件
   - 恢复: 自动下载模型文件
   
4. 内存不足：
   - 检测: 监控内存使用率
   - 恢复: 降级处理，减少batch_size
```

### 6. 性能优化建议

#### 6.1 搜索性能优化

**优化策略：**

```
策略1: 文本特征缓存
- 相同查询文本的特征缓存（LRU缓存）
- 缓存大小: 100个最近查询
- 缓存命中率: >80%（用户重复搜索）

策略2: 批量搜索
- 支持一次搜索多个查询文本
- 批量提取文本特征
- 批量FAISS搜索
- 性能提升: 2-3倍

策略3: 过滤条件优化
- 按时间范围过滤：使用SQLite时间索引
- 按相册过滤：使用SQLite相册索引
- 先过滤再搜索，减少搜索范围

策略4: 索引参数调优
- 根据实际搜索频率调整nprobe
- 高频搜索: nprobe=5（快速）
- 高精度搜索: nprobe=15（精确）
```

#### 6.2 特征提取优化

**优化策略：**

```
策略1: 批量编码
- batch_size=32（平衡速度和内存）
- 批量预处理图片
- 批量ONNX推理
- 性能提升: 4倍

策略2: GPU加速（可选）
- 使用CUDA版本的ONNX Runtime
- GPU批量推理
- 性能提升: 10-20倍
- 适用场景: 有GPU环境

策略3: 图片预处理优化
- 图片尺寸调整（保持CLIP输入尺寸）
- 图片格式转换（统一格式）
- 预处理缓存（相同图片复用）

策略4: 异步特征提取
- 后台异步提取特征
- 不阻塞主线程
- 适合批量导入场景
```

#### 6.3 内存优化

**优化策略：**

```
策略1: 索引压缩（大规模场景）
- 使用IndexIVFPQ压缩索引
- 内存节省: 90%
- 精度损失: 5-10%
- 适用场景: >10万张图片

策略2: 特征缓存限制
- 限制文本特征缓存大小
- 限制图片预处理缓存大小
- 使用LRU缓存策略

策略3: 批量处理优化
- 动态调整batch_size（根据内存）
- 内存不足时减少batch_size
- 内存充足时增加batch_size

策略4: 模型优化
- 使用ONNX模型（比PyTorch模型小）
- 模型量化（可选，减少模型大小）
- 模型共享（多进程共享模型）
```

### 7. 实施路线图

#### 7.1 分阶段实施计划

**阶段1: 基础功能实现（2-4周）**

```
目标：实现核心搜索功能
任务：
1. 实现FAISS索引管理（IndexIDMap2(IndexIVFFlat))
5. 实现数据同步功能（sync action）
6. 实现索引初始化功能（index_init action）
```

**阶段2: 扩展功能实现（2-4周）**

```
目标：实现其他AI功能
任务：
2. 实现标签生成功能（generate_tags action）
3. 实现图片描述功能（generate_description action）
4. 实现相似图片功能（find_similar action）

验证点：
- 各功能响应时间合理
- 分类准确率 >85%
- 标签生成准确率 >80%
```

**阶段3: 性能优化和部署（1-2周）**

```
目标：优化性能和部署
任务：
1. 实现批量编码优化
2. 实现文本特征缓存
3. 实现本地部署方案（stdio通信）
4. 实现监控和日志系统（stderr输出）
5. 实现错误处理和恢复机制（进程重启）

验证点：
- 批量编码速度提升4倍
- 缓存命中率 >80%
- 本地部署成功
- 监控指标正常
```

**阶段4: 生产部署和测试（1-2周）**

```
目标：生产环境部署和测试
任务：
1. 生产环境部署（本地部署）
2. 性能测试（搜索、分类、同步等）
3. 稳定性测试（长时间运行）
4. 用户测试（实际使用反馈）
5. 文档完善（API文档、部署文档）

验证点：
- 生产环境稳定运行
- 性能满足要求
- 用户反馈良好
```

#### 7.2 技术风险评估

**风险1: FAISS索引性能**

```
风险描述：
- 索引构建时间可能较长（10万张图片）
- 搜索性能可能不达预期

应对措施：
- 使用批量编码加速构建
- 使用IndexIDMap2支持删除（避免重建）
- 性能测试验证（提前测试）
- 降级方案：使用IndexFlatIP（暴力搜索）
```

**风险2: Go-Python通信**

```
风险描述：
- stdio通信可能存在管道阻塞
- Python进程崩溃可能导致通信中断

应对措施：
- 使用缓冲区避免阻塞（bufio.Scanner）
- 实现进程监控和自动重启
- 实现超时处理（120秒）
- 实现错误日志记录（stderr输出）
```

**风险3: 数据一致性**

```
风险描述：
- SQLite和元数据可能不一致
- 同步失败可能导致数据丢失

应对措施：
- 实现定期校验机制
- 实现自动恢复机制
- 实现手动重建接口
- 实现错误日志记录
```

**风险4: 内存和资源**

```
风险描述：
- FAISS索引占用内存较大
- 特征提取占用CPU较多

应对措施：
- 监控内存和CPU使用率
- 实现降级处理（减少batch_size）
- 实现资源限制（Docker资源限制）
- 优化索引（使用PQ压缩）
```

### 8. 总结和建议

**最终推荐方案：**

```
AI核心功能推荐方案（中规模1万-10万张）：

索引方案：
✅ IndexIDMap2(IndexIVFFlat)
- 支持删除操作（用户体验好）
- 支持自定义ID（数据同步简单）
- 性能开销可接受（~5-10%）
- 搜索时间：~2-5ms

元数据方案：
✅ 最小化元数据文件（ai_metadata.json）
- 仅存储：vector_id, record_id, file_hash
- 与SQLite解耦（维护简单）
- 数据冗余最小化

同步方案：
✅ 事件驱动同步（stdio通信）
- 实时性好（立即同步）
- 实现简单（stdin/stdout）
- 性能更高（无HTTP开销）
- 更适合桌面应用

部署方案：
✅ 本地部署（stdio通信）
- 无需HTTP服务器（资源占用少）
- 进程常驻（避免启动开销）
- 更简单（易于部署和维护）
- 更适合桌面应用

备选方案：
✅ Docker容器部署（HTTP通信）
- 环境一致（易于管理）
- 资源隔离（故障不影响相册）
- 易于扩展（水平扩展）
- 适合企业/团队相册

性能优化：
✅ 批量编码 + 文本特征缓存 + 过滤优化
- 构建速度提升4倍
- 搜索速度提升2-3倍
- 内存优化可控
```

**关键成功因素：**

```
1. 数据同步机制完善
   - 确保SQLite和元数据一致性
   - 实现自动恢复机制
   - 提供手动重建接口

2. 性能满足用户需求
   - 搜索响应时间 <5ms
   - 分类准确率 >85%
   - 批量操作高效

3. 系统稳定可靠
   - 错误处理完善
   - 自动恢复机制
   - 监控和日志完善

4. 维护简单
   - 文档完善
   - API设计清晰
   - 部署简单
```

**下一步行动：**

```
建议实施顺序：
1. 先实现搜索功能（核心功能）
2. 再实现数据同步机制（基础保障）
3. 然后实现其他AI功能（扩展功能）
4. 最后优化性能和部署（生产优化）

关键里程碑：
- 第2周：搜索功能可用
- 第4周：所有AI功能可用
- 第6周：性能优化完成
- 第8周：生产部署完成
```

这个设计方案为你的AI相册项目提供了完整的实施指导，涵盖了架构设计、功能实现、数据同步、部署运维等所有关键方面。根据这个方案，你可以逐步实施AI核心功能，确保系统稳定、性能优良、易于维护。


