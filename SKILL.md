---
name: A3.X dev_视频转录工具
description: |
  通过 MCP 协议访问视频转录和课件 RAG 系统，支持视频内容检索、课件语义搜索、
  以及视频与课件内容的交叉分析。
  
  使用场景：
  - 用户询问视频课程中的某个概念/术语
  - 用户想了解某个时间点的讲解内容
  - 需要对比视频讲解和课件内容的差异
  - 查找包含代码示例的片段
  
  重要：
  1. 课程名称必须先确认！用户可能输入"finetune"，实际目录是"A4.2_fine_tuning"。
  2. 课程名称大小写敏感！必须使用 list_available_courses 返回的准确名称（如 A3.11_autogen，不是 a3.11_autogen）。
  
  工作流程：
  1. 调用 list_available_courses 获取所有课程
  2. 使用 difflib 模糊匹配用户输入的课程名
  3. 验证课程数据就绪状态
  4. 根据需求调用相应 MCP 工具
license: MIT
compatibility: |
  需要 MCP Server 已启动，且课程数据已构建：
  - 视频索引：.cache/{course_name}/facts/segments.jsonl
  - 课件 RAG：vector_db/course_materials/{course_name}/
---

# Video Transcription RAG MCP Tools

## 何时使用此技能

当用户需要以下功能时，使用此技能：

1. **查询视频课程内容**：
   - "finetune 课程里讲了什么？"
   - "第15分钟讲了什么？"
   - "哪里讲了 LoRA？"

2. **搜索课件内容**：
   - "课件里 LoRA 的定义是什么？"
   - "找一下关于微调的文档"

3. **对比分析**：
   - "视频讲的和课件一致吗？"
   - "代码演示在哪里？"

## 核心规则

### 规则1：课程名称必须先确认（大小写敏感！）

**问题**：用户输入的课程名可能与实际目录名不一致

| 用户输入 | 实际目录名 |
|---------|-----------|
| finetune | A4.2_fine_tuning |
| finetuning | A4.2_fine_tuning |
| prompt | prompt_engineering |
| rag | rag_system |

**重要：课程名称大小写敏感**

必须使用 `list_available_courses` 返回的**准确名称**（包括大小写）：
- 正确: `A3.11_autogen`
- 错误: `a3.11_autogen` (小写a开头)
- 错误: `A3.11_AutoGen` (大小写不一致)

**正确做法**：
1. 调用 `list_available_courses` 获取所有课程
2. 用 Python `difflib` 模糊匹配
3. **必须**使用返回的准确课程名（包括大小写）再调用其他 MCP 工具

### 规则2：先验证数据存在

调用 MCP 工具前，检查：
- `.cache/{course_name}/facts/segments.jsonl`（视频索引）
- `vector_db/course_materials/{course_name}/chroma.sqlite3`（课件 RAG）

## 标准操作流程

### 步骤1：获取可用课程列表

**首先调用 `list_available_courses` 工具**，它会返回所有课程及其数据就绪状态：

```mcp
list_available_courses
{}
```

**返回示例**：
```json
{
  "courses": [
    {
      "name": "A4.2_fine_tuning",
      "has_video_index": true,
      "has_material_rag": true
    },
    {
      "name": "prompt_engineering",
      "has_video_index": true,
      "has_material_rag": false
    }
  ]
}
```

### 步骤2：匹配课程名称

使用 Python 进行模糊匹配：

```python
import difflib

# 用户输入
user_input = 'finetune'

# 从 list_available_courses 返回中提取课程名
available_names = [c['name'] for c in result['courses']]

# 模糊匹配
matches = difflib.get_close_matches(user_input, available_names, n=1, cutoff=0.5)
course_name = matches[0] if matches else available_names[0]

print(f"用户输入 '{user_input}' 匹配到课程: {course_name}")
```

### 步骤3：验证数据就绪

从 `list_available_courses` 的结果中检查：

```python
# 找到匹配的课程信息
course_info = next(c for c in result['courses'] if c['name'] == course_name)

video_ready = course_info['has_video_index']
rag_ready = course_info['has_material_rag']

print(f"视频索引: {'✓' if video_ready else '✗'}")
print(f"课件 RAG: {'✓' if rag_ready else '✗'}")

if not (video_ready or rag_ready):
    raise Exception(f"课程 '{course_name}' 数据未构建")
```

### 步骤4：调用 MCP 工具

根据用户需求选择相应工具（见下文"工具清单"）。

## 工具清单

### 系统工具

#### list_available_courses

**描述**：获取所有可用课程列表及其数据就绪状态。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| （无） | - | - | 此工具无需参数 |

**返回Schema**：
```json
{
  "courses": [
    {
      "name": "string",           // 课程名称
      "has_video_index": "boolean",  // 视频索引是否就绪
      "has_material_rag": "boolean"  // 课件RAG是否就绪
    }
  ]
}
```

**MCP调用示例**：
```mcp
list_available_courses
{}
```

**使用场景**：
- **任何其他工具调用前必须先调用此工具**
- 获取准确的课程名称列表
- 检查课程数据是否已构建

**重要**：此工具必须在所有其他工具之前调用，用于确认课程名称和数据状态。

---

### 视频内容工具

#### video_get_course_structure

**描述**：获取课程的整体结构导航。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名（必须准确） |

**返回Schema**：
```json
[
  {
    "id": "string",           // Segment ID，如 "seg_001"
    "time_start": "number",   // 开始时间（秒）
    "time_end": "number",     // 结束时间（秒）
    "keywords": ["string"],   // 关键词列表
    "features": {             // 内容特征
      "has_code": "boolean",
      "has_command": "boolean",
      "has_url": "boolean"
    }
  }
]
```

**MCP调用示例**：
```mcp
video_get_course_structure
{
  "course_name": "A4.2_fine_tuning"
}
```

**使用场景**：首次访问课程，了解章节划分

---

#### video_get_segment_by_time

**描述**：根据时间戳定位 Segment。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |
| timestamp | number | 是 | 时间戳（秒） |

**返回Schema**：
```json
{
  "id": "string",           // Segment ID
  "time_start": "number",   // 开始时间（秒）
  "time_end": "number",     // 结束时间（秒）
  "text": "string",         // 完整的转录文本
  "keywords": ["string"],   // 关键词列表
  "features": {             // 内容特征
    "has_code": "boolean",
    "has_command": "boolean",
    "has_url": "boolean"
  }
}
```

**MCP调用示例**：
```mcp
video_get_segment_by_time
{
  "course_name": "A4.2_fine_tuning",
  "timestamp": 300.0
}
```

**使用场景**：用户提到具体时间，如"第5分钟"

**注意**：时间需转换为秒，5分钟 = 300.0

---

#### video_get_segments_by_concept

**描述**：根据概念关键词查找 Segments。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |
| concept | string | 是 | 概念关键词 |

**返回Schema**：
```json
[
  {
    "id": "string",           // Segment ID
    "time_start": "number",   // 开始时间（秒）
    "time_end": "number",     // 结束时间（秒）
    "keywords": ["string"]    // 关键词列表
  }
]
```

**MCP调用示例**：
```mcp
video_get_segments_by_concept
{
  "course_name": "A4.2_fine_tuning",
  "concept": "LoRA"
}
```

**使用场景**：用户询问某个概念/术语

---

#### video_get_segments_by_type

**描述**：按内容类型筛选 Segments。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |
| content_type | string | 是 | 类型：has_code / has_command / has_url |

**返回Schema**：
```json
[
  {
    "id": "string",           // Segment ID
    "time_start": "number",   // 开始时间（秒）
    "time_end": "number",     // 结束时间（秒）
    "keywords": ["string"]    // 关键词列表
  }
]
```

**MCP调用示例**：
```mcp
video_get_segments_by_type
{
  "course_name": "A4.2_fine_tuning",
  "content_type": "has_code"
}
```

**使用场景**：找包含代码、命令行或 URL 的片段

---

#### video_get_segment_content

**描述**：获取单个 Segment 完整内容。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |
| segment_id | string | 是 | Segment ID，如 "seg_003" |

**返回Schema**：
```json
{
  "id": "string",           // Segment ID
  "time_start": "number",   // 开始时间（秒）
  "time_end": "number",     // 结束时间（秒）
  "text": "string",         // 完整的转录文本
  "keywords": ["string"],   // 关键词列表
  "features": {             // 内容特征
    "has_code": "boolean",
    "has_command": "boolean",
    "has_url": "boolean"
  }
}
```

**MCP调用示例**：
```mcp
video_get_segment_content
{
  "course_name": "A4.2_fine_tuning",
  "segment_id": "seg_003"
}
```

**使用场景**：已知 segment_id，需要获取转录文本

---

#### video_get_multiple_segments

**描述**：批量获取多个 Segments。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |
| segment_ids | array | 是 | Segment ID 列表 |

**返回Schema**：
```json
[
  {
    "id": "string",
    "time_start": "number",
    "time_end": "number",
    "text": "string",
    "keywords": ["string"],
    "features": {
      "has_code": "boolean",
      "has_command": "boolean",
      "has_url": "boolean"
    }
  }
]
```

**MCP调用示例**：
```mcp
video_get_multiple_segments
{
  "course_name": "A4.2_fine_tuning",
  "segment_ids": ["seg_003", "seg_004", "seg_005"]
}
```

**使用场景**：同时获取多个 segment，或获取相邻片段作为上下文

**优势**：比多次调用 video_get_segment_content 更高效

---

#### video_get_raw_text_range

**描述**：获取时间范围内的原始文本拼接。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |
| start_sec | number | 是 | 开始时间（秒） |
| end_sec | number | 是 | 结束时间（秒） |

**返回Schema**：
```json
{
  "text": "string",           // 拼接后的文本
  "segment_ids": ["string"]   // 涉及的 Segment ID 列表
}
```

**MCP调用示例**：
```mcp
video_get_raw_text_range
{
  "course_name": "A4.2_fine_tuning",
  "start_sec": 240.0,
  "end_sec": 480.0
}
```

**使用场景**：只需要纯文本，不需要 segment 结构

---

#### video_get_segment_metadata

**描述**：获取 Segment 轻量级元数据。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |
| segment_id | string | 是 | Segment ID |

**返回Schema**：
```json
{
  "id": "string",           // Segment ID
  "time_start": "number",   // 开始时间（秒）
  "time_end": "number",     // 结束时间（秒）
  "duration": "number",     // 持续时间（秒）
  "word_count": "number",   // 字数
  "features": {             // 内容特征
    "has_code": "boolean",
    "has_command": "boolean",
    "has_url": "boolean"
  }
}
```

**MCP调用示例**：
```mcp
video_get_segment_metadata
{
  "course_name": "A4.2_fine_tuning",
  "segment_id": "seg_003"
}
```

**使用场景**：只需要元信息，不需要原文

---

#### video_get_course_stats

**描述**：获取课程统计信息。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |

**返回Schema**：
```json
{
  "total_segments": "number",           // Segment 总数
  "total_duration": "number",           // 总时长（秒）
  "keyword_distribution": {             // 关键词分布
    "string": "number"
  }
}
```

**MCP调用示例**：
```mcp
video_get_course_stats
{
  "course_name": "A4.2_fine_tuning"
}
```

**使用场景**：了解课程整体情况，验证时间戳是否有效

---

### 课件内容工具

#### material_search

**描述**：语义搜索课件内容。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |
| query | string | 是 | 搜索查询 |
| top_k | number | 否 | 返回结果数，默认3 |

**返回Schema**：
```json
{
  "query": "string",              // 查询内容
  "results": [
    {
      "id": "string",             // Chunk ID
      "text": "string",           // 文本内容
      "source_file": "string",    // 源文件
      "headings": ["string"],     // 标题层级
      "similarity": "number"      // 相似度分数
    }
  ]
}
```

**MCP调用示例**：
```mcp
material_search
{
  "course_name": "A4.2_fine_tuning",
  "query": "LoRA 原理",
  "top_k": 3
}
```

**使用场景**：查找课件中与查询相关的知识点

---

#### material_batch_search

**描述**：批量语义搜索课件。

**参数**：

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| course_name | string | 是 | 课程名 |
| queries | array | 是 | 查询列表 |
| top_k | number | 否 | 每个查询返回结果数，默认3 |

**返回Schema**：
```json
{
  "results": {
    "query1": [
      {
        "id": "string",
        "text": "string",
        "source_file": "string",
        "headings": ["string"],
        "similarity": "number"
      }
    ],
    "query2": [...]
  }
}
```

**MCP调用示例**：
```mcp
material_batch_search
{
  "course_name": "A4.2_fine_tuning",
  "queries": ["LoRA", "QLoRA", "Adapter"],
  "top_k": 3
}
```

**使用场景**：同时搜索多个相关概念

## 典型场景示例

### 场景A：回答概念相关问题

**用户**："finetune 课程里 LoRA 是什么？"

**执行流程**：

1. **获取课程列表**
   ```mcp
   list_available_courses
   {}
   ```

2. **匹配课程名**
   ```python
   import difflib
   
   user_input = 'finetune'
   available_names = [c['name'] for c in result['courses']]
   matches = difflib.get_close_matches(user_input, available_names, n=1, cutoff=0.5)
   course_name = matches[0] if matches else available_names[0]
   # 结果: 'A4.2_fine_tuning'
   ```

3. **验证数据就绪**
   ```python
   course_info = next(c for c in result['courses'] if c['name'] == course_name)
   video_ready = course_info['has_video_index']
   rag_ready = course_info['has_material_rag']
   ```

4. **搜索课件**
   ```mcp
   material_search
   {
     "course_name": "A4.2_fine_tuning",
     "query": "LoRA 是什么",
     "top_k": 3
   }
   ```

5. **搜索视频**
   ```mcp
   video_get_segments_by_concept
   {
     "course_name": "A4.2_fine_tuning",
     "concept": "LoRA"
   }
   ```

6. **获取视频内容**
   ```mcp
   video_get_multiple_segments
   {
     "course_name": "A4.2_fine_tuning",
     "segment_ids": ["seg_005", "seg_006"]
   }
   ```

7. **综合回答**
   - 结合课件的标准定义
   - 结合视频的讲解内容
   - 给出完整回答

---

### 场景B：定位特定时间

**用户**："第15分钟讲了什么？"

**执行流程**：

1. **获取课程列表**
   ```mcp
   list_available_courses
   {}
   ```

2. **匹配课程名**
   ```python
   # 模糊匹配用户输入的课程名
   ```

3. **时间转换**
   ```python
   timestamp = 15 * 60  # 900秒
   ```

4. **验证时间有效**
   ```mcp
   video_get_course_stats
   {
     "course_name": "A4.2_fine_tuning"
   }
   ```
   检查 `total_duration >= 900`

5. **定位 segment**
   ```mcp
   video_get_segment_by_time
   {
     "course_name": "A4.2_fine_tuning",
     "timestamp": 900
   }
   ```

6. **获取上下文**（前后各一个）
   ```python
   current_id = "seg_005"
   idx = int(current_id.replace("seg_", ""))
   neighbor_ids = [f"seg_{idx-1:03d}", current_id, f"seg_{idx+1:03d}"]
   ```
   ```mcp
   video_get_multiple_segments
   {
     "course_name": "A4.2_fine_tuning",
     "segment_ids": ["seg_004", "seg_005", "seg_006"]
   }
   ```

7. **回答**

---

### 场景C：找代码示例

**用户**："finetune 课程有哪些代码演示？"

**执行流程**：

1. **获取课程列表**
   ```mcp
   list_available_courses
   {}
   ```

2. **匹配课程名**
   ```python
   # 模糊匹配
   ```

3. **筛选代码 segments**
   ```mcp
   video_get_segments_by_type
   {
     "course_name": "A4.2_fine_tuning",
     "content_type": "has_code"
   }
   ```

4. **获取内容**
   ```mcp
   video_get_multiple_segments
   {
     "course_name": "A4.2_fine_tuning",
     "segment_ids": ["seg_008", "seg_012", "seg_015"]
   }
   ```

5. **总结**

---

### 场景D：对比视频和课件

**用户**："视频讲的 LoRA 和课件一致吗？"

**执行流程**：

1. **获取课程列表**
   ```mcp
   list_available_courses
   {}
   ```

2. **匹配课程名**
   ```python
   # 模糊匹配
   ```

3. **获取课件内容**
   ```mcp
   material_search
   {
     "course_name": "A4.2_fine_tuning",
     "query": "LoRA 原理",
     "top_k": 3
   }
   ```

4. **获取视频内容**
   ```mcp
   video_get_segments_by_concept
   {
     "course_name": "A4.2_fine_tuning",
     "concept": "LoRA"
   }
   ```
   ```mcp
   video_get_multiple_segments
   {
     "course_name": "A4.2_fine_tuning",
     "segment_ids": ["seg_005", "seg_006", "seg_007"]
   }
   ```

5. **分析对比**
   - 提取课件关键点
   - 提取视频讲解要点
   - 对比一致性和差异

## 错误处理

### 常见错误及解决方案

#### 错误1：课程不存在

**症状**：`list_available_courses` 返回的课程列表中没有匹配项

**处理代码**：
```python
import difflib

def resolve_course_name(user_input, available_courses):
    """解析用户输入的课程名"""
    available_names = [c['name'] for c in available_courses]
    
    # 尝试模糊匹配
    matches = difflib.get_close_matches(user_input, available_names, n=3, cutoff=0.4)
    
    if matches:
        return matches[0]
    else:
        # 无匹配时返回可用课程列表
        raise ValueError(
            f"未找到课程 '{user_input}'。\n"
            f"可用课程: {', '.join(available_names)}"
        )
```

---

#### 错误2：数据未构建

**症状**：视频索引或课件 RAG 文件不存在

**处理代码**：
```python
def check_course_data(course_name, course_list):
    """检查课程数据就绪状态"""
    course_info = next(
        (c for c in course_list if c['name'] == course_name),
        None
    )
    
    if not course_info:
        raise ValueError(f"课程 '{course_name}' 不存在")
    
    if not course_info['has_video_index'] and not course_info['has_material_rag']:
        raise RuntimeError(
            f"课程 '{course_name}' 数据未就绪。\n"
            f"请先执行构建流程：\n"
            f"  python -m src.pipeline build-course-rag --course {course_name}\n"
            f"  python -m src.pipeline process --course {course_name}"
        )
    
    return course_info
```

---

#### 错误3：Segment ID 不存在

**症状**：`video_get_segment_content` 返回 null 或空结果

**处理代码**：
```python
def get_valid_segment_ids(course_name):
    """获取有效的 Segment ID 列表"""
    # 先获取课程结构
    structure = video_get_course_structure(course_name)
    return [seg['id'] for seg in structure]

def validate_segment_id(segment_id, valid_ids):
    """验证 Segment ID 是否有效"""
    if segment_id not in valid_ids:
        raise ValueError(
            f"Segment ID '{segment_id}' 不存在。\n"
            f"有效的 ID 格式: seg_001, seg_002, ...\n"
            f"可用 ID: {', '.join(valid_ids[:5])}..."
        )
```

---

#### 错误4：时间戳超出范围

**症状**：`video_get_segment_by_time` 返回 null

**处理代码**：
```python
def validate_timestamp(course_name, timestamp):
    """验证时间戳是否有效"""
    # 获取课程统计
    stats = video_get_course_stats(course_name)
    total_duration = stats['total_duration']
    
    if timestamp < 0:
        raise ValueError(f"时间戳不能为负数: {timestamp}")
    
    if timestamp > total_duration:
        raise ValueError(
            f"时间戳 {timestamp} 超出课程范围。\n"
            f"课程总时长: {total_duration} 秒 ({total_duration/60:.1f} 分钟)"
        )
    
    return True

# 使用示例
timestamp = 15 * 60  # 15分钟
validate_timestamp("A4.2_fine_tuning", timestamp)
```

---

#### 错误5：模糊匹配失败

**症状**：用户输入与任何课程名都不匹配

**处理代码**：
```python
import difflib

def fuzzy_match_course(user_input, available_names, cutoff=0.4):
    """模糊匹配课程名"""
    matches = difflib.get_close_matches(user_input, available_names, n=3, cutoff=cutoff)
    
    if not matches:
        # 计算所有相似度
        similarities = [
            (name, difflib.SequenceMatcher(None, user_input, name).ratio())
            for name in available_names
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        suggestions = [f"{name} ({ratio:.0%})" for name, ratio in similarities[:3]]
        
        raise ValueError(
            f"无法匹配课程 '{user_input}'。\n"
            f"您是否想找: {', '.join(suggestions)}?"
        )
    
    return matches[0]
```

---

### 完整错误处理示例

```python
import difflib

def query_course_content(user_input_course, query_type, **kwargs):
    """
    查询课程内容的完整错误处理流程
    
    Args:
        user_input_course: 用户输入的课程名
        query_type: 查询类型 ('concept', 'time', 'code')
        **kwargs: 其他查询参数
    """
    try:
        # 步骤1: 获取课程列表
        course_list = list_available_courses()
        available_names = [c['name'] for c in course_list['courses']]
        
        # 步骤2: 模糊匹配课程名
        course_name = fuzzy_match_course(user_input_course, available_names)
        print(f"✓ 匹配到课程: {course_name}")
        
        # 步骤3: 验证数据就绪
        course_info = next(c for c in course_list['courses'] if c['name'] == course_name)
        
        if query_type == 'concept' and not course_info['has_material_rag']:
            raise RuntimeError(f"课程 '{course_name}' 的课件 RAG 未构建")
        
        if query_type in ['time', 'code'] and not course_info['has_video_index']:
            raise RuntimeError(f"课程 '{course_name}' 的视频索引未构建")
        
        print(f"✓ 数据就绪: 视频={course_info['has_video_index']}, RAG={course_info['has_material_rag']}")
        
        # 步骤4: 执行查询
        if query_type == 'concept':
            return material_search(course_name, kwargs.get('query', ''))
        elif query_type == 'time':
            validate_timestamp(course_name, kwargs.get('timestamp', 0))
            return video_get_segment_by_time(course_name, kwargs.get('timestamp', 0))
        elif query_type == 'code':
            return video_get_segments_by_type(course_name, 'has_code')
        
    except ValueError as e:
        print(f"输入错误: {e}")
    except RuntimeError as e:
        print(f"数据错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")
```

## 注意事项

1. **课程名匹配**：永远不要直接使用用户输入，必须先列出再匹配
2. **数据检查**：调用 MCP 工具前验证数据文件存在
3. **时间单位**：所有时间参数都是秒（float）
4. **segment_id 格式**：`seg_001`, `seg_002` 等三位数字
5. **批量接口优先**：获取多个 segment 时，用 `video_get_multiple_segments`
6. **模糊匹配阈值**：`difflib.get_close_matches` cutoff 建议 0.5-0.6
7. **错误处理**：始终包装 MCP 调用，处理可能的异常情况
