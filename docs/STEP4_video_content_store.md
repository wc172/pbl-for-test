# STEP 4: 内容存储与检索模块

> **模块4**: 构建视频内容的结构化存储与原子检索能力，为上层MCP服务提供内部工具实现。本模块不直接对外暴露，仅通过模块5的MCP服务被调用。
> 
> 依赖：步骤3（LLM校对）

---

## 设计原则

- **原子操作**：每个工具只做一件事，无业务组合逻辑
- **事实存储**：只存储和返回原始事实（segment文本、时间、关键词），不做LLM分析
- **零推理**：不重排序、不总结、不判断，这些交给外部Kimi Agent

---

## 输入

```yaml
input:
  srt_corrected: "courses/{course_name}/transcript_corrected.srt"
    source: "模块3 LLM校对输出"
  
  course_outline: "courses/{course_name}/materials/"
    source: "课件目录（可选，用于语义边界提示）"
```

---

## 处理流程（构建阶段）

```yaml
build_pipeline:
  step_1_parse_srt:
    description: "解析SRT为内部格式（复用现有SRTEntry）"
    output: "List[SRTEntry]"
    
  step_2_semantic_segmentation:
    description: "语义边界检测切分（以Entry为原子单位）"
    strategy:
      primary: "SeqModel边界检测（可选）"
      fallback: "规则边界检测（过渡词/总结词）"
    constraints:
      min_duration: 60    # 最少60秒
      max_duration: 300   # 最多300秒
      min_entries: 3      # 最少3个SRT Entry
    output: "List[Segment]"
    key_constraint: "不切分SRT Entry内部，时间戳精确对应"
    
  step_3_feature_extraction:
    description: "提取Segment特征（规则-based，无LLM）"
    features:
      has_code: "正则匹配代码模式(def/class/import/print()等)"
      has_command: "正则匹配命令行($/>开头)"
      has_url: "正则提取URL (https?://...)"
      word_count: "文本字数统计"
      sentence_count: "句子数统计（基于标点）"
    output: "Segment.features"
    
  step_4_build_index:
    description: "构建多级索引"
    indexes:
      time_index: "每分钟标记点映射到Segment"
      keyword_index: "倒排索引（关键词->Segment IDs）"
      type_index: "特征索引（has_code等布尔特征）"
    output: "index.json"
    
  step_5_save_storage:
    description: "保存到JSON Lines格式"
    output:
      segments.jsonl: "每行一个Segment（原子单位，含完整text）"
      index.json: "结构化索引（时间/关键词/类型）"
      navigation_map.json: "课程导航图（元数据列表）"
```

---

## Segment原子单位结构

```yaml
Segment:
  id: "seg_0001"                 # 唯一标识
  time_start: 9.15               # 开始时间（秒）
  time_end: 76.05                # 结束时间（秒）
  duration: 66.9                 # 持续时长（秒）
  srt_start_idx: 1               # 对应原始SRT起始序号
  srt_end_idx: 8                 # 对应原始SRT结束序号
  text: "完整的转录文本..."       # 原始字幕拼接
  keywords: ["A3.11_autogen"]     # 提取的关键词
  features:
    has_code: false              # 是否包含代码
    has_command: false           # 是否包含命令
    has_url: []                  # 提取的URL列表
    word_count: 82               # 字数统计
    sentence_count: 3            # 句子数统计
```

---

## 输出存储结构

```
.cache/{course_name}/facts/
├── segments.jsonl              # 主存储：每行一个Segment
│   # {"id": "seg_0001", "time_start": 9.15, "time_end": 76.05, "text": "...", ...}
├── index.json                  # 索引：加速查询
│   # {"time": {...}, "keywords": {...}, "types": {...}}
└── navigation_map.json         # 导航图
    # {course_name, total_segments, segment_list, concept_timeline}
```

---

## 核心类设计

```python
# src/pipeline/video_store/interface.py

class VideoContentStore:
    """视频内容存储与检索 - 工具实现层"""
    
    def __init__(self, course_name: str)
    
    # === 定位层工具（4个）===
    def get_course_structure(self) -> List[SegmentMeta]
    """返回课程结构导航（所有Segment的元数据列表，无原文）"""
    
    def get_segment_by_time(self, timestamp: float) -> Optional[Segment]
    """时间戳定位：返回指定时间所在的Segment"""
    
    def get_segments_by_concept(self, concept: str) -> List[SegmentMeta]
    """概念定位：返回包含该关键词的所有Segment元数据"""
    
    def get_segments_by_type(self, content_type: str) -> List[SegmentMeta]
    """类型筛选：如has_code, has_command等"""
    
    # === 提取层工具（3个）===
    def get_segment_content(self, segment_id: str) -> SegmentContent
    """获取单个Segment完整内容（含原文）"""
    
    def get_multiple_segments(self, segment_ids: List[str]) -> List[Segment]
    """批量获取：性能优化，减少IO"""
    
    def get_raw_text_range(self, start_sec: float, end_sec: float) -> str
    """原始文本拼接：获取时间范围内的纯文本"""
    
    # === 元数据层工具（2个）===
    def get_segment_metadata(self, segment_id: str) -> SegmentMeta
    """轻量元数据：无原文，仅meta+features"""
    
    def get_course_stats(self) -> CourseStats
    """课程统计：总片段数、总时长、关键词分布等"""
    
    # === 全局层工具（1个）===
    def get_navigation_map(self) -> NavigationMap
    """全局导航图：完整的课程导航信息"""
```

---

## 快速开始

### 构建内容存储

```python
from src.pipeline.video_store import SegmentBuilder, VideoStorage

# 1. 构建Segments（从校正后的SRT）
builder = SegmentBuilder(
    "my_course",
    use_semantic_segmenter=True,  # 使用SeqModel进行语义分割
    min_duration=60,              # 最少60秒
    max_duration=300              # 最多300秒
)

from pathlib import Path
srt_path = Path(f"courses/my_course/transcript_corrected.srt")
segments = builder.build(srt_path)

# 2. 保存存储
storage = VideoStorage("my_course")
storage.save_segments(segments)
```

### 命令行使用

```bash
# 执行步骤4（构建视频内容存储）
python -m src.pipeline --course my_course --step 4
```

---

## 工具接口使用（供模块5调用）

```python
from src.pipeline.video_store import VideoContentStore

# 初始化存储
store = VideoContentStore("my_course")

# 1. 获取课程结构
structure = store.get_course_structure()
# 返回: [{"id": "seg_0001", "time_start": 9.15, "time_end": 76.05, ...}, ...]

# 2. 时间戳定位
segment = store.get_segment_by_time(30.0)
# 返回: {"id": "seg_0001", "text": "...", ...}

# 3. 获取Segment内容
content = store.get_segment_content("seg_0001")
# 返回: {"id": "seg_0001", "text": "完整的转录文本...", ...}

# 4. 批量获取
segments = store.get_multiple_segments(["seg_0001", "seg_0009", "seg_0033"])

# 5. 获取时间范围文本
result = store.get_raw_text_range(0, 300)
# 返回: {"text": "拼接后的文本...", "segment_ids": [...], ...}

# 6. 获取课程统计
stats = store.get_course_stats()
# 返回: {"course_name": "my_course", "total_segments": 69, "total_duration": 7101.96, ...}
```

---

## 语义分割实现

### SeqModel语义分割

使用阿里达摩院 `nlp_bert_document-segmentation_chinese-base` 模型：

```python
# src/pipeline/video_store/segmenter.py

class SeqModelSegmenter:
    """基于transformers直接加载SeqModel"""
    
    def __init__(self, model_path="models/nlp_bert_document-segmentation_chinese-base"):
        from transformers import BertTokenizer, BertForTokenClassification
        import torch
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        # 标签映射 (0: B-EOP段落结束, 1: O非结束)
        self.eop_label = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def is_boundary(self, prev_entries: List[SRTEntry], next_entries: List[SRTEntry]) -> bool:
        """判断两组Entry之间是否是语义边界"""
        prev_text = " ".join(e.text for e in prev_entries)
        next_text = " ".join(e.text for e in next_entries)
        combined = prev_text + "\n" + next_text
        
        # Tokenize and predict
        inputs = self.tokenizer(combined, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # 检查是否包含段落结束标记
        # ...
```

### 备选方案

当SeqModel不可用时，自动降级为 `EntryGroupSegmenter`（基于规则：过渡词/总结词检测）。

---

## Entry级语义分组算法

```python
def build_segments(self, entries: List[SRTEntry]) -> List[Segment]:
    """
    核心算法：以SRT Entry为原子单位的语义分组
    
    保证：
    1. 不切分任何Entry内部
    2. Segment时间戳 = 组内第一个Entry.start ~ 最后一个Entry.end
    3. SRT索引可追踪（记录srt_start_idx和srt_end_idx）
    """
    segments = []
    current_group: List[SRTEntry] = []
    
    for entry in entries:
        current_group.append(entry)
        
        # 检查是否满足切分条件
        if self._should_split(current_group, entries, current_idx):
            segment = self._create_segment(current_group)
            segments.append(segment)
            current_group = []
    
    # 处理剩余Entry...
    return segments
```

---

## 文件结构

### 源码文件

```
src/pipeline/video_store/
├── __init__.py                    # 模块导出
│   └── 导出: SegmentBuilder, VideoStorage, VideoContentStore, ...
│
├── builder.py                     # Segment构建器
│   ├── SegmentBuilder             # 主构建器类
│   │   ├── build(srt_path)        # 从SRT构建Segments
│   │   ├── _build_segments()      # Entry级语义分组
│   │   ├── _should_split()        # 判断是否切分
│   │   ├── _create_segment()      # 创建Segment
│   │   └── _extract_features()    # 提取文本特征
│   │
│   └── Segment                    # Segment数据类
│       ├── id, time_start, time_end, duration
│       ├── srt_start_idx, srt_end_idx
│       ├── text, keywords, features
│       └── to_dict()
│
├── storage.py                     # 存储管理
│   └── VideoStorage               # JSON Lines存储
│       ├── save_segments()        # 保存到segments.jsonl
│       ├── load_segments()        # 加载Segments
│       ├── load_index()           # 加载索引
│       └── load_navigation_map()  # 加载导航图
│
├── interface.py                   # 工具接口
│   └── VideoContentStore          # 9个工具接口实现
│       ├── get_course_structure()         # 获取课程结构
│       ├── get_segment_by_time()          # 时间戳定位
│       ├── get_segments_by_concept()      # 概念定位
│       ├── get_segments_by_type()         # 类型筛选
│       ├── get_segment_content()          # 获取Segment内容
│       ├── get_multiple_segments()        # 批量获取
│       ├── get_raw_text_range()           # 时间范围文本
│       ├── get_segment_metadata()         # 获取元数据
│       └── get_navigation_map()           # 获取导航图
│
└── segmenter.py                   # 语义分割器
    ├── SeqModelSegmenter          # 基于BERT的语义分割
    │   └── is_boundary()          # 判断语义边界
    └── EntryGroupSegmenter        # 规则分割（备选）
        └── is_boundary()          # 基于过渡词判断
```

### 输出文件

```
.cache/{course_name}/facts/
├── segments.jsonl                 # Segment主存储（JSON Lines）
│   # {"id": "seg_0001", "time_start": 9.15, "time_end": 76.05, ...}
│   # {"id": "seg_0009", "time_start": 76.05, "time_end": 143.9, ...}
│
├── index.json                     # 结构化索引
│   # {
│   #   "time": {"9": "seg_0001", "76": "seg_0009"},
│   #   "keywords": {"AutoGen": ["seg_0001"]},
│   #   "types": {"has_code": ["seg_0009"]},
│   #   "total_segments": 69
│   # }
│
└── navigation_map.json            # 导航图
    # {
    #   "course_name": "...",
    #   "total_segments": 69,
    #   "segment_list": [...],
    #   "concept_timeline": [...]
    # }
```

## 源码文件结构

```
src/pipeline/video_store/
├── __init__.py
├── builder.py             # SegmentBuilder：构建segments.jsonl
├── storage.py             # VideoStorage：JSON Lines读写+索引
├── segmenter.py           # SeqModelSegmenter + EntryGroupSegmenter
└── interface.py           # VideoContentStore：9个工具接口

.cache/{course_name}/facts/
├── segments.jsonl         # Segment主存储
├── index.json             # 结构化索引
└── navigation_map.json    # 导航图
```

---

## 测试验证

### 单元测试

```bash
$ python -m pytest tests/unit/test_video_store.py -v

============================= test session starts ==============================
collected 15 items

tests/unit/test_video_store.py::TestSegmentBuilder::test_create_segment_from_entries PASSED
tests/unit/test_video_store.py::TestSegmentBuilder::test_build_segments_with_rules PASSED
tests/unit/test_video_store.py::TestSegmentBuilder::test_segment_duration_constraints PASSED
tests/unit/test_video_store.py::TestVideoStorage::test_save_and_load_segments PASSED
tests/unit/test_video_store.py::TestVideoStorage::test_index_building PASSED
tests/unit/test_video_store.py::TestVideoContentStore::test_get_course_structure PASSED
tests/unit/test_video_store.py::TestVideoContentStore::test_get_segment_by_time PASSED
tests/unit/test_video_store.py::TestVideoContentStore::test_get_segments_by_type PASSED
tests/unit/test_video_store.py::TestVideoContentStore::test_get_segment_content PASSED
tests/unit/test_video_store.py::TestVideoContentStore::test_get_multiple_segments PASSED
tests/unit/test_video_store.py::TestVideoContentStore::test_get_raw_text_range PASSED
tests/unit/test_video_store.py::TestVideoContentStore::test_get_course_stats PASSED
tests/unit/test_video_store.py::TestSegmenter::test_entry_group_segmenter PASSED
tests/unit/test_video_store.py::TestSegmenter::test_seqmodel_segmenter_init SKIPPED
tests/unit/test_video_store.py::TestVideoStoreIntegration::test_full_pipeline PASSED

======================== 14 passed, 1 skipped in 8.08s =========================
```

---

## 关键设计决策

| 决策 | 说明 |
|------|------|
| **JSON Lines存储** | 轻量、可读、无需SQLite/Chroma，适合<100个Segment |
| **无LLM处理** | 特征提取用规则（正则），不用LLM，避免过度设计 |
| **Segment为原子单位** | 所有操作围绕Segment ID，类似课件RAG的Chunk ID |
| **与课件RAG解耦** | 视频检索结构化为主，课件检索语义为主，各自最优 |
| **transformers直接加载** | 绕过ModelScope依赖问题，使用transformers原生API |
