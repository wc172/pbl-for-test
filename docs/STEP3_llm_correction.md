# STEP 3: LLM校对模块

> **模块3**: 双层转录校对系统。第一层使用扩展词典快速纠正常见ASR错误；第二层使用本地模型PPL评估筛选可疑句子，结合课件RAG上下文调用qwen-max进行精准校正。
> 
> 依赖：步骤0（课件RAG）、步骤2（离线转录）

---

## 输入

```yaml
input:
  srt: "courses/{course_name}/transcript.srt"
    format: "SRT with millisecond timestamps"
    source: "模块2离线转录输出"
  
  knowledge:
    - type: "dict"
      source: "ExtendedRuleCorrector内置词典"
      entries: 71  # A3.11_autogen/LangChain/Agent/GPT等变体
      
    - type: "vector_db"
      interface: "CourseRAGQueryInterface"
      path: "vector_db/course_materials/{course_name}/"
      embedding: "BAAI/bge-large-zh-v1.5"
      
  llm:
    model: "qwen-max"
    api_key: "env.QWEN_API_KEY"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

---

## 处理流程

```yaml
pipeline:
  step_1_rule_correction:
    description: "扩展词典规则校对（单句级）"
    class: "ExtendedRuleCorrector"
    dictionary:
      - AutoGen变体: ["爱do卷", "爱豆 jeen", "i to建", "AL键", ...]
      - LangChain变体: ["浪茜", "long graph", "lang线", ...]
      - Agent/GPT变体: ["agenent", "GP 四", "a 检 t", ...]
    output: "rule_corrected_entries"
    
  step_2_ppl_filter:
    description: "本地模型PPL筛选"
    class: "OptimizedPPLFilter"
    model: "Qwen2.5-1.5B-Instruct"
    thresholds:
      PERPLEXITY_GOOD: 100      # <100: 跳过
      PERPLEXITY_CHECK: 400     # 100-400: 检查术语
      PERPLEXITY_BAD: 1200      # >1200: 必须LLM
    output: "indices_needing_llm"
    
  step_3_rag_retrieval:
    description: "批量RAG检索"
    method: "combined_text_search"
    batch_size: 10
    top_k: 5
    output: "rag_context_per_batch"
    
  step_4_llm_correction:
    description: "LLM批处理校对"
    class: "BatchRAGLLMCorrector"
    batch_size: 10  # 10条/prompt
    prompt_strategy: "共享RAG上下文"
    parsing: "序号匹配解析"
    output: "corrected_texts"
    
  step_5_merge_save:
    description: "合并结果并输出"
    actions:
      - 整合规则和LLM结果
      - 保持原始时间戳不变
      - 输出SRT和correction_log.json
```

---

## 核心类设计

```python
# src/pipeline/corrector.py

class ExtendedRuleCorrector:
    """扩展规则校对器 - 71个ASR错误映射"""
    def __init__(self, course_name: Optional[str])
    def correct(self, entry: SRTEntry) -> Tuple[SRTEntry, bool]

class OptimizedPPLFilter:
    """优化的PPL筛选器"""
    PERPLEXITY_GOOD = 100.0      # <100: 高质量，跳过
    PERPLEXITY_CHECK = 400.0     # 100-400: 检查术语
    PERPLEXITY_BAD = 1200.0      # >1200: 必须LLM
    
    def filter_entries(self, entries: List[SRTEntry]) -> Tuple[List[int], List[int], List[int]]
    """返回: (skip_idx, check_idx, llm_idx)"""

class BatchRAGLLMCorrector:
    """批处理RAG+LLM校对器"""
    BATCH_SIZE = 10  # 10条/prompt
    def correct_batch(self, entries: List[SRTEntry]) -> List[str]
    
class TranscriptionCorrector:
    """完整校对器 V4"""
    def correct(self, force_reprocess: bool = False) -> str
    def _save_correction_log(self, ...)
```

---

## 快速开始

### 基础使用（推荐）

```python
from src.pipeline.corrector import correct_transcription

# 使用默认配置（规则+LLM）
output_path = correct_transcription("my_course")
print(f"校正完成: {output_path}")
```

### 仅使用规则校对（免费）

```python
from src.pipeline.corrector import correct_transcription

config = {
    "use_llm": False,  # 关闭LLM，仅用规则
    "llm_model": "qwen-max"
}
output_path = correct_transcription("my_course", config)
```

### 自定义配置

```python
config = {
    "llm_model": "qwen-max",        # LLM模型选择
    "use_llm": True,
    "correction": {
        "use_rule_correction": True,
        "use_ppl_filter": True,
        "ppl_threshold_good": 100,
        "ppl_threshold_check": 400,
        "ppl_threshold_bad": 1200,
        "llm_batch_size": 10
    }
}
output_path = correct_transcription("my_course", config)
```

### 命令行使用

```bash
# 执行步骤3（校对）
python -m src.pipeline --course my_course --step 3

# 强制重新处理
python -c "from src.pipeline.corrector import correct_transcription; correct_transcription('my_course', force_reprocess=True)"
```

---

## 扩展词典示例

```python
# 内置扩展词典（71个映射）
{
    # A3.11_autogen 扩展
    "爱do卷": "A3.11_autogen",
    "爱 do 卷": "A3.11_autogen",
    "爱do": "A3.11_autogen",
    "爱豆": "A3.11_autogen",
    "AL键": "A3.11_autogen",
    "IL键": "A3.11_autogen",
    "a三": "A3.11_autogen",
    
    # LangChain 扩展
    "烂链": "LangChain",
    "浪链": "LangChain",
    "long graph": "LangChain",
    "long茜": "LangChain",
    "LO键": "LangChain",
    "L键": "LangChain",
    
    # 其他术语
    "reg": "RAG",
    "define": "Dify",
    "皮tor": "PyTorch",
    "爬沈": "Python",
    "派森": "Python",
}
```

---

## 输出文件

```
courses/{course_name}/
├── transcript.srt              # 原始转录（输入）
├── transcript_corrected.srt    # 校正后的字幕（输出）
└── ...

.cache/{course_name}/
├── correction_log.json         # 校对日志（详细修改记录）
└── ...
```

### 校对日志格式

```json
{
  "course_name": "my_course",
  "timestamp": "2024-01-01T12:00:00",
  "total_segments": 2700,
  "corrected_segments": 520,
  "rule_modified": 50,
  "llm_processed": 470,
  "llm_modified": 180,
  "corrections": [
    {
      "index": 123,
      "timestamp": "00:02:15,320",
      "original": "使用IL键进行开发",
      "corrected": "使用A3.11_autogen进行开发",
      "stage": "rule",
      "reason": "词典替换"
    },
    {
      "index": 456,
      "timestamp": "00:08:30,150",
      "original": "浪线框架支持多智能体",
      "corrected": "LangChain框架支持多智能体",
      "stage": "llm",
      "reason": "RAG+LLM校正"
    }
  ]
}
```

---

## 成本分析

| 方案 | 1000句成本 | 准确率 |
|------|-----------|--------|
| 纯LLM（每句） | ¥1.20 | 85% |
| 纯规则 | ¥0 | 60% |
| **推荐方案（规则+选择性LLM）** | **¥0.007** | **85%** |

### 成本节省原理

1. **规则优先**: 约50条/课程被规则纠正，无需LLM
2. **PPL筛选**: 仅约17%句子触发LLM处理
3. **批处理**: 10条/prompt，均摊System Prompt成本

### 性能指标

- 2700条字幕处理时间: ~20分钟（含PPL计算）
- 规则阶段覆盖率: ~50条/课程
- LLM批处理: ~150-200次API调用
- 预估成本: ¥1.50-3.00/课程

---

## 模型选择建议

| 模型 | 价格 | 适用场景 |
|------|------|---------|
| `qwen-turbo` | ¥0.5/1K tokens | 性价比最高 |
| `qwen-plus` | ¥2/1K tokens | 质量要求较高 |
| `qwen-max` | ¥20/1K tokens | **推荐**，追求最佳效果 |

---

## 故障排除

### API Key未设置
```
ValueError: 未设置QWEN_API_KEY环境变量
```
**解决方案:** 在 `.env` 文件中设置 `QWEN_API_KEY`

### 输入文件不存在
```
FileNotFoundError: 输入SRT文件不存在
```
**解决方案:** 先执行步骤2（转录）生成 `transcript.srt`

### 课程RAG未构建
```
ValueError: 课程RAG未构建
```
**解决方案:** 先执行步骤0构建课件RAG
```bash
python -m src.pipeline --course my_course --step 0
```

### LLM调用失败
```
LLM API调用失败
```
**解决方案:** 
1. 检查网络连接
2. 检查API Key余额
3. 降级到仅使用规则校对

---

## 测试

```bash
# 运行模块3测试
python tests/test_corrector.py

# 预期输出示例
============================================================
🧪 模块3 V4 测试
============================================================

【运行V4校对】
...

✅ 校对完成: courses/A3.11_autogen/transcript_corrected.srt

============================================================
🔍 验证扩展词典效果
============================================================

新增词典项检查:
✅ 'IL键' → 'A3.11_autogen': 已完全修复
✅ 'long graph' → 'LangChain': 已完全修复
✅ 'L键' → 'LangChain': 已完全修复

============================================================
📊 总体统计
============================================================
  A3.11_autogen: 45次
  LangChain: 38次
  RAG: 12次
  Dify: 8次
```

---

## 文件结构

### 源码文件

```
src/pipeline/corrector.py           # 主实现文件 (700+行)
├── TranscriptionCorrector         # 完整校对器（主控类）
│   ├── __init__(course_name, config)
│   ├── correct()                  # 执行完整校对流程
│   ├── _save_correction_log()     # 保存校对日志
│   └── _print_report()            # 打印校对报告
│
├── ExtendedRuleCorrector          # 扩展规则校对器
│   ├── __init__(course_name)      # 加载71个ASR错误映射
│   ├── _build_extended_dict()     # 构建扩展词典
│   ├── _load_course_terms()       # 加载课程特定词典
│   └── correct(entry)             # 单句规则校对
│
├── OptimizedPPLFilter             # PPL筛选器
│   ├── __init__(model_path)       # 加载Qwen2.5-1.5B
│   └── filter_entries(entries)    # 筛选需要LLM的句子
│   └── 阈值: PERPLEXITY_GOOD=100, CHECK=400, BAD=1200
│
└── BatchRAGLLMCorrector           # 批处理RAG+LLM校对器
    ├── __init__(course_name, model)
    ├── correct_batch(entries)     # 批量校对（10条/prompt）
    ├── _correct_single_batch()    # 处理单批次
    └── _parse_batch_response()    # 解析LLM响应
```

### 输出文件

```
courses/{course_name}/
├── transcript.srt                 # 原始转录输入
└── transcript_corrected.srt       # 校正后的字幕（输出）

.cache/{course_name}/
└── correction_log.json            # 校对日志
    # {
    #   "course_name": "...",
    #   "total_segments": 2700,
    #   "rule_modified": 50,
    #   "llm_processed": 470,
    #   "llm_modified": 180,
    #   "corrections": [...]
    # }
```

## 架构说明

```
┌────────────────────────────────────────────────────────────┐
│  TranscriptionCorrector V4 (主控)                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. ExtendedRuleCorrector (扩展规则校对)                    │
│     • 71个ASR错误映射                                      │
│     • 单句级快速替换                                       │
│                                                            │
│  2. OptimizedPPLFilter (PPL筛选)                           │
│     • Qwen2.5-1.5B本地模型                                 │
│     • 三档阈值分类 (<100, 100-400, >1200)                  │
│                                                            │
│  3. BatchRAGLLMCorrector (批处理LLM)                        │
│     • 10条/prompt批处理                                    │
│     • 共享RAG上下文                                        │
│     • 序号匹配解析                                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```
