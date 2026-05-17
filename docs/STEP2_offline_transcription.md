# STEP 2: 离线转录模块

> **模块2**: 使用 FunASR/Paraformer 进行离线语音识别，生成带时间戳的SRT字幕。本模块仅输出文件，不提供编程接口。
> 
> 依赖：步骤1（输入预处理）

---

## 输入

```yaml
input:
  audio: "courses/{course_name}/audio.wav"  # 16kHz, 16bit PCM（来自模块1）
```

---

## 处理流程

```yaml
pipeline:
  step_1_load_audio:
    description: "加载并验证音频文件"
    
  step_2_transcribe:
    description: "Paraformer语音识别"
    tool: "funasr"
    model: "paraformer-zh"      # 三合一模型（ASR+VAD+PUNC）
    sentence_timestamp: true    # 关键：获取句子级别时间戳
    
  step_3_save_results:
    description: "保存SRT和JSON结果"
    output:
      - "courses/{course_name}/transcript.srt"
      - ".cache/{course_name}/transcription_result.json"
```

---

## 输出文件

### 1. SRT字幕文件

```yaml
path: "courses/{course_name}/transcript.srt"
format: "SubRip"
encoding: "utf-8"

example: |
  1
  00:00:09,150 --> 00:00:15,320
  好了，同学们晚上好，我们再等一等。
  
  2
  00:00:15,320 --> 00:00:22,150
  其他的同学，我们八点零一正式开上课。
```

### 2. JSON原始结果文件

```yaml
path: ".cache/{course_name}/transcription_result.json"
format: "JSON"
encoding: "utf-8"

content_schema:
  text: str                    # 完整文本（所有片段拼接）
  segments: List[Segment]      # 片段列表
  duration_ms: int             # 总时长（毫秒）
  
Segment:
  index: int          # 序号（从1开始）
  start_ms: int       # 开始时间（毫秒）
  end_ms: int         # 结束时间（毫秒）
  text: str           # 文本内容

example: |
  {
    "text": "好了，同学们晚上好...",
    "segments": [
      {"index": 1, "start_ms": 9150, "end_ms": 15320, "text": "好了，同学们晚上好，我们再等一等。"},
      {"index": 2, "start_ms": 15320, "end_ms": 22150, "text": "其他的同学，我们八点零一正式开上课。"}
    ],
    "duration_ms": 7147680
  }
```

---

## 快速开始

### Python 调用

```python
from src.pipeline.transcriber import transcribe_audio

# 执行转录（从步骤1的 audio.wav 生成 transcript.srt）
srt_path = transcribe_audio("my_course")
print(f"转录完成: {srt_path}")
```

### 命令行使用

```bash
# 执行步骤2（转录）
python -m src.pipeline --course my_course --step 2
```

---

## 下游模块使用（模块3）

模块3（LLM校对）应读取上述两个文件：

```python
from pathlib import Path
import json

course_name = "xxx"

# 1. 读取JSON获取结构化数据（推荐）
json_path = Path(f".cache/{course_name}/transcription_result.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    
full_text = data["text"]           # 完整文本，用于LLM理解上下文
segments = data["segments"]        # 片段列表，用于逐段校正
duration_ms = data["duration_ms"]  # 总时长

# 2. 如需SRT原始内容
srt_path = Path(f"courses/{course_name}/transcript.srt")
with open(srt_path, 'r', encoding='utf-8') as f:
    srt_content = f.read()
```

### 模块3开发注意事项

1. 必须处理 `segments` 列表为空的情况（转录失败）
2. `start_ms` 和 `end_ms` 是毫秒时间戳，需转换为SRT格式 `HH:MM:SS,mmm`
3. 校正后的文本需保持原有的时间戳结构，只修改 `text` 字段
4. 输出SRT时保持原有编号（`index`）不变，避免时间轴错乱

---

## 关于 Paraformer-large-vad-punc 模型

这是一个**三合一模型**，名称中的后缀含义：
- `paraformer-large`: 主ASR模型（语音识别）
- `vad`: 内置语音活动检测（Voice Activity Detection）
- `punc`: 内置标点恢复（Punctuation Restoration）

**无需单独下载**：
- ❌ VAD模型（如 fsmn-vad）- 已内置
- ❌ 标点模型（如 ct-punc-c）- 已内置

**可选配置**：
- ✅ 语言模型（LM）- 可提升准确率但增加耗时
- ✅ 热词（hotword）- 提升特定术语识别率

---

## 配置项

```yaml
# config/pipeline.yaml

transcription:
  # ASR模型配置（支持在线下载或本地路径）
  model: "models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
  local_model: true                # 是否使用本地模型（自动检测）
  model_revision: "v2.0.4"         # ModelScope版本（在线模式使用）
  device: "cuda"                   # cuda 或 cpu
  
  # 批处理参数
  batch_size_s: 60                 # 秒，建议60秒（1分钟）
  
  # 解码优化参数
  beam_size: 10                    # 集束搜索宽度（5-15）
  decoding_ctc_weight: 0.0         # CTC权重（0.0-1.0）
  decoding_mode: "model1"          # 解码模式
  
  # 热词功能（可选）
  hotword: ""                      # 逗号分隔，如 "PyTorch,神经网络"
  
  # 语言模型（可选，提升准确率但增加耗时）
  lm_model: ""                     # 本地N-gram语言模型路径
  lm_weight: 0.15
  lm_beam_size: 10
```

---

## 文件结构

### 源码文件

```
src/pipeline/transcriber.py         # 主实现文件
├── AudioTranscriber               # 离线转录器类
│   ├── __init__(course_name, config)
│   ├── transcribe()               # 执行完整转录流程
│   ├── _load_model()              # 加载Paraformer模型
│   ├── _load_hotwords()           # 加载热词配置
│   └── _save_results()            # 保存SRT和JSON结果
│
├── TranscriptionSegment           # 转录片段数据类
│   ├── index                      # 序号
│   ├── start_ms                   # 开始时间（毫秒）
│   ├── end_ms                     # 结束时间（毫秒）
│   ├── text                       # 文本内容
│   └── to_srt_format()            # 转换为SRT格式
│
└── 便捷函数
    └── transcribe_audio()         # 便捷函数：执行转录
```

### 输出文件

```
courses/{course_name}/
└── transcript.srt                 # SRT字幕文件
    # 1
    # 00:00:09,150 --> 00:00:15,320
    # 好了，同学们晚上好...

.cache/{course_name}/
└── transcription_result.json      # JSON原始结果
    # {
    #   "text": "完整文本...",
    #   "segments": [{"index": 1, "start_ms": 9150, ...}],
    #   "duration_ms": 7147680
    # }
```

## ASR 模型对比

| 特性 | faster-whisper | Paraformer (FunASR) |
|------|----------------|---------------------|
| 时间戳粒度 | 单词级 | **句子级** |
| 时间戳单位 | 秒(float) | **毫秒(int)** |
| 内置VAD | 需单独启用 | **已集成** |
| 内置标点 | 依赖模型 | **已集成PUNC** |
| 长音频支持 | 需手动分块 | **直接支持数小时** |
| 断点续传 | 需要 | **不需要** |
| 中文优化 | 一般 | **优秀** |

---

## 本地模型配置

### 模型文件放置结构

```
models/
└── speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
    ├── config.yaml              # 模型配置文件
    ├── model.pb                 # 模型权重文件
    ├── tokens.txt               # 词汇表
    ├── am.mvn                   # 特征归一化文件
    └── example/
        └── asr_example.wav      # 示例音频
```

### 自动检测规则

- 路径以 `models/` 开头 → 自动识别为本地模型
- 路径包含 `/` 或 `\` → 自动识别为本地模型
- 否则 → 视为 ModelScope 模型ID，从云端下载

### 模型下载方式

```bash
# 方式1: 首次运行时自动下载（在线模式）
python -m src.pipeline --course my_course --step 2

# 方式2: 手动下载后配置本地路径（离线模式）
# 下载地址: https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
# 放置到: models/ 目录下
# 修改配置: local_model: true
```
