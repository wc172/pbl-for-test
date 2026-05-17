# 视频转录 + RAG 系统

基于 LangChain 1.0 + MCP + FunASR/Paraformer 的视频内容智能处理与检索系统。

## 系统概述

本系统支持：
- 🎬 **视频转录**: 使用 FunASR/Paraformer 进行中文语音识别（内置VAD+标点）
- 📝 **智能校对**: 利用课件RAG和术语词典进行LLM纠错
- 📑 **多级摘要**: 生成全局/分段摘要，支持时间戳查询
- 🔍 **语义检索**: RAG检索+重排序，生成带引用的回答
- 🔧 **MCP工具**: 封装为MCP工具供外部Agent调用

## 核心特性

| 特性 | 说明 |
|------|------|
| **中文优化** | Paraformer 针对中文语音识别优化，准确率 SOTA |
| **长音频支持** | 直接支持数小时音频，无需分片 |
| **内置VAD** | 自动语音活动检测，跳过静音段 |
| **内置标点** | 自动添加标点符号 |
| **句子级时间戳** | 输出句子级时间戳，适合字幕生成 |

## 项目结构

```
.
├── src/                        # 源代码
│   ├── pipeline/               # 处理流水线
│   │   ├── course_rag.py       # 步骤0: 课件RAG构建
│   │   ├── preprocessor.py     # 步骤1: 输入预处理
│   │   ├── transcriber.py      # 步骤2: 离线转录
│   │   ├── corrector.py        # 步骤3: LLM校对
│   │   ├── summarizer.py       # 步骤4: 多级摘要
│   │   └── main.py             # 流水线主控
│   ├── rag/                    # RAG检索模块
│   │   └── retriever.py        # 检索+重排序
│   ├── mcp_server/             # MCP服务
│   │   └── server.py           # MCP服务器
│   ├── cli/                    # 命令行工具
│   │   └── main.py             # CLI入口
│   ├── utils/                  # 工具函数
│   │   ├── checkpoint.py       # 断点管理
│   │   └── srt_parser.py       # SRT解析
│   └── models/                 # 数据模型
│       └── schemas.py          # Pydantic模型
├── config/                     # 配置文件
│   ├── pipeline.yaml           # 流水线配置
│   └── logging.yaml            # 日志配置
├── courses/                    # 课程数据
├── vector_db/                  # 向量数据库
├── knowledge_base/             # 知识库
│   └── common_misrecognition.md  # 常见误识别词典
├── .cache/                     # 缓存/断点
├── tests/                      # 测试
│   ├── unit/                   # 单元测试
│   └── integration/            # 集成测试
├── docs/                       # 文档
├── logs/                       # 日志文件
├── .env                        # 环境变量
├── requirements.txt            # Python依赖
└── README.md                   # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env: QWEN_API_KEY=your_key
```

### 2. 目录初始化

```bash
mkdir -p courses/my_course/materials

# 放入课件
cp lecture.ipynb courses/my_course/materials/

# 放入视频
cp video.mp4 courses/my_course/input.mp4
```

**⚠️ 注意：课程名称大小写敏感**
- 课程目录名必须与查询时使用的大小写完全一致
- 建议使用 `list_available_courses` 工具获取准确的课程名称

### 3. 执行流水线

```bash
# 步骤0: 构建课件RAG（必须先执行）
python -m src.pipeline --course my_course --step 0

# 步骤1-5: 完整处理
python -m src.pipeline --course my_course

# 或分步执行
python -m src.pipeline --course my_course --step 1
python -m src.pipeline --course my_course --step 2
python -m src.pipeline --course my_course --step 3
python -m src.pipeline --course my_course --step 4
```

### 4. 启动MCP服务

```bash
# stdio模式（KimiCode）
python -m src.mcp_server

# sse模式
python -m src.mcp_server --transport sse --port 8080
```

### 5. 查询测试

```bash
python -m src.cli query --course my_course "老师在第5分钟讲了什么"
```

## 模块依赖关系

```
步骤0: 课件RAG构建
    │
    ▼
步骤1: 输入预处理 ──依赖──┐
    │                    │
    ▼                    │
步骤2: 离线转录          │
    │                    │
    ▼                    │
步骤3: LLM校对 ──────────┤──依赖步骤0的课件RAG
    │                    │
    ▼                    │
步骤4: 多级摘要          │
    │                    │
    ▼                    │
步骤5: RAG检索重排序 ────┘
    │
    ▼
步骤6: MCP服务（暴露工具）
```

## 配置说明

详见 `config/pipeline.yaml`:

```yaml
# 音频处理（适配Paraformer）
audio:
  samplerate: 16000      # Paraformer要求16kHz
  channels: 1            # 单声道
  bit_depth: 16          # 16bit PCM
  format: "wav"

# 转录配置（FunASR/Paraformer）
transcription:
  model: "paraformer-zh"
  model_revision: "v2.0.4"
  device: "cpu"
  batch_size_s: 300      # 批处理大小（秒）
  hotword: ""            # 热词列表，如"PyTorch,神经网络"

llm:
  model: "qwen-max"
  
rag:
  embedding_model: "BAAI/bge-large-zh-v1.5"
```

## ASR 模型对比

| 特性 | faster-whisper | Paraformer (本系统) |
|------|----------------|---------------------|
| 中文准确率 | 一般 | **优秀** |
| 时间戳粒度 | 单词级 | **句子级** |
| 内置VAD | 需单独启用 | **已集成** |
| 内置标点 | 依赖模型 | **已集成** |
| 长音频支持 | 需手动分块 | **直接支持** |
| 断点续传 | 需要 | **不需要** |

## 许可证

MIT License
