# STEP 0: 课件RAG构建模块

> **模块0**: 将课程材料（markdown、jupyter notebook）转换为向量索引，供后续LLM校对和检索使用。
> 
> 依赖：无（最先执行）

---

## 输入

```yaml
input:
  path: "courses/{course_name}/materials/"
  supported_formats: ["*.md", "*.ipynb"]
  structure: "原始课件文件"
```

---

## 处理流程

```yaml
pipeline:
  step_1_detect:
    description: "检测输入文件类型"
    action: "glob扫描目录"
  
  step_2_preprocess_ipynb:
    tool: "nbconvert"
    description: "Jupyter Notebook预处理，仅保留markdown和代码，删除运行结果"
    command: |
      jupyter nbconvert --to markdown input.ipynb \
        --TemplateExporter.exclude_output=True \
        --TemplateExporter.exclude_input_prompt=True \
        --TemplateExporter.exclude_output_prompt=True
        
  step_3_chunk:
    description: "文本分块"
    chunk_size: 512
    overlap: 50
    preserve_structure: true  # 保留标题层级
    
  step_4_embed:
    model: "BAAI/bge-large-zh-v1.5"
    device: "cpu"  # 或cuda
    output_dim: 1024
    
  step_5_store:
    vector_db: "chroma"
    metadata_store: "json"  # 存储原文和章节信息
```

---

## 输出

```yaml
output:
  processed_materials: "courses/{course_name}/processed_materials/"
    # 存放转换后的.md文件
  vector_index: "vector_db/course_materials/{course_name}/"
    files:
      - "index"              # 向量索引
      - "documents.json"     # 原文和元数据
      - "metadata.json"      # 课程信息
```

---

## 核心类设计

```python
# src/pipeline/course_rag.py

class CourseRAGBuilder:
    """课件RAG构建器 - 用于构建课程向量索引"""
    
    def __init__(self, course_name: str, config: Optional[Dict] = None)
    def detect_files(self) -> List[Path]                    # 步骤1: 检测文件
    def preprocess_notebooks(self, files: List[Path])       # 步骤2: 预处理Notebook
    def chunk_documents(self, files: List[Path])            # 步骤3: 文本分块
    def embed_chunks(self, documents)                       # 步骤4: 生成嵌入
    def store_vectors(self, chunks_with_embeddings)         # 步骤5: 存储向量
    def build(self) -> Dict[str, Any]                       # 执行完整构建
    def query(self, query_text: str, top_k: int = 5)        # 测试查询


class CourseRAGQueryInterface:
    """课件RAG查询接口 - 供其他模块使用（只读）"""
    
    def __init__(self, course_name: str, config: Optional[Dict] = None)
    
    # 核心查询方法
    def search(self, query_text: str, top_k: int = 5) -> List[Dict]
    """语义搜索课程内容"""
    
    def search_by_heading(self, heading_keyword: str, top_k: int = 10) -> List[Dict]
    """按标题关键词搜索"""
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> Dict[str, List[Dict]]
    """批量搜索多个查询"""
    
    # 辅助方法
    def exists(self) -> bool
    def get_stats(self) -> Dict[str, Any]
    def get_document_outline(self) -> List[Dict]
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]
    def format_for_llm(self, results: List[Dict], max_length: int = 2000) -> str
```

---

## 快速开始

### 1. 准备测试数据

```bash
mkdir -p courses/test_course/materials
```

创建测试文件 `courses/test_course/materials/test1.md`：

```markdown
# 深度学习基础

本文介绍深度学习的基本概念。

## 神经网络

神经网络是深度学习的核心组件。

### 前馈神经网络

前馈神经网络是最基本的网络结构。

## 优化算法

### 梯度下降

梯度下降是训练神经网络的基础算法。
```

### 2. 构建课程RAG

```python
from src.pipeline.course_rag import CourseRAGBuilder

# 初始化构建器
builder = CourseRAGBuilder("test_course", {
    "embedding_model": "models/bge-large-zh-v1.5",
    "embedding_device": "cpu",
    "chunk_size": 512,
    "chunk_overlap": 50
})

# 执行完整构建
result = builder.build()
# 返回: {status: "success", total_chunks: 79, vector_dim: 1024, ...}
```

### 3. 命令行构建

```bash
# 步骤0: 构建课件RAG（必须先执行）
python -m src.pipeline --course my_course --step 0

# 带查询测试
python -m src.pipeline.course_rag --course test_course --query "神经网络"
```

---

## 测试方式

### 单元测试

```bash
# 安装 pytest
pip install pytest

# 运行单元测试
python -m pytest tests/unit/test_course_rag.py -v
```

### 交互式测试

```python
from src.pipeline.course_rag import CourseRAGBuilder

# 初始化
builder = CourseRAGBuilder("test_course")

# 步骤1: 检测文件
files = builder.detect_files()
print(f"检测到 {len(files)} 个文件")

# 步骤2-5: 完整构建
result = builder.build()
print(f"\n构建结果: {result}")

# 测试查询
if result['status'] == 'success':
    results = builder.query("神经网络是什么", top_k=3)
    print(f"\n查询结果:")
    for r in results:
        print(f"  [{1-r['distance']:.4f}] {r['text'][:50]}...")
```

---

## 验证结果

### 检查输出文件

```bash
# 查看向量数据库
ls vector_db/course_materials/test_course/

# 应包含：
# - chroma.sqlite3 (或相关数据库文件)
# - documents.json
# - metadata.json
```

### 查看构建信息

```bash
cat vector_db/course_materials/test_course/metadata.json
```

---

## 下游模块使用

### LLM校对模块调用（模块3使用）

```python
from src.pipeline.course_rag import CourseRAGQueryInterface

# 初始化查询接口（只读，不重新构建）
query_interface = CourseRAGQueryInterface("my_course")

# 检查课程RAG是否存在
if not query_interface.exists():
    raise Exception("请先执行步骤0构建课件RAG")

# 语义搜索相关内容
results = query_interface.search("PyTorch神经网络", top_k=3)

# 格式化为LLM上下文
context = query_interface.format_for_llm(results)

# 构建Prompt
prompt = f"""根据以下课件内容纠正转录文本：

{context}

待校对文本: "{transcribed_text}"
请输出更正后的文本。"""
```

### RAG检索模块调用（模块5使用）

```python
query_interface = CourseRAGQueryInterface(course_name)

# 多路召回
semantic_results = query_interface.search(user_query, top_k=5)
heading_results = query_interface.search_by_heading("关键概念")

# 合并结果
all_results = semantic_results + heading_results
```

---

## 常见问题

### Q1: 模型加载失败
```
错误: FileNotFoundError: models/bge-large-zh-v1.5/config.json
```
**解决:** 检查模型文件是否存在，确认路径正确

### Q2: 找不到课程材料
```
错误: FileNotFoundError: Materials directory not found
```
**解决:**
```bash
mkdir -p courses/my_course/materials
# 然后添加课件文件
```

### Q3: 内存不足
```
错误: RuntimeError: out of memory
```
**解决:** 使用CPU设备或减小batch_size

---

## 关键特性

| 特性 | 实现 |
|------|------|
| 标题层级保留 | 支持H1-H6六级标题，分块时保留完整层级信息 |
| Notebook处理 | nbconvert转换，排除output和prompt |
| 智能分块 | chunk_size=512, overlap=50，优先按标题分割 |
| 本地模型 | 支持本地BGE模型路径，避免重复下载 |
| 增量查询 | 查询接口与构建分离，支持只读访问 |
| 批量搜索 | 支持一次查询多个术语（用于校对） |

---

## 文件结构

### 源码文件

```
src/pipeline/course_rag.py          # 主实现文件 (1200+行)
├── CourseRAGBuilder               # 构建器类 - 完整RAG构建流程
│   ├── detect_files()             # 检测课件文件
│   ├── preprocess_notebooks()     # Notebook预处理
│   ├── chunk_documents()          # 文档分块
│   ├── embed_chunks()             # 生成向量嵌入
│   ├── store_vectors()            # 存储到向量库
│   └── build()                    # 执行完整构建
│
├── CourseRAGQueryInterface        # 查询接口类（供其他模块使用）
│   ├── search()                   # 语义搜索
│   ├── search_by_heading()        # 按标题搜索
│   ├── batch_search()             # 批量搜索
│   ├── get_document_outline()     # 获取文档大纲
│   └── format_for_llm()           # 格式化为LLM上下文
│
├── BGEEmbedder                    # BGE嵌入模型封装
│   ├── embed()                    # 单文本嵌入
│   └── embed_chunks()             # 批量分块嵌入
│
├── ChromaVectorStore              # Chroma向量存储
│   ├── add_chunks()               # 添加分块
│   ├── query()                    # 向量检索
│   └── delete_course()            # 删除课程数据
│
├── MarkdownProcessor              # Markdown处理器
│   ├── extract_structure()        # 提取文档结构
│   └── chunk_text()               # 文本分块
│
└── NotebookProcessor              # Notebook处理器
    └── convert_to_markdown()      # nbconvert转换
```

### 输出文件

```
vector_db/course_materials/{course_name}/
├── chroma.sqlite3                 # Chroma主数据库（向量索引）
├── documents.json                 # 文档片段原文和元数据
└── metadata.json                  # 构建元数据（时间、配置等）

courses/{course_name}/
└── processed_materials/           # 转换后的Markdown文件
    └── *.md                       # 从ipynb转换的Markdown
```
