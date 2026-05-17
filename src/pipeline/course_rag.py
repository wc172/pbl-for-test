"""
模块0: 课件RAG构建模块

将课程材料（markdown、jupyter notebook）转换为向量索引，
供后续LLM校对和检索使用。

处理流程:
1. 检测输入文件类型 (glob扫描目录)
2. Jupyter Notebook预处理 (nbconvert，排除output)
3. 文本分块 (chunk_size=512, overlap=50, 保留标题层级)
4. 向量嵌入 (BAAI/bge-large-zh-v1.5)
5. 存储到Chroma向量数据库
"""

import os
import re
import json
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import yaml

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """文本分块数据结构"""
    id: str
    text: str
    source_file: str
    chunk_index: int
    start_pos: int
    end_pos: int
    headings: List[str] = field(default_factory=list)  # 层级标题
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "headings": self.headings,
            "metadata": self.metadata
        }


@dataclass
class ProcessedDocument:
    """处理后的文档"""
    file_path: str
    file_type: str  # 'markdown' | 'notebook'
    title: str
    content: str
    chunks: List[TextChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarkdownProcessor:
    """Markdown文档处理器"""
    
    # Markdown标题正则
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def extract_structure(self, content: str) -> List[Dict[str, Any]]:
        """
        提取文档结构（保留标题层级）
        
        Returns:
            结构化块列表，每个块包含文本和标题层级信息
        """
        lines = content.split('\n')
        blocks = []
        current_block = {"text": [], "headings": []}
        current_headings = [""] * 6  # 支持6级标题层级 (H1-H6)
        
        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)
            
            if heading_match:
                # 保存当前块
                if current_block["text"]:
                    blocks.append({
                        "text": '\n'.join(current_block["text"]).strip(),
                        "headings": [h for h in current_block["headings"] if h]
                    })
                    current_block = {"text": [], "headings": []}
                
                # 更新标题层级
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # 确保列表足够长
                if level > len(current_headings):
                    current_headings.extend([""] * (level - len(current_headings)))
                
                current_headings[level - 1] = title
                # 清除更低级别的标题
                for i in range(level, len(current_headings)):
                    current_headings[i] = ""
                    
                current_block["headings"] = current_headings.copy()
                current_block["text"].append(line)
            else:
                current_block["text"].append(line)
        
        # 保存最后一个块
        if current_block["text"]:
            blocks.append({
                "text": '\n'.join(current_block["text"]).strip(),
                "headings": [h for h in current_block["headings"] if h]
            })
        
        return blocks
    
    def chunk_text(self, content: str, source_file: str) -> List[TextChunk]:
        """
        智能分块：按标题结构分块，同时控制块大小
        
        策略:
        1. 优先按标题分割（保留语义完整性）
        2. 大段落进一步切分（保持overlap）
        3. 每个块保留完整的标题层级上下文
        """
        blocks = self.extract_structure(content)
        chunks = []
        chunk_index = 0
        
        # 使用文件名（含扩展名）作为ID前缀，避免 .md 和 .ipynb 冲突
        file_name = Path(source_file).name.replace('.', '_')
        
        for block in blocks:
            text = block["text"]
            headings = block["headings"]
            
            # 如果块太大，进一步切分
            if len(text) > self.chunk_size * 1.5:
                sub_chunks = self._split_large_block(text, headings)
                for sub_text in sub_chunks:
                    chunk = TextChunk(
                        id=f"{file_name}_{chunk_index:04d}",
                        text=sub_text,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_pos=0,  # 精确位置需要更复杂的计算
                        end_pos=0,
                        headings=headings,
                        metadata={"type": "markdown", "level": len(headings)}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                chunk = TextChunk(
                    id=f"{file_name}_{chunk_index:04d}",
                    text=text,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    start_pos=0,
                    end_pos=len(text),
                    headings=headings,
                    metadata={"type": "markdown", "level": len(headings)}
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _split_large_block(self, text: str, headings: List[str]) -> List[str]:
        """将大文本块切分成多个小块"""
        chunks = []
        start = 0
        
        while start < len(text):
            # 找到合适的结束位置
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            # 尝试在句子边界切分
            search_text = text[start:end]
            # 寻找最后一个句号、问号或换行
            for sep in ['\n\n', '. ', '。', '? ', '？', '! ', '！']:
                pos = search_text.rfind(sep)
                if pos > self.chunk_size * 0.5:  # 至少保留一半内容
                    end = start + pos + len(sep)
                    break
            
            chunks.append(text[start:end].strip())
            start = end - self.overlap  # 重叠区域
        
        return chunks
    
    def process(self, file_path: str) -> ProcessedDocument:
        """处理Markdown文件"""
        logger.info(f"Processing Markdown: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取标题（第一个H1）
        title_match = self.HEADING_PATTERN.search(content)
        title = title_match.group(2) if title_match else Path(file_path).stem
        
        # 分块
        chunks = self.chunk_text(content, file_path)
        
        return ProcessedDocument(
            file_path=file_path,
            file_type='markdown',
            title=title,
            content=content,
            chunks=chunks,
            metadata={
                "processed_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "char_count": len(content)
            }
        )


class NotebookProcessor:
    """Jupyter Notebook处理器"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.md_processor = MarkdownProcessor(chunk_size, overlap)
    
    def convert_to_markdown(self, notebook_path: str, output_path: Optional[str] = None) -> str:
        """
        使用nbconvert将Notebook转换为Markdown
        
        配置:
        - exclude_output=True: 删除运行结果
        - exclude_input_prompt=True: 删除输入提示
        - exclude_output_prompt=True: 删除输出提示
        """
        try:
            from nbconvert import MarkdownExporter
            import nbformat
        except ImportError:
            logger.error("nbconvert or nbformat not installed. Run: pip install nbconvert nbformat")
            raise
        
        logger.info(f"Converting Notebook: {notebook_path}")
        
        # 读取notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # 配置exporter（排除输出）
        exporter = MarkdownExporter(
            exclude_output=True,
            exclude_input_prompt=True,
            exclude_output_prompt=True
        )
        
        # 转换
        markdown_content, resources = exporter.from_notebook_node(notebook)
        
        # 可选：保存转换后的markdown
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"Saved converted markdown to: {output_path}")
        
        return markdown_content
    
    def process(self, file_path: str, output_dir: Optional[str] = None) -> ProcessedDocument:
        """处理Notebook文件"""
        # 转换为markdown
        if output_dir:
            output_path = os.path.join(
                output_dir, 
                Path(file_path).stem + '.md'
            )
        else:
            output_path = None
        
        markdown_content = self.convert_to_markdown(file_path, output_path)
        
        # 使用Markdown处理器处理
        # 创建临时文档对象
        temp_doc = self.md_processor.chunk_text(markdown_content, file_path)
        
        # 为每个chunk添加notebook特有的元数据，并修改ID前缀以区分.ipynb
        for chunk in temp_doc:
            chunk.metadata.update({
                "type": "notebook",
                "original_format": "ipynb"
            })
            # 修改ID前缀：将 _md 替换为 _ipynb
            chunk.id = chunk.id.replace('_md_', '_ipynb_')
        
        return ProcessedDocument(
            file_path=file_path,
            file_type='notebook',
            title=Path(file_path).stem,
            content=markdown_content,
            chunks=temp_doc,
            metadata={
                "processed_at": datetime.now().isoformat(),
                "converted_to_markdown": output_path is not None,
                "chunk_count": len(temp_doc),
                "char_count": len(markdown_content)
            }
        )


class BGEEmbedder:
    """BGE嵌入模型封装"""
    
    MODEL_NAME = "BAAI/bge-large-zh-v1.5"
    OUTPUT_DIM = 1024
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cpu", 
                 cache_dir: Optional[str] = None,
                 trust_remote_code: bool = False):
        self.model_path = model_path or self.MODEL_NAME
        self.device = device
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.model = None
        self._is_local = self._check_is_local_path(self.model_path)
        self._load_model()
    
    def _check_is_local_path(self, path: str) -> bool:
        """检查是否为本地路径"""
        # 检查是否是 HuggingFace Hub ID 格式 (如 "BAAI/bge-large-zh-v1.5")
        if '/' in path and not os.path.isabs(path):
            # 可能是组织/模型名格式，检查是否本地存在
            if os.path.isdir(path):
                return True
            return False
        return os.path.exists(path) and os.path.isdir(path)
    
    def _load_model(self):
        """加载BGE模型"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        
        load_source = "local" if self._is_local else "HuggingFace Hub"
        logger.info(f"Loading BGE model from {load_source}: {self.model_path}")
        logger.info(f"Device: {self.device}")
        
        try:
            # sentence-transformers 会自动处理本地路径和 Hub 路径
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device,
                cache_folder=self.cache_dir,
                trust_remote_code=self.trust_remote_code
            )
            logger.info(f"✅ BGE model loaded successfully from {load_source}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            if self._is_local:
                logger.error("Local model loading failed. Please check:")
                logger.error(f"  1. Path exists: {self.model_path}")
                logger.error(f"  2. Directory contains model files (config.json, pytorch_model.bin, etc.)")
                logger.error(f"  3. Model format is compatible with sentence-transformers")
            raise
    
    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
        
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        
        # BGE模型推荐在查询前添加指令（对检索任务有帮助）
        # 这里我们使用文档嵌入，使用空指令或文档指令
        instruction = ""
        
        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化，便于余弦相似度计算
        )
        
        return embeddings.tolist()
    
    def embed_chunks(self, chunks: List[TextChunk], batch_size: int = 32) -> List[Tuple[TextChunk, List[float]]]:
        """
        为分块生成嵌入
        
        策略：将标题和正文组合，增强语义表示
        """
        texts = []
        for chunk in chunks:
            # 组合标题和正文（标题重要，放在前面）
            heading_text = ' > '.join(chunk.headings) if chunk.headings else ""
            if heading_text:
                text = f"{heading_text}\n{chunk.text}"
            else:
                text = chunk.text
            texts.append(text)
        
        embeddings = self.embed(texts, batch_size=batch_size)
        
        return list(zip(chunks, embeddings))


class ChromaVectorStore:
    """Chroma向量数据库存储"""
    
    COLLECTION_NAME = "course_materials"
    
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        self._init_db()
    
    def _init_db(self):
        """初始化Chroma数据库"""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise
        
        logger.info(f"Initializing ChromaDB at: {self.persist_dir}")
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        logger.info(f"ChromaDB collection ready: {self.COLLECTION_NAME}")
    
    def add_chunks(self, course_name: str, chunks_with_embeddings: List[Tuple[TextChunk, List[float]]]):
        """
        添加分块到向量数据库
        
        Args:
            course_name: 课程名称
            chunks_with_embeddings: (chunk, embedding) 元组列表
        """
        if not chunks_with_embeddings:
            logger.warning("No chunks to add")
            return
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk, embedding in chunks_with_embeddings:
            # 确保ID唯一
            doc_id = f"{course_name}_{chunk.id}"
            ids.append(doc_id)
            embeddings.append(embedding)
            documents.append(chunk.text)
            
            # 元数据
            metadata = {
                "course_name": course_name,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "headings": json.dumps(chunk.headings, ensure_ascii=False),
                **chunk.metadata
            }
            metadatas.append(metadata)
        
        # 批量添加
        logger.info(f"Adding {len(ids)} chunks to ChromaDB")
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info("Chunks added successfully")
    
    def query(self, query_embedding: List[float], course_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索相似文档
        
        Args:
            query_embedding: 查询向量
            course_name: 课程名称（过滤条件）
            top_k: 返回数量
        
        Returns:
            检索结果列表
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"course_name": course_name}
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return formatted_results
    
    def delete_course(self, course_name: str):
        """删除课程的向量数据"""
        self.collection.delete(where={"course_name": course_name})
        logger.info(f"Deleted all data for course: {course_name}")
    
    def get_stats(self, course_name: str) -> Dict[str, int]:
        """获取课程统计信息"""
        all_data = self.collection.get(where={"course_name": course_name})
        return {
            "total_chunks": len(all_data['ids'])
        }


class CourseRAGBuilder:
    """课件RAG构建器"""
    
    SUPPORTED_FORMATS = ['*.md', '*.ipynb', '*.markdown']
    
    def __init__(self, course_name: str, config: Optional[Dict[str, Any]] = None):
        self.course_name = course_name  # 保持原样，课程名大小写敏感
        self.config = config or {}
        
        # 路径配置
        self.materials_dir = Path(f"courses/{course_name}/materials")
        self.output_dir = Path(f"courses/{course_name}/processed_materials")
        self.vector_db_dir = Path(f"vector_db/course_materials/{course_name}")
        
        # 处理参数
        self.chunk_size = self.config.get('chunk_size', 512)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        self.embedding_device = self.config.get('embedding_device', 'cpu')
        
        # 模型配置 - 支持本地路径或 HuggingFace Hub ID
        self.embedding_model_path = self.config.get('embedding_model')
        # 从环境变量获取（优先级低于配置）
        if not self.embedding_model_path:
            self.embedding_model_path = os.environ.get('BGE_MODEL_PATH')
        
        # 处理器实例
        self.md_processor = MarkdownProcessor(self.chunk_size, self.chunk_overlap)
        self.nb_processor = NotebookProcessor(self.chunk_size, self.chunk_overlap)
        
        # 懒加载的组件
        self._embedder: Optional[BGEEmbedder] = None
        self._vector_store: Optional[ChromaVectorStore] = None
    
    @property
    def embedder(self) -> BGEEmbedder:
        """懒加载嵌入模型"""
        if self._embedder is None:
            if self.embedding_model_path:
                logger.info(f"Using custom embedding model: {self.embedding_model_path}")
                self._embedder = BGEEmbedder(
                    model_path=self.embedding_model_path,
                    device=self.embedding_device
                )
            else:
                logger.info(f"Using default embedding model: {BGEEmbedder.MODEL_NAME}")
                self._embedder = BGEEmbedder(device=self.embedding_device)
        return self._embedder
    
    @property
    def vector_store(self) -> ChromaVectorStore:
        """懒加载向量存储"""
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore(str(self.vector_db_dir))
        return self._vector_store
    
    def detect_files(self) -> List[Path]:
        """
        步骤1: 检测输入文件类型
        
        扫描 courses/{course_name}/materials/ 目录下的:
        - *.md, *.markdown - Markdown文件
        - *.ipynb - Jupyter Notebook文件
        
        Returns:
            检测到的文件路径列表
        """
        logger.info(f"Scanning directory: {self.materials_dir}")
        
        if not self.materials_dir.exists():
            raise FileNotFoundError(f"Materials directory not found: {self.materials_dir}")
        
        files = []
        for pattern in self.SUPPORTED_FORMATS:
            matched = list(self.materials_dir.glob(pattern))
            files.extend(matched)
            logger.info(f"  Pattern '{pattern}': {len(matched)} files")
        
        # 去重并排序
        files = sorted(set(files))
        
        logger.info(f"Total files detected: {len(files)}")
        for f in files:
            logger.info(f"  - {f.name}")
        
        return files
    
    def preprocess_notebooks(self, notebook_files: List[Path]) -> List[str]:
        """
        步骤2: Jupyter Notebook预处理
        
        使用nbconvert转换notebook:
        - 仅保留markdown和代码
        - 删除运行结果(output)
        - 删除输入/输出提示
        
        Args:
            notebook_files: Notebook文件路径列表
        
        Returns:
            转换后的markdown文件路径列表
        """
        if not notebook_files:
            return []
        
        logger.info(f"Preprocessing {len(notebook_files)} notebooks")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        converted_files = []
        for nb_file in notebook_files:
            output_path = self.output_dir / f"{nb_file.stem}.md"
            try:
                self.nb_processor.convert_to_markdown(
                    str(nb_file), 
                    str(output_path)
                )
                converted_files.append(str(output_path))
            except Exception as e:
                logger.error(f"Failed to convert {nb_file}: {e}")
        
        return converted_files
    
    def chunk_documents(self, files: List[Path]) -> List[ProcessedDocument]:
        """
        步骤3: 文本分块
        
        处理策略:
        - chunk_size: 512字符
        - overlap: 50字符
        - preserve_structure: 保留标题层级信息
        
        Args:
            files: 要处理的文件路径列表
        
        Returns:
            处理后的文档对象列表
        """
        logger.info(f"Chunking {len(files)} documents")
        
        processed_docs = []
        
        for file_path in files:
            try:
                if file_path.suffix == '.ipynb':
                    # Notebook - 先转换再处理
                    doc = self.nb_processor.process(str(file_path), str(self.output_dir))
                else:
                    # Markdown - 直接处理
                    doc = self.md_processor.process(str(file_path))
                
                processed_docs.append(doc)
                logger.info(f"  {file_path.name}: {len(doc.chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        total_chunks = sum(len(doc.chunks) for doc in processed_docs)
        logger.info(f"Total chunks created: {total_chunks}")
        
        return processed_docs
    
    def embed_chunks(self, documents: List[ProcessedDocument]) -> List[Tuple[TextChunk, List[float]]]:
        """
        步骤4: 生成向量嵌入
        
        模型: BAAI/bge-large-zh-v1.5
        - output_dim: 1024
        - device: cpu/cuda
        - normalize_embeddings: True
        
        Args:
            documents: 处理后的文档列表
        
        Returns:
            (chunk, embedding) 元组列表
        """
        # 收集所有chunks
        all_chunks = []
        for doc in documents:
            all_chunks.extend(doc.chunks)
        
        if not all_chunks:
            logger.warning("No chunks to embed")
            return []
        
        logger.info(f"Embedding {len(all_chunks)} chunks using BGE model")
        
        chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)
        
        logger.info(f"Embedding completed. Dimension: {BGEEmbedder.OUTPUT_DIM}")
        
        return chunks_with_embeddings
    
    def store_vectors(self, chunks_with_embeddings: List[Tuple[TextChunk, List[float]]]):
        """
        步骤5: 存储到向量数据库
        
        使用ChromaDB:
        - 持久化存储
        - 余弦相似度
        - 按course_name过滤
        
        Args:
            chunks_with_embeddings: 带嵌入向量的chunks
        """
        logger.info(f"Storing vectors to ChromaDB: {self.vector_db_dir}")
        
        # 确保目录存在
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        # 强制重新初始化 vector_store，确保状态干净
        self._vector_store = None
        
        # 清除旧的课程数据
        try:
            self.vector_store.delete_course(self.course_name)
            logger.info(f"Deleted old data for course: {self.course_name}")
        except Exception as e:
            logger.warning(f"No old data to delete or delete failed: {e}")
        
        # 添加新数据
        self.vector_store.add_chunks(self.course_name, chunks_with_embeddings)
        
        # 保存元数据
        self._save_metadata(chunks_with_embeddings)
        
        # 统计信息
        stats = self.vector_store.get_stats(self.course_name)
        logger.info(f"Storage completed. Total chunks stored: {stats['total_chunks']}")
    
    def _save_metadata(self, chunks_with_embeddings: List[Tuple[TextChunk, List[float]]]):
        """保存处理后的元数据（原文和章节信息）"""
        metadata_file = self.vector_db_dir / "documents.json"
        
        documents_data = []
        for chunk, _ in chunks_with_embeddings:
            documents_data.append(chunk.to_dict())
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_file}")
    
    def build(self) -> Dict[str, Any]:
        """
        执行完整的RAG构建流程
        
        Returns:
            构建结果统计信息
        """
        logger.info(f"=" * 60)
        logger.info(f"Starting Course RAG Build: {self.course_name}")
        logger.info(f"=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 步骤1: 检测文件
            files = self.detect_files()
            if not files:
                logger.warning("No files found to process")
                return {"status": "skipped", "reason": "no_files"}
            
            # 分离notebook和普通markdown
            notebook_files = [f for f in files if f.suffix == '.ipynb']
            markdown_files = [f for f in files if f.suffix in ['.md', '.markdown']]
            
            # 步骤2: 预处理Notebook
            if notebook_files:
                self.preprocess_notebooks(notebook_files)
            
            # 步骤3: 分块
            all_files = markdown_files + notebook_files
            documents = self.chunk_documents(all_files)
            
            if not documents:
                logger.warning("No documents processed successfully")
                return {"status": "failed", "reason": "processing_failed"}
            
            # 步骤4: 生成嵌入
            chunks_with_embeddings = self.embed_chunks(documents)
            
            # 步骤5: 存储
            self.store_vectors(chunks_with_embeddings)
            
            # 统计
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                "status": "success",
                "course_name": self.course_name,
                "files_processed": len(documents),
                "total_chunks": len(chunks_with_embeddings),
                "vector_dim": BGEEmbedder.OUTPUT_DIM,
                "duration_seconds": duration,
                "processed_at": end_time.isoformat()
            }
            
            # 保存构建信息
            info_file = self.vector_db_dir / "metadata.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"=" * 60)
            logger.info(f"Build Completed Successfully!")
            logger.info(f"  Files processed: {result['files_processed']}")
            logger.info(f"  Total chunks: {result['total_chunks']}")
            logger.info(f"  Duration: {duration:.2f}s")
            logger.info(f"=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"Build failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        测试查询功能
        
        Args:
            query_text: 查询文本
            top_k: 返回数量
        
        Returns:
            检索结果
        """
        logger.info(f"Querying: '{query_text}'")
        
        # 生成查询向量
        query_embedding = self.embedder.embed([query_text], show_progress=False)[0]
        
        # 检索
        results = self.vector_store.query(query_embedding, self.course_name, top_k)
        
        logger.info(f"Found {len(results)} results")
        for i, r in enumerate(results[:3]):
            logger.info(f"  {i+1}. [{r['distance']:.4f}] {r['text'][:80]}...")
        
        return results


class CourseRAGQueryInterface:
    """
    课件RAG查询接口
    
    供其他模块使用的统一查询接口，不需要重新构建索引。
    
    使用示例:
        ```python
        # LLM校对模块使用
        from src.pipeline.course_rag import CourseRAGQueryInterface
        
        query_interface = CourseRAGQueryInterface("A3.11_autogen")
        
        # 检索相关内容
        results = query_interface.search("PyTorch神经网络", top_k=3)
        
        # 格式化供LLM使用
        context = query_interface.format_for_llm(results)
        ```
    """
    
    def __init__(self, course_name: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化查询接口
        
        Args:
            course_name: 课程名称（大小写敏感，需与目录名一致）
            config: 配置字典
                - embedding_model: 模型路径 (默认使用环境变量或默认路径)
                - embedding_device: cpu/cuda
        """
        self.course_name = course_name  # 保持原样，课程名大小写敏感
        self.config = config or {}
        
        # 路径
        self.vector_db_dir = Path(f"vector_db/course_materials/{course_name}")
        
        # 模型配置
        self.embedding_model_path = self.config.get('embedding_model') or os.environ.get('BGE_MODEL_PATH', 'models/bge-large-zh-v1.5')
        self.embedding_device = self.config.get('embedding_device', 'cpu')
        
        # 懒加载组件
        self._embedder: Optional[BGEEmbedder] = None
        self._vector_store: Optional[ChromaVectorStore] = None
        
        # 验证课程是否存在
        if not self.exists():
            logger.warning(f"Course RAG not found: {course_name}. Please build it first.")
    
    @property
    def embedder(self) -> BGEEmbedder:
        """懒加载嵌入模型"""
        if self._embedder is None:
            self._embedder = BGEEmbedder(
                model_path=self.embedding_model_path,
                device=self.embedding_device
            )
        return self._embedder
    
    @property
    def vector_store(self) -> ChromaVectorStore:
        """懒加载向量存储"""
        if self._vector_store is None:
            if not self.vector_db_dir.exists():
                raise FileNotFoundError(f"Vector DB not found: {self.vector_db_dir}")
            self._vector_store = ChromaVectorStore(str(self.vector_db_dir))
        return self._vector_store
    
    def exists(self) -> bool:
        """检查课程RAG是否已构建"""
        metadata_file = self.vector_db_dir / "metadata.json"
        return metadata_file.exists()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取课程统计信息"""
        if not self.exists():
            return {"exists": False}
        
        metadata_file = self.vector_db_dir / "metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        stats = self.vector_store.get_stats(self.course_name)
        return {
            "exists": True,
            "course_name": self.course_name,
            "total_chunks": stats["total_chunks"],
            **metadata
        }
    
    def search(self, query_text: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        语义搜索课程内容
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件 (暂不支持，保留接口)
        
        Returns:
            检索结果列表，每项包含:
                - id: 片段ID
                - text: 文本内容
                - metadata: 元数据 (source_file, headings等)
                - distance: 距离 (越小越相似)
                - similarity: 相似度 (0-1，越大越相似)
        """
        if not self.exists():
            logger.error(f"Cannot search: Course RAG not found: {self.course_name}")
            return []
        
        logger.debug(f"Searching course '{self.course_name}': '{query_text}'")
        
        # 生成查询向量
        query_embedding = self.embedder.embed([query_text], show_progress=False)[0]
        
        # 检索
        results = self.vector_store.query(query_embedding, self.course_name, top_k)
        
        # 添加相似度分数 (1 - distance，因为使用余弦距离)
        for r in results:
            r['similarity'] = 1 - r['distance']
        
        return results
    
    def search_by_heading(self, heading_keyword: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        按标题关键词搜索
        
        用于查找特定章节的内容。
        
        Args:
            heading_keyword: 标题关键词
            top_k: 返回数量
        
        Returns:
            包含该标题的片段列表
        """
        # 获取所有数据
        all_data = self.vector_store.collection.get(
            where={"course_name": self.course_name}
        )
        
        results = []
        for i, metadata in enumerate(all_data['metadatas']):
            headings_str = metadata.get('headings', '[]')
            try:
                headings = json.loads(headings_str)
                if any(heading_keyword.lower() in h.lower() for h in headings):
                    results.append({
                        'id': all_data['ids'][i],
                        'text': all_data['documents'][i],
                        'metadata': metadata,
                        'headings': headings
                    })
            except json.JSONDecodeError:
                continue
        
        return results[:top_k]
    
    def get_document_outline(self) -> List[Dict[str, Any]]:
        """
        获取课程文档大纲
        
        Returns:
            章节结构列表
        """
        documents_file = self.vector_db_dir / "documents.json"
        if not documents_file.exists():
            return []
        
        with open(documents_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # 提取唯一的标题层级
        outline = []
        seen_headings = set()
        
        for doc in documents:
            headings = doc.get('headings', [])
            if headings:
                heading_key = ' > '.join(headings)
                if heading_key not in seen_headings:
                    seen_headings.add(heading_key)
                    outline.append({
                        'level': len(headings),
                        'title': headings[-1],
                        'full_path': headings,
                        'source_file': doc.get('source_file', '')
                    })
        
        return outline
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取特定片段
        
        Args:
            chunk_id: 片段ID (格式: {course_name}_{source}_{index})
        
        Returns:
            片段信息
        """
        full_id = f"{self.course_name}_{chunk_id}"
        
        try:
            result = self.vector_store.collection.get(ids=[full_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
        
        return None
    
    def format_for_llm(self, search_results: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """
        将搜索结果格式化为LLM可用的上下文
        
        Args:
            search_results: search()返回的结果
            max_length: 最大上下文长度
        
        Returns:
            格式化后的上下文字符串
        """
        if not search_results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results, 1):
            text = result['text']
            headings = result['metadata'].get('headings', '[]')
            source = result['metadata'].get('source_file', 'unknown')
            similarity = result.get('similarity', 0)
            
            try:
                headings_list = json.loads(headings) if isinstance(headings, str) else headings
                heading_str = ' > '.join(headings_list) if headings_list else '无标题'
            except:
                heading_str = str(headings)
            
            part = f"""
【参考{i}】(相关度: {similarity:.2%})
来源: {source}
章节: {heading_str}
内容: {text}
"""
            
            if current_length + len(part) > max_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "\n---".join(context_parts)
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量搜索多个查询
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回数量
        
        Returns:
            {query: results} 字典
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k=top_k)
        return results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Course RAG Builder")
    parser.add_argument("--course", required=True, help="Course name")
    parser.add_argument("--device", default="cpu", help="Embedding device (cpu/cuda)")
    parser.add_argument("--query", help="Test query after build")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 构建配置
    config = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_device": args.device
    }
    
    # 执行构建
    builder = CourseRAGBuilder(args.course, config)
    result = builder.build()
    
    if result["status"] == "success" and args.query:
        print("\n" + "=" * 60)
        print("Testing Query:")
        builder.query(args.query)


if __name__ == "__main__":
    main()
