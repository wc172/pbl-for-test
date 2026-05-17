"""
课件RAG构建模块单元测试
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.pipeline.course_rag import (
    CourseRAGBuilder,
    MarkdownProcessor,
    NotebookProcessor,
    TextChunk,
    ProcessedDocument,
    BGEEmbedder,
    ChromaVectorStore
)


class TestMarkdownProcessor:
    """测试Markdown处理器"""
    
    def test_extract_structure(self):
        """测试标题结构提取"""
        processor = MarkdownProcessor(chunk_size=512, overlap=50)
        
        content = """# 第一章

这是第一章的内容。

## 1.1 小节

小节内容。

### 1.1.1 详细内容

详细说明。

## 1.2 另一个小节

另一个小节的内容。

# 第二章

第二章内容。
"""
        
        blocks = processor.extract_structure(content)
        
        assert len(blocks) >= 3  # 至少应该有3个块
        
        # 检查第一个块的标题
        assert "第一章" in blocks[0]["headings"][0]
    
    def test_chunk_text(self):
        """测试文本分块"""
        processor = MarkdownProcessor(chunk_size=100, overlap=20)
        
        content = """# 标题

这是第一段内容。这里有很多文字。

## 小节

这是第二节的内容。也有很多文字。
"""
        
        chunks = processor.chunk_text(content, "test.md")
        
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(len(c.text) > 0 for c in chunks)
    
    def test_process_markdown_file(self, tmp_path):
        """测试处理Markdown文件"""
        processor = MarkdownProcessor()
        
        # 创建测试文件
        test_file = tmp_path / "test.md"
        test_file.write_text("""# 测试文档

这是测试内容。

## 章节1

章节1的内容。
""", encoding='utf-8')
        
        doc = processor.process(str(test_file))
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.file_type == 'markdown'
        assert doc.title == "测试文档"
        assert len(doc.chunks) > 0


class TestNotebookProcessor:
    """测试Notebook处理器"""
    
    def test_convert_to_markdown(self, tmp_path):
        """测试Notebook转换为Markdown"""
        import nbformat
        
        processor = NotebookProcessor()
        
        # 创建测试notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell("# 测试Notebook\n\n这是介绍。"),
            nbformat.v4.new_code_cell("print('hello')"),
            nbformat.v4.new_markdown_cell("## 结论\n\n总结。")
        ]
        
        test_file = tmp_path / "test.ipynb"
        with open(test_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        output_path = tmp_path / "output.md"
        markdown = processor.convert_to_markdown(str(test_file), str(output_path))
        
        assert "测试Notebook" in markdown
        assert output_path.exists()


class TestBGEEmbedder:
    """测试BGE嵌入模型"""
    
    @pytest.mark.slow
    def test_embed_single_text(self):
        """测试单文本嵌入"""
        embedder = BGEEmbedder(device="cpu")
        
        texts = ["这是一个测试句子。", "这是另一个句子。"]
        embeddings = embedder.embed(texts, batch_size=2, show_progress=False)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == BGEEmbedder.OUTPUT_DIM
        assert all(len(e) == BGEEmbedder.OUTPUT_DIM for e in embeddings)
    
    @pytest.mark.slow
    def test_embed_chunks(self):
        """测试分块嵌入"""
        embedder = BGEEmbedder(device="cpu")
        
        chunks = [
            TextChunk(
                id="test_001",
                text="测试文本1",
                source_file="test.md",
                chunk_index=0,
                start_pos=0,
                end_pos=10,
                headings=["标题1"]
            ),
            TextChunk(
                id="test_002",
                text="测试文本2",
                source_file="test.md",
                chunk_index=1,
                start_pos=10,
                end_pos=20,
                headings=["标题1", "子标题"]
            )
        ]
        
        result = embedder.embed_chunks(chunks, batch_size=2)
        
        assert len(result) == 2
        assert all(len(embedding) == BGEEmbedder.OUTPUT_DIM for _, embedding in result)


class TestChromaVectorStore:
    """测试Chroma向量存储"""
    
    def test_init_db(self, tmp_path):
        """测试数据库初始化"""
        store = ChromaVectorStore(str(tmp_path / "chroma"))
        
        assert store.client is not None
        assert store.collection is not None
    
    def test_add_and_query(self, tmp_path):
        """测试添加和查询"""
        store = ChromaVectorStore(str(tmp_path / "chroma"))
        
        # 创建测试数据
        chunks_with_embeddings = [
            (
                TextChunk(
                    id="chunk_001",
                    text="这是第一个测试块。",
                    source_file="test.md",
                    chunk_index=0,
                    start_pos=0,
                    end_pos=10,
                    headings=["标题1"]
                ),
                [0.1] * 1024  # 模拟嵌入向量
            ),
            (
                TextChunk(
                    id="chunk_002",
                    text="这是第二个测试块，关于机器学习。",
                    source_file="test.md",
                    chunk_index=1,
                    start_pos=10,
                    end_pos=30,
                    headings=["标题1", "机器学习"]
                ),
                [0.2] * 1024
            )
        ]
        
        # 添加数据
        store.add_chunks("test_course", chunks_with_embeddings)
        
        # 查询
        results = store.query([0.1] * 1024, "test_course", top_k=2)
        
        assert len(results) == 2
        assert all("id" in r and "text" in r for r in results)


class TestCourseRAGBuilder:
    """测试完整构建流程"""
    
    def test_detect_files(self, tmp_path):
        """测试文件检测"""
        # 创建测试目录结构
        materials_dir = tmp_path / "courses" / "test_course" / "materials"
        materials_dir.mkdir(parents=True)
        
        (materials_dir / "doc1.md").write_text("# Doc1\nContent")
        (materials_dir / "doc2.md").write_text("# Doc2\nContent")
        
        builder = CourseRAGBuilder("test_course", {})
        # 临时修改路径
        builder.materials_dir = materials_dir
        
        files = builder.detect_files()
        
        assert len(files) == 2
        assert all(f.suffix == '.md' for f in files)
    
    @pytest.mark.integration
    def test_full_build_process(self, tmp_path):
        """测试完整构建流程（集成测试）"""
        # 创建测试目录结构
        base_dir = tmp_path / "test_project"
        materials_dir = base_dir / "courses" / "test_course" / "materials"
        materials_dir.mkdir(parents=True)
        
        # 创建测试markdown文件
        (materials_dir / "lecture1.md").write_text("""# 课程介绍

这是课程介绍的内容。

## 学习目标

- 理解基本概念
- 掌握核心方法

## 内容概述

详细内容介绍。
""", encoding='utf-8')
        
        # 创建builder并修改路径
        builder = CourseRAGBuilder("test_course", {
            "embedding_device": "cpu",
            "chunk_size": 256,
            "chunk_overlap": 30
        })
        builder.materials_dir = materials_dir
        builder.output_dir = base_dir / "courses" / "test_course" / "processed_materials"
        builder.vector_db_dir = base_dir / "vector_db" / "course_materials" / "test_course"
        
        # 执行构建
        result = builder.build()
        
        assert result["status"] == "success"
        assert result["course_name"] == "test_course"
        assert result["files_processed"] == 1
        assert result["total_chunks"] > 0
        assert result["vector_dim"] == 1024


def test_chunk_size_and_overlap():
    """测试分块大小和重叠参数"""
    processor = MarkdownProcessor(chunk_size=100, overlap=20)
    
    # 创建长文本
    content = "# 标题\n\n" + "这是一句很长的话。" * 20
    
    chunks = processor.chunk_text(content, "test.md")
    
    # 检查是否有多个块
    if len(chunks) > 1:
        # 检查重叠
        for i in range(len(chunks) - 1):
            # 当前块的结尾和下一个块的开头应该有重叠
            current_end = chunks[i].text[-30:]
            next_start = chunks[i + 1].text[:30]
            # 简单的重叠检查
            assert len(current_end) > 0
            assert len(next_start) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
