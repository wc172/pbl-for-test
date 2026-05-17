"""
课件RAG模块快速测试脚本

用法:
    python test_rag_quick.py
"""

import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data():
    """创建测试数据"""
    course_name = "test_course"
    materials_dir = Path(f"courses/{course_name}/materials")
    materials_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试Markdown文件
    test_content = """# 深度学习基础

本文介绍深度学习的基本概念和应用。

## 神经网络

神经网络是深度学习的核心组件，模拟生物神经元的工作方式。

### 前馈神经网络

前馈神经网络（Feedforward Neural Network）是最基本的网络结构，信息单向传播。

### 卷积神经网络

卷积神经网络（CNN）特别适用于图像处理任务，具有局部连接和权值共享的特点。

## 优化算法

训练神经网络需要使用优化算法来最小化损失函数。

### 梯度下降法

梯度下降是训练神经网络的基础算法，通过计算损失函数对参数的梯度来更新参数。

公式：$\\theta = \\theta - \\alpha \\nabla J(\\theta)$

### Adam优化器

Adam（Adaptive Moment Estimation）是一种自适应学习率优化算法，结合了动量和RMSprop的优点。

# PyTorch入门

PyTorch是Facebook开发的深度学习框架，以其动态计算图和易用性而闻名。

## 张量操作

张量（Tensor）是PyTorch的基本数据结构，类似于NumPy的ndarray，但可以在GPU上运行。

```python
import torch
x = torch.randn(3, 3)
```

## 自动求导

autograd是PyTorch的自动微分引擎，可以自动计算梯度。

# 自然语言处理

NLP是人工智能的重要分支，研究计算机如何理解和生成人类语言。

## Transformer架构

Transformer是2017年提出的革命性架构，基于自注意力机制，成为现代NLP的基础。

## 大语言模型

GPT、BERT等大语言模型在多种NLP任务上取得了突破性进展。
"""
    
    test_file = materials_dir / "deep_learning_intro.md"
    test_file.write_text(test_content, encoding='utf-8')
    
    logger.info(f"✅ 测试数据已创建: {test_file}")
    return course_name


def test_module(course_name: str):
    """测试RAG模块"""
    from src.pipeline.course_rag import CourseRAGBuilder
    
    logger.info("=" * 60)
    logger.info("开始测试课件RAG模块")
    logger.info("=" * 60)
    
    # 配置使用本地模型
    config = {
        "embedding_model": "models/bge-large-zh-v1.5",
        "embedding_device": "cpu",
        "chunk_size": 512,
        "chunk_overlap": 50
    }
    
    # 初始化builder
    builder = CourseRAGBuilder(course_name, config)
    
    # 步骤1: 检测文件
    logger.info("\n[步骤1] 检测文件...")
    files = builder.detect_files()
    logger.info(f"  检测到 {len(files)} 个文件")
    
    # 完整构建
    logger.info("\n[步骤2] 执行完整构建流程...")
    result = builder.build()
    
    if result["status"] != "success":
        logger.error(f"❌ 构建失败: {result}")
        return False
    
    logger.info(f"✅ 构建成功!")
    logger.info(f"  处理文件数: {result['files_processed']}")
    logger.info(f"  总分块数: {result['total_chunks']}")
    logger.info(f"  耗时: {result['duration_seconds']:.2f}秒")
    
    # 测试查询
    logger.info("\n[步骤3] 测试查询功能...")
    
    test_queries = [
        "什么是神经网络",
        "PyTorch的张量",
        "梯度下降算法",
        "Transformer架构"
    ]
    
    for query in test_queries:
        logger.info(f"\n  查询: '{query}'")
        results = builder.query(query, top_k=2)
        if results:
            logger.info(f"  找到 {len(results)} 个结果:")
            for i, r in enumerate(results[:2], 1):
                similarity = 1 - r['distance']
                text = r['text'][:60].replace('\n', ' ')
                logger.info(f"    {i}. [{similarity:.4f}] {text}...")
        else:
            logger.warning("  未找到结果")
    
    return True


def verify_output(course_name: str):
    """验证输出"""
    logger.info("\n" + "=" * 60)
    logger.info("验证输出文件")
    logger.info("=" * 60)
    
    # 检查向量数据库
    vector_db_dir = Path(f"vector_db/course_materials/{course_name}")
    if vector_db_dir.exists():
        logger.info(f"✅ 向量数据库目录存在: {vector_db_dir}")
        files = list(vector_db_dir.glob("*"))
        logger.info(f"   包含文件: {[f.name for f in files]}")
    else:
        logger.error(f"❌ 向量数据库目录不存在: {vector_db_dir}")
    
    # 检查metadata
    metadata_file = vector_db_dir / "metadata.json"
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"✅ 元数据文件存在，包含 {metadata.get('total_chunks', 0)} 个分块")
    
    # 检查处理后的文件
    processed_dir = Path(f"courses/{course_name}/processed_materials")
    if processed_dir.exists():
        logger.info(f"✅ 处理后文件目录存在: {processed_dir}")


def cleanup(course_name: str):
    """清理测试数据"""
    import shutil
    
    logger.info("\n" + "=" * 60)
    logger.info("清理测试数据")
    logger.info("=" * 60)
    
    paths_to_remove = [
        f"courses/{course_name}",
        f"vector_db/course_materials/{course_name}",
        f".cache/{course_name}"
    ]
    
    for path in paths_to_remove:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            logger.info(f"  已删除: {path}")
    
    logger.info("✅ 清理完成")


def main():
    course_name = "test_course"
    
    try:
        # 1. 创建测试数据
        create_test_data()
        
        # 2. 测试模块
        success = test_module(course_name)
        
        if success:
            # 3. 验证输出
            verify_output(course_name)
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 所有测试通过!")
            logger.info("=" * 60)
            
            # 询问是否清理
            response = input("\n是否清理测试数据? (y/n): ").strip().lower()
            if response == 'y':
                cleanup(course_name)
            else:
                logger.info("保留测试数据，可用于进一步测试")
                logger.info(f"课程路径: courses/{course_name}/")
                logger.info(f"向量数据库: vector_db/course_materials/{course_name}/")
        else:
            logger.error("❌ 测试失败")
            return 1
            
    except Exception as e:
        logger.exception("测试过程中发生错误")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
