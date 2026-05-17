"""
课件RAG构建示例

这个脚本演示如何使用 CourseRAGBuilder 构建课程向量索引。

用法:
    python examples/build_course_rag.py --course A3.11_autogen
    python examples/build_course_rag.py --course A3.11_autogen --query "AutoGen是什么"
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.course_rag import CourseRAGBuilder


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="构建课程RAG索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python examples/build_course_rag.py --course A3.11_autogen
    python examples/build_course_rag.py --course A3.11_autogen --query "AutoGen是什么"
    python examples/build_course_rag.py --course A3.11_autogen --device cuda
        """
    )
    
    parser.add_argument(
        "--course", 
        required=True,
        help="课程名称（对应 courses/{course}/materials/ 目录）"
    )
    parser.add_argument(
        "--device", 
        default="cpu",
        choices=["cpu", "cuda"],
        help="嵌入模型设备 (默认: cpu)"
    )
    parser.add_argument(
        "--model-path",
        default="models/bge-large-zh-v1.5",
        help="本地模型路径 (默认: models/bge-large-zh-v1.5)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=512,
        help="分块大小 (默认: 512)"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=50,
        help="分块重叠 (默认: 50)"
    )
    parser.add_argument(
        "--query", 
        help="构建后执行测试查询"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/rag_build_{args.course}.log", encoding='utf-8')
        ]
    )
    
    print(f"{'='*60}")
    print(f"课程RAG构建工具")
    print(f"{'='*60}")
    print(f"课程名称: {args.course}")
    print(f"设备: {args.device}")
    print(f"模型路径: {args.model_path}")
    print(f"分块大小: {args.chunk_size}")
    print(f"分块重叠: {args.chunk_overlap}")
    print(f"{'='*60}\n")
    
    # 配置
    config = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_device": args.device,
        "embedding_model": args.model_path
    }
    
    # 创建builder
    builder = CourseRAGBuilder(args.course, config)
    
    # 执行构建
    result = builder.build()
    
    if result["status"] == "success":
        print(f"\n{'='*60}")
        print("✅ 构建成功!")
        print(f"{'='*60}")
        print(f"处理文件数: {result['files_processed']}")
        print(f"总分块数: {result['total_chunks']}")
        print(f"向量维度: {result['vector_dim']}")
        print(f"耗时: {result['duration_seconds']:.2f}秒")
        print(f"向量数据库: {builder.vector_db_dir}")
        print(f"{'='*60}\n")
        
        # 执行测试查询
        if args.query:
            print(f"\n{'='*60}")
            print(f"测试查询: '{args.query}'")
            print(f"{'='*60}")
            
            results = builder.query(args.query, top_k=5)
            
            print(f"\n找到 {len(results)} 个相关片段:\n")
            for i, r in enumerate(results, 1):
                distance = r['distance']
                text = r['text'][:150].replace('\n', ' ')
                source = r['metadata'].get('source_file', 'unknown')
                print(f"{i}. [相似度: {1-distance:.4f}] [{source}]")
                print(f"   {text}...")
                print()
    else:
        print(f"\n❌ 构建失败: {result.get('error', '未知错误')}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
