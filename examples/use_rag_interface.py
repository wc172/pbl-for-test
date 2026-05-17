"""
课件RAG查询接口使用示例

演示其他模块如何使用 CourseRAGQueryInterface 查询课件内容。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.course_rag import CourseRAGQueryInterface


def demo_basic_search():
    """基本搜索示例"""
    print("=" * 60)
    print("示例1: 基本语义搜索")
    print("=" * 60)
    
    # 初始化查询接口
    query_interface = CourseRAGQueryInterface("A3.11_autogen")
    
    # 检查是否存在
    if not query_interface.exists():
        print("❌ 课程RAG不存在，请先构建: python examples/build_course_rag.py --course A3.11_autogen")
        return
    
    # 搜索
    results = query_interface.search("AutoGen是什么", top_k=3)
    
    print(f"\n找到 {len(results)} 个结果:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [相似度: {r['similarity']:.2%}]")
        print(f"   来源: {r['metadata'].get('source_file', 'unknown')}")
        print(f"   内容: {r['text'][:100]}...")
        print()


def demo_format_for_llm():
    """格式化给LLM使用"""
    print("=" * 60)
    print("示例2: 格式化上下文给LLM")
    print("=" * 60)
    
    query_interface = CourseRAGQueryInterface("A3.11_autogen")
    
    if not query_interface.exists():
        print("❌ 课程RAG不存在")
        return
    
    # 搜索
    query = "神经网络"
    results = query_interface.search(query, top_k=3)
    
    # 格式化为LLM上下文
    context = query_interface.format_for_llm(results, max_length=1500)
    
    print(f"\n查询: {query}")
    print(f"\n格式化后的上下文:\n")
    print(context)
    print("\n" + "=" * 60)
    print("提示: 这个上下文可以直接拼接到LLM的prompt中")
    print("=" * 60)


def demo_get_outline():
    """获取文档大纲"""
    print("=" * 60)
    print("示例3: 获取课程大纲")
    print("=" * 60)
    
    query_interface = CourseRAGQueryInterface("A3.11_autogen")
    
    if not query_interface.exists():
        print("❌ 课程RAG不存在")
        return
    
    outline = query_interface.get_document_outline()
    
    print(f"\n课程包含 {len(outline)} 个章节:\n")
    for item in outline[:10]:  # 只显示前10个
        indent = "  " * (item['level'] - 1)
        print(f"{indent}- {item['title']}")
    
    if len(outline) > 10:
        print(f"\n... 还有 {len(outline) - 10} 个章节")


def demo_search_by_heading():
    """按标题搜索"""
    print("=" * 60)
    print("示例4: 按标题关键词搜索")
    print("=" * 60)
    
    query_interface = CourseRAGQueryInterface("A3.11_autogen")
    
    if not query_interface.exists():
        print("❌ 课程RAG不存在")
        return
    
    # 搜索包含"Agent"的章节
    results = query_interface.search_by_heading("Agent", top_k=5)
    
    print(f"\n找到 {len(results)} 个包含'Agent'标题的片段:\n")
    for i, r in enumerate(results, 1):
        headings = r.get('headings', [])
        print(f"{i}. {' > '.join(headings)}")
        print(f"   {r['text'][:80]}...")
        print()


def demo_batch_search():
    """批量搜索"""
    print("=" * 60)
    print("示例5: 批量搜索（用于LLM校对多个术语）")
    print("=" * 60)
    
    query_interface = CourseRAGQueryInterface("A3.11_autogen")
    
    if not query_interface.exists():
        print("❌ 课程RAG不存在")
        return
    
    # 批量查询（例如校对用户提到的多个术语）
    terms = ["ConversableAgent", "GroupChat", "UserProxyAgent"]
    
    results = query_interface.batch_search(terms, top_k=2)
    
    for term, term_results in results.items():
        print(f"\n术语 '{term}':")
        if term_results:
            for r in term_results:
                print(f"  - [{r['similarity']:.2%}] {r['text'][:60]}...")
        else:
            print("  (未找到相关内容)")


def demo_llm_correction_workflow():
    """LLM校对工作流示例"""
    print("=" * 60)
    print("示例6: LLM校对工作流（模块3使用场景）")
    print("=" * 60)
    
    query_interface = CourseRAGQueryInterface("A3.11_autogen")
    
    if not query_interface.exists():
        print("❌ 课程RAG不存在")
        return
    
    # 模拟转录文本（可能包含错误）
    transcribed_text = "奥托肯是一个多智能体框架"
    
    print(f"\n转录文本: '{transcribed_text}'")
    print("怀疑'奥托肯'可能是术语错误，搜索课件...")
    
    # 搜索相关内容
    results = query_interface.search("AutoGen框架是什么", top_k=3)
    
    # 格式化为LLM上下文
    context = query_interface.format_for_llm(results)
    
    # 构建prompt（实际使用时发给LLM）
    prompt = f"""根据以下课件内容，判断"奥托肯"应该更正为什么术语：

{context}

待校对的文本: "奥托肯是一个多智能体框架"

请输出更正后的文本。"""
    
    print("\n生成的Prompt（片段）:")
    print("-" * 60)
    print(prompt[:500] + "...")
    print("-" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="课件RAG查询接口示例")
    parser.add_argument("--demo", type=int, default=0, help="运行指定示例 (1-6), 0=运行所有")
    
    args = parser.parse_args()
    
    demos = [
        (1, demo_basic_search, "基本语义搜索"),
        (2, demo_format_for_llm, "格式化给LLM"),
        (3, demo_get_outline, "获取课程大纲"),
        (4, demo_search_by_heading, "按标题搜索"),
        (5, demo_batch_search, "批量搜索"),
        (6, demo_llm_correction_workflow, "LLM校对工作流"),
    ]
    
    if args.demo == 0:
        # 运行所有示例
        for num, func, name in demos:
            print("\n")
            func()
            input("\n按回车继续下一个示例...")
    elif 1 <= args.demo <= 6:
        # 运行指定示例
        demos[args.demo - 1][1]()
    else:
        print(f"无效的示例编号: {args.demo}")
        print(f"可用示例: 1-{len(demos)}")


if __name__ == "__main__":
    main()
