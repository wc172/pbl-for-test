"""
命令行接口 (CLI)

提供命令行工具用于执行流水线任务。
"""

import argparse
import sys
from pathlib import Path


def cmd_build_course_rag(args):
    """命令: 构建课件RAG"""
    # TODO: 实现构建RAG逻辑
    print(f"Building course RAG for: {args.course}")


def cmd_process(args):
    """命令: 完整处理流程"""
    # TODO: 实现完整处理
    print(f"Processing course: {args.course}")


def cmd_step0(args):
    """命令: 步骤0 - 课件RAG"""
    print(f"Step 0: Build course RAG for {args.course}")


def cmd_step1(args):
    """命令: 步骤1 - 预处理"""
    print(f"Step 1: Preprocess for {args.course}")


def cmd_step2(args):
    """命令: 步骤2 - 转录"""
    print(f"Step 2: Transcribe for {args.course}")


def cmd_step3(args):
    """命令: 步骤3 - 校对"""
    print(f"Step 3: Correct for {args.course}")


def cmd_step4(args):
    """命令: 步骤4 - 摘要"""
    print(f"Step 4: Summarize for {args.course}")


def cmd_query(args):
    """命令: 查询"""
    print(f"Querying course {args.course}: {args.query}")


def main():
    parser = argparse.ArgumentParser(
        description="视频转录+RAG系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 构建课件RAG
  python -m src.cli build-course-rag --course my_course
  
  # 完整处理
  python -m src.cli process --course my_course
  
  # 分步执行
  python -m src.cli step0 --course my_course
  python -m src.cli step1 --course my_course
  
  # 查询
  python -m src.cli query --course my_course "老师在第5分钟讲了什么"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # build-course-rag 命令
    parser_build = subparsers.add_parser("build-course-rag", help="构建课件RAG")
    parser_build.add_argument("--course", required=True, help="课程名称")
    parser_build.set_defaults(func=cmd_build_course_rag)
    
    # process 命令
    parser_process = subparsers.add_parser("process", help="完整处理流程")
    parser_process.add_argument("--course", required=True, help="课程名称")
    parser_process.set_defaults(func=cmd_process)
    
    # 分步命令
    for i, (name, desc) in enumerate([
        ("step0", "步骤0: 课件RAG构建"),
        ("step1", "步骤1: 输入预处理"),
        ("step2", "步骤2: 离线转录"),
        ("step3", "步骤3: LLM校对"),
        ("step4", "步骤4: 多级摘要"),
    ]):
        p = subparsers.add_parser(name, help=desc)
        p.add_argument("--course", required=True, help="课程名称")
        p.set_defaults(func=[cmd_step0, cmd_step1, cmd_step2, cmd_step3, cmd_step4][i])
    
    # query 命令
    parser_query = subparsers.add_parser("query", help="查询课程内容")
    parser_query.add_argument("--course", required=True, help="课程名称")
    parser_query.add_argument("query", help="查询内容")
    parser_query.set_defaults(func=cmd_query)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
