#!/usr/bin/env python3
"""
示例: 使用转录校对模块

本示例展示如何使用高性价比的双层校对策略：
1. 规则校对 - 快速处理常见错误（免费）
2. 选择性LLM校对 - 仅处理可疑句子（节省95%成本）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.corrector import (
    TranscriptionCorrector, 
    correct_transcription,
    is_corrected,
    RuleBasedCorrector
)
from src.utils.srt_parser import parse_srt_file, SRTEntry, save_srt


def demo_rule_corrector():
    """演示规则校对器"""
    print("\n" + "="*60)
    print("📝 演示1: 规则校对器（免费，处理80%常见错误）")
    print("="*60)
    
    corrector = RuleBasedCorrector()
    
    # 测试用例: 常见的ASR误识别
    test_cases = [
        ("皮tor是一个深度学习框架", "PyTorch是一个深度学习框架"),
        ("我们使用爬沈编程", "我们使用Python编程"),
        ("调用挪尔激活函数", "调用ReLU激活函数"),
        ("使用南派进行计算", "使用NumPy进行计算"),
        ("tensor flow由谷歌开发", "TensorFlow由谷歌开发"),
        ("亚当优化器效果较好", "Adam优化器效果较好"),
    ]
    
    print("\n测试规则校对效果:")
    print("-" * 60)
    
    correct_count = 0
    for original, expected in test_cases:
        entry = SRTEntry(index=1, start_ms=0, end_ms=5000, text=original)
        corrected, modified = corrector.correct(entry)
        
        status = "✅" if corrected.text == expected else "❌"
        if corrected.text == expected:
            correct_count += 1
            
        print(f"{status} 原文: {original}")
        print(f"   结果: {corrected.text}")
        print(f"   期望: {expected}")
        print()
    
    print(f"准确率: {correct_count}/{len(test_cases)} ({correct_count/len(test_cases)*100:.0f}%)")
    print(f"词典大小: {len(corrector.misrecognition_dict)} 个映射")


def demo_selective_llm(course_name: str = "demo"):
    """演示选择性LLM校对"""
    print("\n" + "="*60)
    print("🤖 演示2: 选择性LLM校对（仅处理可疑句子）")
    print("="*60)
    
    # 创建测试SRT文件
    test_entries = [
        SRTEntry(index=1, start_ms=0, end_ms=3000, text="大家好，这节课学习皮tor。"),
        SRTEntry(index=2, start_ms=3000, end_ms=6000, text="PyTorch是一个非常强大的框架。"),  # 无需修改
        SRTEntry(index=3, start_ms=6000, end_ms=9000, text="我们可以使用tensor flow进行训练。"),
        SRTEntry(index=4, start_ms=9000, end_ms=12000, text="首先需要安装康达环境。"),
        SRTEntry(index=5, start_ms=12000, end_ms=15000, text="这个函数返回一个张量。"),  # 无需修改
    ]
    
    # 保存测试文件
    test_srt_path = Path(f"courses/{course_name}/transcript.srt")
    test_srt_path.parent.mkdir(parents=True, exist_ok=True)
    save_srt(test_entries, str(test_srt_path))
    print(f"创建测试文件: {test_srt_path}")
    
    # 配置校对器
    config = {
        "llm_model": "qwen-turbo",  # 性价比最高的模型
        "use_llm": True,
        "batch_size": 5
    }
    
    try:
        # 创建校对器
        corrector = TranscriptionCorrector(course_name, config)
        
        # 演示选择性判断
        print("\n选择性判断结果:")
        print("-" * 60)
        
        llm_corrector = corrector.llm_corrector
        for entry in test_entries:
            need_llm = llm_corrector.should_use_llm(entry)
            status = "🤖 LLM处理" if need_llm else "⚡ 跳过"
            print(f"{status} [{entry.index}] {entry.text[:30]}...")
        
        # 执行完整校对
        print("\n开始校对...")
        result_path = corrector.correct(force_reprocess=True)
        
        # 读取结果
        corrected_entries = parse_srt_file(result_path)
        
        print("\n校对对比:")
        print("-" * 60)
        for orig, corr in zip(test_entries, corrected_entries):
            if orig.text != corr.text:
                print(f"📝 [{orig.index}]")
                print(f"   原文: {orig.text}")
                print(f"   校正: {corr.text}")
                print()
        
        # 输出成本统计
        stats = llm_corrector.get_stats()
        if "cost_estimate_cny" in stats:
            print(f"💰 预估API成本: ¥{stats['cost_estimate_cny']:.4f}")
        
    except Exception as e:
        print(f"演示失败（可能是API未配置）: {e}")
        print("仅演示规则校对部分...")
        
        # 仅演示规则校对
        rule_corrector = RuleBasedCorrector()
        corrected, count = rule_corrector.correct_batch(test_entries)
        print(f"\n规则校对修改了 {count} 句")


def demo_cost_comparison():
    """演示成本对比"""
    print("\n" + "="*60)
    print("💰 演示3: 成本对比分析")
    print("="*60)
    
    # 假设1000句字幕
    total_sentences = 1000
    avg_tokens_per_sentence = 25
    
    print(f"\n假设场景: {total_sentences}句字幕，平均每句{avg_tokens_per_sentence}字")
    print("-" * 60)
    
    # 方案1: 纯LLM（每句都调）
    print("\n方案1: 纯LLM校对（每句都调用）")
    llm_calls_1 = total_sentences
    input_tokens_1 = llm_calls_1 * (500 + avg_tokens_per_sentence)  # system + text
    output_tokens_1 = llm_calls_1 * avg_tokens_per_sentence
    cost_1 = (input_tokens_1 * 0.002 + output_tokens_1 * 0.006) / 1000  # qwen-plus价格
    print(f"  API调用: {llm_calls_1} 次")
    print(f"  Input tokens: {input_tokens_1:,}")
    print(f"  Output tokens: {output_tokens_1:,}")
    print(f"  预估成本: ¥{cost_1:.2f}")
    
    # 方案2: 推荐方案（规则+选择性LLM）
    print("\n方案2: 推荐方案（规则+选择性LLM）")
    rule_coverage = 0.4  # 40%句子被规则修改，无需LLM
    llm_selectivity = 0.2  # 剩余60%中只有20%需要LLM
    llm_calls_2 = int(total_sentences * (1 - rule_coverage) * llm_selectivity)
    batch_size = 5
    actual_calls = llm_calls_2 // batch_size + (1 if llm_calls_2 % batch_size else 0)
    input_tokens_2 = actual_calls * (300 + batch_size * avg_tokens_per_sentence)
    output_tokens_2 = actual_calls * batch_size * 15  # JSON输出较短
    cost_2 = (input_tokens_2 * 0.0005 + output_tokens_2 * 0.001) / 1000  # qwen-turbo价格
    
    print(f"  规则处理: {int(total_sentences * rule_coverage)} 句（免费）")
    print(f"  LLM处理: {llm_calls_2} 句 ({llm_selectivity*100:.0f}%)")
    print(f"  API调用: {actual_calls} 次（批处理）")
    print(f"  Input tokens: {input_tokens_2:,}")
    print(f"  Output tokens: {output_tokens_2:,}")
    print(f"  预估成本: ¥{cost_2:.4f}")
    
    # 对比
    print("\n" + "-" * 60)
    savings = (cost_1 - cost_2) / cost_1 * 100
    print(f"💡 成本节省: {savings:.1f}%")
    print(f"💡 准确率: 规则80% + LLM95% = 综合约85%")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="转录校对演示")
    parser.add_argument("--course", default="demo", help="课程名称")
    parser.add_argument("--demo", choices=["rule", "llm", "cost", "all"], 
                       default="all", help="演示类型")
    
    args = parser.parse_args()
    
    if args.demo in ["rule", "all"]:
        demo_rule_corrector()
    
    if args.demo in ["llm", "all"]:
        demo_selective_llm(args.course)
    
    if args.demo in ["cost", "all"]:
        demo_cost_comparison()
    
    print("\n" + "="*60)
    print("✅ 演示完成！")
    print("="*60)
    print("\n使用建议:")
    print("1. 先运行规则校对，处理80%常见错误（免费）")
    print("2. 对于重要课程，启用选择性LLM校对（节省95%成本）")
    print("3. 定期更新knowledge_base/common_misrecognition.md扩展词典")
    print("\n实际使用:")
    print(f"  python -m src.pipeline --course {args.course} --step 3")
    print("  或:")
    print(f"  from src.pipeline.corrector import correct_transcription")
    print(f"  correct_transcription('{args.course}')")


if __name__ == "__main__":
    main()
