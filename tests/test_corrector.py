#!/usr/bin/env python3
"""
模块3 V4 测试 - 验证批处理和词典扩展
"""

import sys
import os
sys.path.insert(0, '.')

COURSE = "A3.11_autogen"

print("=" * 70)
print("🧪 模块3 V4 测试")
print("=" * 70)
print("改进点:")
print("  1. 扩展词典（long graph、IL键等）")
print("  2. LLM批处理（5条/prompt）")
print("  3. RAG批量检索")
print()

# 清理
for f in [f'courses/{COURSE}/transcript_corrected.srt']:
    if os.path.exists(f):
        os.remove(f)
        print(f"✓ 已清理: {f}")

if not os.path.exists(f'courses/{COURSE}/transcript.srt'):
    print(f"❌ 输入文件不存在")
    sys.exit(1)

print(f"✓ 输入就绪\n")

# ========== 运行校对 ==========
print("【运行V4校对】")
print("-" * 70)

from src.pipeline.corrector import correct_transcription

config = {
    "llm_model": "qwen-max",
    "use_llm": True,
}

try:
    result = correct_transcription(COURSE, config)
    print(f"\n✅ 校对完成: {result}")
    
    # ========== 验证 ==========
    print("\n" + "=" * 70)
    print("🔍 验证扩展词典效果")
    print("=" * 70)
    
    from src.utils.srt_parser import parse_srt_file
    entries = parse_srt_file(result)
    
    # 检查新增词典项
    new_patterns = [
        ("IL键", "A3.11_autogen"),
        ("long graph", "LangChain"),
        ("L键", "LangChain"),
        ("lang线", "LangChain"),
        ("i图建", "A3.11_autogen"),
    ]
    
    print("\n新增词典项检查:")
    for wrong, correct in new_patterns:
        # 检查是否还有错误形式
        wrong_count = sum(1 for e in entries if wrong in e.text)
        correct_count = sum(1 for e in entries if correct in e.text)
        
        if wrong_count == 0:
            print(f"✅ '{wrong}' → '{correct}': 已完全修复")
        else:
            print(f"⚠️  '{wrong}': 仍有{wrong_count}处")
            # 显示示例
            for e in entries:
                if wrong in e.text:
                    print(f"      [{e.index}] {e.text[:50]}...")
                    break
    
    # 检查批处理效果（查看RAG上下文使用）
    print("\n" + "=" * 70)
    print("📝 批处理效果验证")
    print("=" * 70)
    
    # 显示几个连续条目的处理结果
    print("\n连续条目示例（验证批处理一致性）:")
    for e in entries[20:26]:
        text_short = e.text[:55] + "..." if len(e.text) > 55 else e.text
        print(f"  [{e.index:3d}] {text_short}")
    
    # 总体统计
    print("\n" + "=" * 70)
    print("📊 总体统计")
    print("=" * 70)
    
    keywords = ['A3.11_autogen', 'LangChain', 'RAG', 'Dify']
    for kw in keywords:
        count = sum(1 for e in entries if kw in e.text)
        print(f"  {kw}: {count}次")
    
    # 检查剩余错误
    all_errors = ['浪线', '浪嵌', 'i图建', 'IL键', 'LO键', 'long graph', '爱do']
    remaining = []
    for e in entries:
        for err in all_errors:
            if err in e.text:
                remaining.append((e.index, err, e.text[:40]))
    
    if remaining:
        print(f"\n⚠️  剩余错误 ({len(remaining)}处，前5个):")
        for idx, err, text in remaining[:5]:
            print(f"    [{idx}] '{err}': {text}...")
    else:
        print("\n✅ 所有已知错误形式均已修复")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ 测试完成")
print("=" * 70)
