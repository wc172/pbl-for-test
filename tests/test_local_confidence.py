#!/usr/bin/env python3
"""
本地模型置信度评估测试 V2

重点：基于PPL阈值筛选ASR错误
"""

import sys
import time
sys.path.insert(0, '.')

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"

print("=" * 70)
print("🧪 本地模型置信度评估测试 V2 (PPL阈值筛选)")
print("=" * 70)
print(f"模型路径: {MODEL_PATH}")
print()

from src.pipeline.local_confidence import LocalConfidenceEvaluator

evaluator = LocalConfidenceEvaluator(MODEL_PATH)

if not evaluator.is_available():
    print("❌ 模型不可用")
    sys.exit(1)

print(f"✅ 模型加载成功")
print(f"   PPL阈值: <{evaluator.PERPLEXITY_GOOD}(跳过) / "
      f"{evaluator.PERPLEXITY_CHECK}(检查) / "
      f">{evaluator.PERPLEXITY_BAD}(必须LLM)")
print()

# ========== 测试：SRT文件PPL分布分析 ==========
print("【测试】SRT文件PPL分布分析")
print("-" * 70)

SRT_PATH = "courses/A3.11_autogen/transcript.srt"

if not __import__('os').path.exists(SRT_PATH):
    print(f"❌ SRT文件不存在: {SRT_PATH}")
    sys.exit(1)

from src.utils.srt_parser import parse_srt_file

entries = parse_srt_file(SRT_PATH)
print(f"加载了 {len(entries)} 条字幕")

# 取前100条分析
test_entries = entries[:100]
texts = [e.text for e in test_entries]

print(f"\n评估前 {len(texts)} 条...")
start = time.time()
scores = evaluator.evaluate_batch(texts, batch_size=8)
elapsed = time.time() - start

print(f"✅ 完成，耗时: {elapsed:.2f}s, 速度: {len(texts)/elapsed:.1f} 条/秒")

# 按PPL区间统计
bins = {
    "优秀 (<50)": [],
    "正常 (50-200)": [],
    "可疑 (200-800)": [],
    "错误 (>800)": [],
}

for entry, score in zip(test_entries, scores):
    ppl = score.perplexity
    if ppl < 50:
        bins["优秀 (<50)"].append((entry, score))
    elif ppl < 200:
        bins["正常 (50-200)"].append((entry, score))
    elif ppl < 800:
        bins["可疑 (200-800)"].append((entry, score))
    else:
        bins["错误 (>800)"].append((entry, score))

print(f"\n📊 PPL分布统计:")
for bin_name, items in bins.items():
    pct = len(items) / len(test_entries) * 100
    print(f"  {bin_name:15s}: {len(items):3d}条 ({pct:5.1f}%)")

# 显示错误示例
if bins["错误 (>800)"]:
    print(f"\n❌ 高PPL错误示例 (必须LLM校正):")
    for entry, score in bins["错误 (>800)"][:5]:
        print(f"  [{entry.index:3d}] PPL:{score.perplexity:8.1f} | {entry.text[:45]}...")

# 显示优秀示例
if bins["优秀 (<50)"]:
    print(f"\n✅ 低PPL优秀示例 (可能正确):")
    for entry, score in bins["优秀 (<50)"][:5]:
        print(f"  [{entry.index:3d}] PPL:{score.perplexity:6.1f} | {entry.text[:45]}...")

# ========== LLM处理量预估 ==========
print(f"\n📈 LLM处理量预估:")
needs_llm_count = len(bins["错误 (>800)"])
check_count = len(bins["可疑 (200-800)"])
skip_count = len(bins["优秀 (<50)"]) + len(bins["正常 (50-200)"])

print(f"  必须LLM (>800):  {needs_llm_count:3d}条 ({needs_llm_count/len(test_entries)*100:.1f}%)")
print(f"  需要检查 (200-800): {check_count:3d}条 ({check_count/len(test_entries)*100:.1f}%)")
print(f"  可跳过 (<200):   {skip_count:3d}条 ({skip_count/len(test_entries)*100:.1f}%)")

# 全量预估
full_estimate = {
    "必须LLM": len([s for s in scores if s.perplexity > 800]) / len(scores) * 1018,
    "需要检查": len([s for s in scores if 200 <= s.perplexity <= 800]) / len(scores) * 1018,
    "可跳过": len([s for s in scores if s.perplexity < 200]) / len(scores) * 1018,
}

print(f"\n📊 全量预估 (共1018条):")
for k, v in full_estimate.items():
    print(f"  {k}: ~{v:.0f}条")

# 成本估算 (假设qwen-max ¥0.02/1K tokens)
avg_tokens_per_sentence = 30
cost_per_sentence = 0.02 / 1000 * avg_tokens_per_sentence
total_cost = full_estimate["必须LLM"] * cost_per_sentence

print(f"\n💰 预估成本 (仅处理必须LLM的部分):")
print(f"  处理条数: {full_estimate['必须LLM']:.0f}")
print(f"  预估费用: ¥{total_cost:.2f}")

print("\n" + "=" * 70)
print("✅ 测试完成")
print("=" * 70)

print("\n💡 结论:")
print(f"  - PPL>{evaluator.PERPLEXITY_BAD} 的句子肯定是ASR错误，必须LLM校正")
print(f"  - 全量约 {full_estimate['必须LLM']:.0f} 条需要LLM，成本约 ¥{total_cost:.2f}")
print(f"  - 相比纯LLM方案 (¥20.36)，节省 {(1-total_cost/20.36)*100:.0f}% 成本")
