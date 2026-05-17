#!/usr/bin/env python3
"""测试术语提取"""

import sys
sys.path.insert(0, '.')

from src.pipeline.term_extractor import extract_course_terms

# 提取并保存
errors = extract_course_terms('A3.11_autogen')

print(f'生成了 {len(errors)} 个错误映射')
print()

print('AutoGen相关:')
for error, correct in errors.items():
    if correct == 'A3.11_autogen':
        print(f'  "{error}" -> "{correct}"')

print()
print('LangChain相关:')
for error, correct in errors.items():
    if correct == 'LangChain':
        print(f'  "{error}" -> "{correct}"')

print()
print('其他关键术语:')
keywords = ['AssistantAgent', 'UserProxyAgent', 'GroupChat', 'Agent']
for error, correct in list(errors.items())[:50]:
    if any(k in correct for k in keywords):
        print(f'  "{error}" -> "{correct}"')
