"""
模块3: LLM校对模块

改进点:
1. 扩展词典：覆盖更多细微ASR错误（long graph、IL键等）
2. LLM批处理：5条一个prompt，节省token
3. RAG增强：批量检索课件内容，提高上下文质量

处理流程:
1. 规则校对（扩展词典）
2. PPL筛选（>400即送入LLM）
3. 批处理RAG检索
4. 批处理LLM校正（5条/prompt）
5. 合并结果
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# 加载环境变量
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from src.utils.srt_parser import SRTEntry, parse_srt_file, save_srt, ms_to_srt_time

logger = logging.getLogger(__name__)


class ExtendedRuleCorrector:
    """扩展规则校对器 - 覆盖更多细微错误"""
    
    def __init__(self, course_name: Optional[str] = None):
        self.misrecognition_dict = self._build_extended_dict()
        if course_name:
            course_dict = self._load_course_terms(course_name)
            self.misrecognition_dict.update(course_dict)
        self._compile_patterns()
        logger.info(f"扩展规则校对器：{len(self.misrecognition_dict)} 个映射")
    
    def _build_extended_dict(self) -> Dict[str, str]:
        """构建扩展词典 - 包含细微ASR错误"""
        
        return {
            # ========== A3.11_autogen 扩展 (基于LLM日志分析) ==========
            "爱do卷": "A3.11_autogen",
            "爱 do 卷": "A3.11_autogen",      # 带空格版本
            "爱do俊": "A3.11_autogen",
            "爱 do 俊": "A3.11_autogen",      # 带空格版本
            "爱do": "A3.11_autogen",
            "爱 do": "A3.11_autogen",         # 带空格版本
            "adojen": "A3.11_autogen",
            "ado卷": "A3.11_autogen",
            "爱豆 jeen": "A3.11_autogen",     # LLM发现
            "爱豆": "A3.11_autogen",          # 简写
            "爱豆键": "A3.11_autogen",        # LLM发现
            "爱 dol 键": "A3.11_autogen",     # LLM发现
            "idol in": "A3.11_autogen",       # LLM发现
            "i dol 键": "A3.11_autogen",      # LLM发现
            "a 度键": "A3.11_autogen",        # LLM发现
            "i 键头": "A3.11_autogen",        # LLM发现
            "i to键": "A3.11_autogen",        # 合并版本
            "i to": "A3.11_autogen",          # 简写
            "i图建": "A3.11_autogen",
            "i图": "A3.11_autogen",
            "奥特真": "A3.11_autogen",
            "奥特根": "A3.11_autogen",
            "auto真": "A3.11_autogen",
            "奥gen": "A3.11_autogen",
            "AL键": "A3.11_autogen",
            "爱真": "A3.11_autogen",
            "IL键": "A3.11_autogen",
            "augen": "A3.11_autogen",
            "奥真": "A3.11_autogen",
            "sualt": "A3.11_autogen",         # LLM发现
            "a 三": "A3.11_autogen",          # a三→A3.11_autogen (LLM高频)
            "a三": "A3.11_autogen",           # 合并版本
            
            # ========== LangChain 扩展 (基于LLM日志分析) ==========
            "烂链": "LangChain",
            "浪链": "LangChain",
            "浪茜": "LangChain",
            "烂茜": "LangChain",
            "浪线": "LangChain",
            "浪嵌": "LangChain",
            "long 嵌": "LangChain",     # 带空格版本
            "long芡": "LangChain",      # LLM发现
            "long 芡": "LangChain",     # 带空格版本
            "朗茜": "LangChain",        # LLM发现
            "LO键": "LangChain",
            "L键": "LangChain",
            "long graph": "LangChain",
            "long茜": "LangChain",
            "lang线": "LangChain",
            "long线": "LangChain",
            
            # ========== 其他术语 (基于LLM日志分析) ==========
            "a3": "A3",                 # 保留A3映射
            "reg": "RAG",
            "define": "Dify",
            "陶天集团": "淘宝天猫集团",
            "精营智能": "径营智能",
            
            # ========== GPT/LLM 相关 (LLM日志发现) ==========
            "GP 四": "GPT-4",
            "GPD": "GPT",
            "GB 三点五": "GPT-3.5",
            "GP 3.5": "GPT-3.5",
            "GP 4": "GPT-4",
            "切 TTP": "ChatGPT",
            "ena PI": "API",            # API的ASR错误
            
            # ========== Agent 相关 (LLM日志发现) ==========
            "agenent": "agent",         # LLM发现
            "a检t": "agent",            # LLM发现
            "a 检 t": "agent",          # 带空格版本
            
            # ========== 技术框架 ==========
            "皮tor": "PyTorch",
            "tensor flow": "TensorFlow",
            "康达": "Conda",
            "朱庇特": "Jupyter",
            "爬沈": "Python",
            "派森": "Python",
            
            # ========== 常见组合错误 ==========
            "人工智能base": "人工智能基础",
            "大模型ase": "大模型应用",
        }
    
    @classmethod
    def export_builtin_dict(cls, output_path: str = "knowledge_base/common_misrecognition_export.json"):
        """
        导出内置通用词典到文件
        
        Args:
            output_path: 输出文件路径
            
        Example:
            >>> ExtendedRuleCorrector.export_builtin_dict("correction_dict.json")
        """
        # 创建临时实例来获取内置词典
        instance = cls.__new__(cls)
        builtin_dict = instance._build_extended_dict()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(builtin_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 内置词典已导出: {output_file}")
        print(f"   共 {len(builtin_dict)} 条规则")
        
        # 按目标分类显示
        from collections import defaultdict
        by_target = defaultdict(list)
        for wrong, correct in builtin_dict.items():
            by_target[correct].append(wrong)
        
        print(f"\n分类统计:")
        for target, errors in sorted(by_target.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {target}: {len(errors)}条")
        
        return output_file
    
    def _load_course_terms(self, course_name: str) -> Dict[str, str]:
        """
        加载课程特定词典
        
        逻辑：
        1. 优先查找 correction_log.json，存在则转换为 correction_dict.json
        2. 然后查找 correction_dict.json，存在则读取
        3. 如果都不存在，可使用内置词典导出作为基础
        """
        course_dir = Path(f"courses/{course_name}")
        log_file = course_dir / "correction_log.json"
        dict_file = course_dir / "correction_dict.json"
        
        # 步骤1: 如果有 correction_log.json，转换为 correction_dict.json
        if log_file.exists():
            logger.info(f"发现 {log_file.name}，转换为 {dict_file.name}...")
            correction_dict = self._convert_log_to_dict(log_file)
            
            # 合并已有的 correction_dict.json（如果有）
            if dict_file.exists():
                with open(dict_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                correction_dict.update(existing)
                logger.info(f"  合并已有 {len(existing)} 条规则")
            
            # 保存
            with open(dict_file, 'w', encoding='utf-8') as f:
                json.dump(correction_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"  已生成 {len(correction_dict)} 条规则")
            return correction_dict
        
        # 步骤2: 读取已有的 correction_dict.json
        if dict_file.exists():
            with open(dict_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {}
    
    def _convert_log_to_dict(self, log_file: Path) -> Dict[str, str]:
        """
        将 correction_log.json 转换为 correction_dict.json
        
        提取所有 LLM 修改的 (original -> corrected) 映射
        """
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log = json.load(f)
            
            correction_dict = {}
            
            for c in log.get('corrections', []):
                # 只提取 LLM 修改（规则修改通常已在内置词典）
                if 'llm' in c.get('stage', ''):
                    orig = c['original']
                    corr = c['corrected']
                    
                    # 过滤：长度适中，且确实不同
                    if len(orig) <= 50 and orig != corr:
                        correction_dict[orig] = corr
            
            return correction_dict
            
        except Exception as e:
            logger.warning(f"转换 {log_file} 失败: {e}")
            return {}
    
    def _compile_patterns(self):
        sorted_items = sorted(
            self.misrecognition_dict.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self._replace_patterns = [
            (re.compile(re.escape(wrong), re.IGNORECASE), correct)
            for wrong, correct in sorted_items
        ]
    
    def correct(self, entry: SRTEntry) -> Tuple[SRTEntry, bool]:
        text = entry.text
        original = text
        
        for pattern, correct in self._replace_patterns:
            if pattern.search(text):
                text = pattern.sub(correct, text)
        
        text = re.sub(r'([a-zA-Z]),([a-zA-Z])', r'\1, \2', text)
        text = re.sub(r'  +', ' ', text)
        
        modified = (text != original)
        
        return SRTEntry(
            index=entry.index,
            start_ms=entry.start_ms,
            end_ms=entry.end_ms,
            text=text
        ), modified


class OptimizedPPLFilter:
    """
    优化的PPL筛选器
    
    基于本地模型(Qwen2.5-1.5B)计算困惑度，筛选需要LLM校对的句子。
    兼顾准确率和速度，只筛选明显有问题的句子。
    """
    
    # PPL阈值（基于实测数据）
    PERPLEXITY_GOOD = 100.0      # <100: 高质量，跳过
    PERPLEXITY_CHECK = 400.0     # 100-400: 检查是否含课程术语
    PERPLEXITY_BAD = 1200.0      # >1200: 肯定是ASR错误，必须LLM
    
    def __init__(self, model_path: str = "models/Qwen2.5-1.5B-Instruct"):
        """
        初始化PPL筛选器
        
        Args:
            model_path: 本地模型路径
        """
        from src.pipeline.local_confidence import LocalConfidenceEvaluator
        self.evaluator = LocalConfidenceEvaluator(model_path)
        self._available = self.evaluator.is_available()
        
        if self._available:
            logger.info(f"PPL筛选器初始化完成，模型: {model_path}")
        else:
            logger.warning(f"PPL模型不可用: {model_path}，将回退到规则筛选")
    
    def filter_entries(self, entries: List[SRTEntry]) -> Tuple[List[int], List[int], List[int]]:
        """
        筛选条目，返回三类索引
        
        Args:
            entries: SRT条目列表
            
        Returns:
            (skip_idx, check_idx, llm_idx)
            - skip_idx: PPL<200，跳过
            - check_idx: PPL 200-800，检查术语
            - llm_idx: PPL>800，必须LLM
        """
        if not self._available:
            # 模型不可用，全部跳过（依赖规则校对）
            return list(range(len(entries))), [], []
        
        skip_idx = []
        check_idx = []
        llm_idx = []
        
        # 提取文本
        texts = [e.text for e in entries]
        
        # 批量计算PPL
        logger.info(f"计算PPL中... (共{len(texts)}条)")
        ppl_scores = self.evaluator.compute_batch(texts, batch_size=16)
        
        # 分类
        for i, ppl in enumerate(ppl_scores):
            if ppl < self.PERPLEXITY_GOOD:
                skip_idx.append(i)
            elif ppl < self.PERPLEXITY_CHECK:
                check_idx.append(i)
            else:
                llm_idx.append(i)
        
        # 打印统计
        total = len(entries)
        logger.info(f"PPL筛选结果:")
        logger.info(f"  跳过: {len(skip_idx)}条 ({len(skip_idx)/total*100:.1f}%)")
        logger.info(f"  检查: {len(check_idx)}条 ({len(check_idx)/total*100:.1f}%)")
        logger.info(f"  LLM:  {len(llm_idx)}条 ({len(llm_idx)/total*100:.1f}%)")
        
        return skip_idx, check_idx, llm_idx


class BatchRAGLLMCorrector:
    """批处理RAG+LLM校对器 - 10条一个prompt"""
    
    BATCH_SIZE = 10  # 每批处理10条
    
    def __init__(self, course_name: str, model: str = "qwen-max"):
        self.course_name = course_name
        self.model = model
        
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("未设置QWEN_API_KEY")
        
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 初始化RAG
        from src.pipeline.course_rag import CourseRAGQueryInterface
        self.rag = CourseRAGQueryInterface(course_name)
        
        if not self.rag.exists():
            raise ValueError(f"课程RAG未构建: {course_name}")
    
    def correct_batch(self, entries: List[SRTEntry]) -> List[str]:
        """
        批量校对
        
        Args:
            entries: 待校对的条目列表
            
        Returns:
            纠正后的文本列表
        """
        if not entries:
            return []
        
        results = []
        
        # 分批处理
        for i in range(0, len(entries), self.BATCH_SIZE):
            batch = entries[i:i + self.BATCH_SIZE]
            batch_results = self._correct_single_batch(batch)
            results.extend(batch_results)
            
            if (i // self.BATCH_SIZE + 1) % 5 == 0:
                print(f"    LLM批处理进度: {min(i + self.BATCH_SIZE, len(entries))}/{len(entries)}")
        
        return results
    
    def _correct_single_batch(self, entries: List[SRTEntry]) -> List[str]:
        """处理单批次"""
        if len(entries) == 1:
            # 单条使用简单prompt
            return [self._correct_single(entries[0])]
        
        # 多条批量RAG检索
        combined_text = " ".join([e.text for e in entries])
        try:
            rag_results = self.rag.search(combined_text, top_k=5)
            rag_context = self.rag.format_for_llm(rag_results, max_length=2000)
        except Exception as e:
            logger.warning(f"RAG检索失败: {e}")
            rag_context = "未找到相关课件内容"
        
        # 构建批量prompt
        items_text = "\n".join([
            f"{i+1}. {e.text}"
            for i, e in enumerate(entries)
        ])
        
        prompt = f"""请根据课件内容，批量纠正以下{len(entries)}条转录文本中的错误。

【课件相关内容】
{rag_context}

【待纠正文本】（每行一个，保留序号）
{items_text}

【纠正要求】
1. 修正技术术语（如"浪线"→"LangChain"，"IL键"→"A3.11_autogen"）
2. 保持原意不变
3. 按序号输出纠正后的文本

【输出格式】
1. [纠正后的文本1]
2. [纠正后的文本2]
..."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的课程字幕校对专家，擅长批量处理。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800  # 10条需要更多token
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析结果
            return self._parse_batch_response(content, entries)
            
        except Exception as e:
            logger.error(f"LLM批处理失败: {e}")
            # 失败时返回原文
            return [e.text for e in entries]
    
    def _parse_batch_response(self, content: str, entries: List[SRTEntry]) -> List[str]:
        """解析批量响应"""
        results = [e.text for e in entries]  # 默认返回原文
        
        # 尝试按行解析
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # 匹配 "1. xxx" 或 "1: xxx" 格式
            match = re.match(r'^(\d+)[:\.\s]+(.+)$', line)
            if match:
                idx = int(match.group(1)) - 1
                text = match.group(2).strip()
                if 0 <= idx < len(entries):
                    # 过滤常见前缀
                    prefixes = ['纠正后：', '纠正：', 'corrected:', 'output:']
                    for prefix in prefixes:
                        if text.lower().startswith(prefix.lower()):
                            text = text[len(prefix):].strip()
                    
                    # 检查结果合理性
                    original = entries[idx].text
                    if len(text) >= len(original) * 0.5 and len(text) <= len(original) * 2:
                        results[idx] = text
        
        return results
    
    def _correct_single(self, entry: SRTEntry) -> str:
        """单条校对（fallback）"""
        try:
            rag_results = self.rag.search(entry.text, top_k=3)
            rag_context = self.rag.format_for_llm(rag_results, max_length=1500)
        except:
            rag_context = "未找到相关课件内容"
        
        prompt = f"""请纠正转录文本中的错误。

【课件内容】{rag_context[:500]}

【文本】{entry.text}

只输出纠正后的文本，不要解释。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是字幕校对专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            corrected = response.choices[0].message.content.strip()
            
            if len(corrected) < len(entry.text) * 0.5:
                return entry.text
            
            return corrected
            
        except:
            return entry.text


class TranscriptionCorrector:
    """完整校对器 V4"""
    
    def __init__(self, course_name: str, config: Optional[Dict] = None):
        self.course_name = course_name
        self.config = config or {}
        
        self.rule_corrector = ExtendedRuleCorrector(course_name)
        self.ppl_filter = OptimizedPPLFilter()
        
        self.use_llm = self.config.get("use_llm", True)
        if self.use_llm:
            try:
                self.llm_corrector = BatchRAGLLMCorrector(
                    course_name,
                    model=self.config.get("llm_model", "qwen-max")
                )
            except Exception as e:
                logger.warning(f"LLM初始化失败: {e}")
                self.use_llm = False
        
        self.input_srt = Path(f"courses/{course_name}/transcript.srt")
        self.output_srt = Path(f"courses/{course_name}/transcript_corrected.srt")
        self.log_path = Path(f".cache/{course_name}/correction_log.json")
    
    def correct(self, force_reprocess: bool = False) -> str:
        """执行校对"""
        if not self.input_srt.exists():
            raise FileNotFoundError(f"输入文件不存在: {self.input_srt}")
        
        if not force_reprocess and self.output_srt.exists():
            return str(self.output_srt)
        
        # 初始化修改记录
        corrections_log = []
        
        # 1. 读取SRT
        entries = parse_srt_file(str(self.input_srt))
        print(f"共 {len(entries)} 句字幕")
        
        # 2. 规则校对
        print("\n1. 规则校对...")
        rule_results = []
        rule_modified_count = 0
        for entry in entries:
            corrected, was_modified = self.rule_corrector.correct(entry)
            rule_results.append(corrected)
            
            if was_modified:
                rule_modified_count += 1
                corrections_log.append({
                    "index": entry.index,
                    "timestamp": ms_to_srt_time(entry.start_ms),
                    "original": entry.text,
                    "corrected": corrected.text,
                    "stage": "rule",
                    "reason": "词典替换"
                })
        
        # 3. PPL筛选
        print("\n2. PPL筛选...")
        skip_idx, check_idx, llm_idx = self.ppl_filter.filter_entries(rule_results)
        all_llm_idx = sorted(set(check_idx + llm_idx))
        
        print(f"\n  处理计划: 跳过{len(skip_idx)}条, LLM处理{len(all_llm_idx)}条")  # 4. LLM批处理校对
        final_results = list(rule_results)
        llm_stats = {"count": 0, "modified": 0}
        
        if self.use_llm and all_llm_idx:
            print(f"\n3. RAG+LLM批处理校对 (batch_size={self.llm_corrector.BATCH_SIZE})...")
            
            llm_entries = [rule_results[i] for i in all_llm_idx]
            corrected_texts = self.llm_corrector.correct_batch(llm_entries)
            
            # 应用结果
            for idx, corrected_text in zip(all_llm_idx, corrected_texts):
                original_entry = rule_results[idx]
                original_text = original_entry.text
                
                if corrected_text != original_text:
                    final_results[idx] = SRTEntry(
                        index=original_entry.index,
                        start_ms=original_entry.start_ms,
                        end_ms=original_entry.end_ms,
                        text=corrected_text
                    )
                    llm_stats["modified"] += 1
                    corrections_log.append({
                        "index": original_entry.index,
                        "timestamp": ms_to_srt_time(original_entry.start_ms),
                        "original": original_text,
                        "corrected": corrected_text,
                        "stage": "llm",
                        "reason": "RAG+LLM校正"
                    })
                llm_stats["count"] += 1
            
            print(f"\n  LLM修改: {llm_stats['modified']}/{llm_stats['count']}")
        
        # 5. 保存SRT
        save_srt(final_results, str(self.output_srt))
        
        # 6. 保存校正日志
        self._save_correction_log(len(entries), corrections_log, llm_stats)
        
        # 7. 打印报告
        self._print_report(entries, rule_results, final_results, all_llm_idx, llm_stats, corrections_log)
        
        return str(self.output_srt)
    
    def _save_correction_log(self, total_segments: int, corrections: list, llm_stats: dict):
        """保存校正日志到JSON"""
        log_data = {
            "course_name": self.course_name,
            "timestamp": datetime.now().isoformat(),
            "total_segments": total_segments,
            "corrected_segments": len(corrections),
            "rule_modified": sum(1 for c in corrections if c["stage"] == "rule"),
            "llm_processed": llm_stats["count"],
            "llm_modified": llm_stats["modified"],
            "corrections": corrections
        }
        
        # 确保目录存在
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📝 校正日志已保存: {self.log_path}")
    
    def _print_report(self, original, rule_results, final, llm_indices, llm_stats, corrections_log=None):
        """打印报告"""
        rule_modified = sum(1 for c in corrections_log if c["stage"] == "rule") if corrections_log else 0
        llm_modified = sum(1 for c in corrections_log if c["stage"] == "llm") if corrections_log else llm_stats["modified"]
        
        print("\n" + "=" * 60)
        print("📊 校对完成摘要")
        print("=" * 60)
        print(f"总字幕数:     {len(original)}")
        print(f"规则修改:     {rule_modified}")
        print(f"LLM处理:      {llm_stats['count']} (批处理，每批{self.llm_corrector.BATCH_SIZE}条)")
        print(f"LLM修改:      {llm_modified}")
        print(f"总修改数:     {len(corrections_log) if corrections_log else rule_modified + llm_modified}")
        
        # 成本估算（批处理更省token）
        batch_count = (llm_stats['count'] + self.llm_corrector.BATCH_SIZE - 1) // self.llm_corrector.BATCH_SIZE
        estimated_tokens = batch_count * 350 + llm_stats['count'] * 30  # prompt (10条上下文) + output
        cost = estimated_tokens * 0.00002  # qwen-max ¥0.02/1K
        print(f"预估成本:     ¥{cost:.3f} (批处理节省约60%)")
        print("=" * 60)


def correct_transcription(course_name: str, config: Optional[Dict] = None) -> str:
    """便捷函数"""
    corrector = TranscriptionCorrector(course_name, config)
    return corrector.correct()


if __name__ == "__main__":
    import sys
    course = sys.argv[1] if len(sys.argv) > 1 else "A3.11_autogen"
    
    config = {
        "llm_model": "qwen-max",
        "use_llm": True,
    }
    
    result = correct_transcription(course, config)
    print(f"\n✅ 完成: {result}")
