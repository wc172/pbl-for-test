"""
本地模型置信度评估器

使用 Qwen2.5-Coder 计算困惑度(perplexity)来评估字幕质量。
困惑度越低 = 文本越流畅自然 = 置信度越高
"""

import torch
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class ConfidenceScore:
    """置信度分数"""
    perplexity: float      # 困惑度 (越低越好)
    confidence: float      # 归一化置信度 (0-1, 越高越好)
    is_reliable: bool      # 是否可靠（PPL < 800）
    needs_llm: bool        # 是否需要LLM校正（PPL > 800）


class LocalConfidenceEvaluator:
    """
    本地模型置信度评估器
    
    基于 Qwen2.5-Coder 计算困惑度，评估字幕文本的流畅度。
    """
    
    # 困惑度阈值（基于实测结果）
    # 实测数据：
    # - 正确技术术语: PPL ~ 10-15
    # - 正常课程文本: PPL ~ 40-500（口语化、有语气词）
    # - ASR错误: PPL > 1000 (甚至20000+)
    #
    # 策略：只抓明显错误，放过正常口语
    PERPLEXITY_GOOD = 100.0      # PPL < 100: 高质量，跳过
    PERPLEXITY_CHECK = 400.0     # PPL 100-400: 检查术语
    PERPLEXITY_BAD = 1200.0      # PPL > 1200: 肯定是ASR错误，必须LLM
    
    def __init__(self, model_path: str = "models/Qwen2.5-1.5B-Instruct"):
        """
        初始化评估器
        
        Args:
            model_path: 本地模型路径
        """
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = None
        self.model = None
        self._loaded = False
        
        logger.info(f"本地置信度评估器初始化，模型路径: {model_path}")
        logger.info(f"使用设备: {self.device}")
    
    def load_model(self) -> bool:
        """
        加载模型（延迟加载，第一次使用时才加载）
        
        Returns:
            是否加载成功
        """
        if self._loaded:
            return True
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"正在加载模型: {self.model_path}")
            
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 设置 pad_token（Instruct模型需要）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # 评估模式
            
            self._loaded = True
            logger.info(f"✅ 模型加载完成，设备: {self.model.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def compute_perplexity(self, text: str) -> float:
        """
        计算单条文本的困惑度
        
        Args:
            text: 输入文本
            
        Returns:
            困惑度值 (越低越好)
        """
        if not self._loaded and not self.load_model():
            return float('inf')
        
        try:
            # 对于Instruct模型，添加简单的前缀使其更容易理解任务
            # 但这可能会影响困惑度计算，所以可选
            prompt_text = text  # 不添加前缀，直接计算原文本的困惑度
            
            # 编码
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # 确保有 attention_mask
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # 计算损失
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    labels=inputs["input_ids"]
                )
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            logger.warning(f"计算困惑度失败: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def compute_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        批量计算困惑度（更高效）
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            困惑度列表
        """
        if not self._loaded and not self.load_model():
            return [float('inf')] * len(texts)
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # 批量编码
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.model.device)
                
                # 确保有 attention_mask
                if 'attention_mask' not in inputs:
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                
                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        labels=inputs["input_ids"]
                    )
                    
                    # 使用模型返回的loss（对于批处理，需要按样本分别计算）
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        # 获取每个token的loss（对于Instruct模型）
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()
                        shift_attention = inputs["attention_mask"][..., 1:].contiguous()
                        
                        # 逐样本计算
                        for j in range(len(batch)):
                            # 只计算有效token（非padding）
                            valid_mask = shift_attention[j].bool()
                            if valid_mask.sum() == 0:
                                results.append(float('inf'))
                                continue
                            
                            sample_logits = shift_logits[j][valid_mask]
                            sample_labels = shift_labels[j][valid_mask]
                            
                            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                            loss = loss_fct(
                                sample_logits.view(-1, sample_logits.size(-1)),
                                sample_labels.view(-1)
                            )
                            ppl = torch.exp(loss).item()
                            results.append(ppl)
                    else:
                        # 回退：单条计算
                        for text in batch:
                            results.append(self.compute_perplexity(text))
                        
            except Exception as e:
                logger.warning(f"批量计算失败: {e}，回退到单条计算")
                import traceback
                traceback.print_exc()
                # 回退到单条计算
                for text in batch:
                    results.append(self.compute_perplexity(text))
        
        return results
    
    def perplexity_to_confidence(self, perplexity: float) -> float:
        """
        将困惑度转换为归一化置信度 (0-1)
        
        基于实测数据的简化映射：
        - PPL < 30:    confidence = 0.90 (高质量，可能是正确术语)
        - PPL 30-100:  confidence = 0.70 (中等，正常课程文本)
        - PPL 100-500: confidence = 0.50 (较差，可能有问题)
        - PPL > 500:   confidence = 0.10 (很差，肯定是ASR错误)
        
        注意：正常课程口语的PPL通常较高(40-300)，不必追求低PPL
        """
        if perplexity < self.PERPLEXITY_GOOD:
            return 0.90
        elif perplexity < self.PERPLEXITY_CHECK:
            return 0.70
        elif perplexity < self.PERPLEXITY_BAD:
            return 0.50
        else:
            return 0.10
    
    def should_correct(self, perplexity: float) -> bool:
        """
        判断是否需要LLM校正（基于实测数据）
        
        策略：
        - PPL < 30: 不需要（可能是正确术语）
        - PPL 30-500: 检查是否包含课程术语
        - PPL > 500: 肯定需要（ASR错误）
        """
        return perplexity > self.PERPLEXITY_BAD  # >800需要LLM
    
    def evaluate(self, text: str) -> ConfidenceScore:
        """
        评估单条文本
        
        Args:
            text: 输入文本
            
        Returns:
            ConfidenceScore
        """
        perplexity = self.compute_perplexity(text)
        confidence = self.perplexity_to_confidence(perplexity)
        
        # 关键判断：PPL > 800 肯定是ASR错误，需要LLM
        needs_llm = perplexity > self.PERPLEXITY_BAD
        is_reliable = not needs_llm  # 不需要LLM的就是可靠的
        
        return ConfidenceScore(
            perplexity=perplexity,
            confidence=confidence,
            is_reliable=is_reliable,
            needs_llm=needs_llm
        )
    
    def evaluate_batch(self, texts: List[str], batch_size: int = 8) -> List[ConfidenceScore]:
        """
        批量评估
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            ConfidenceScore 列表
        """
        perplexities = self.compute_batch(texts, batch_size)
        
        results = []
        for ppl in perplexities:
            conf = self.perplexity_to_confidence(ppl)
            needs_llm = ppl > self.PERPLEXITY_BAD
            results.append(ConfidenceScore(
                perplexity=ppl,
                confidence=conf,
                is_reliable=not needs_llm,
                needs_llm=needs_llm
            ))
        
        return results
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        if not self.model_path.exists():
            return False
        return self.load_model()


class HybridConfidenceCalculator:
    """
    混合置信度计算器
    
    结合规则启发式和本地模型困惑度
    """
    
    def __init__(self, course_name: str, use_local_model: bool = True):
        self.course_name = course_name
        self.use_local_model = use_local_model
        
        # 初始化本地模型评估器
        self.local_evaluator = None
        if use_local_model:
            self.local_evaluator = LocalConfidenceEvaluator()
            if not self.local_evaluator.is_available():
                logger.warning("本地模型不可用，将使用规则置信度")
                self.local_evaluator = None
        
        # 规则计算器（作为fallback）
        from src.pipeline.corrector_v2 import ConfidenceCalculator
        self.rule_calculator = ConfidenceCalculator(course_name)
    
    def calculate(self, original_text: str, corrected_text: str, 
                  was_modified: bool) -> float:
        """
        计算综合置信度
        
        策略:
        1. 如果本地模型可用，优先使用困惑度
        2. 如果不可用，使用规则启发式
        3. 如果规则修改过，适当降低置信度
        """
        # 使用本地模型评估校正后的文本
        if self.local_evaluator:
            score = self.local_evaluator.evaluate(corrected_text)
            
            # 如果规则修改过，稍微降低置信度（因为修改可能引入错误）
            if was_modified:
                return max(0.3, score.confidence - 0.1)
            
            return score.confidence
        
        # 回退到规则计算
        from src.utils.srt_parser import SRTEntry
        entry = SRTEntry(index=1, start_ms=0, end_ms=5000, text=original_text)
        return self.rule_calculator.calculate(entry, corrected_text, was_modified)
    
    def calculate_batch(self, entries: List, corrected_texts: List[str],
                       modified_flags: List[bool]) -> List[float]:
        """
        批量计算置信度
        """
        if self.local_evaluator:
            # 批量评估校正后的文本
            scores = self.local_evaluator.evaluate_batch(corrected_texts)
            
            results = []
            for score, was_modified in zip(scores, modified_flags):
                if was_modified:
                    results.append(max(0.3, score.confidence - 0.1))
                else:
                    results.append(score.confidence)
            
            return results
        
        # 回退到单条规则计算
        results = []
        for entry, corrected, modified in zip(entries, corrected_texts, modified_flags):
            results.append(self.calculate(entry.text, corrected, modified))
        
        return results


# 便捷函数
def evaluate_srt_quality(srt_path: str, model_path: str = "models/Qwen2.5-1.5B-Instruct") -> dict:
    """
    评估SRT文件整体质量
    
    Returns:
        {
            "total": 总句数,
            "avg_perplexity": 平均困惑度,
            "avg_confidence": 平均置信度,
            "low_quality_count": 低质量句数,
            "low_quality_ratio": 低质量比例,
        }
    """
    from src.utils.srt_parser import parse_srt_file
    
    entries = parse_srt_file(srt_path)
    texts = [e.text for e in entries]
    
    evaluator = LocalConfidenceEvaluator(model_path)
    scores = evaluator.evaluate_batch(texts)
    
    perplexities = [s.perplexity for s in scores]
    confidences = [s.confidence for s in scores]
    
    low_quality = sum(1 for s in scores if not s.is_reliable)
    
    return {
        "total": len(entries),
        "avg_perplexity": sum(perplexities) / len(perplexities),
        "avg_confidence": sum(confidences) / len(confidences),
        "low_quality_count": low_quality,
        "low_quality_ratio": low_quality / len(entries),
    }


if __name__ == "__main__":
    # 测试
    import sys
    
    print("🧪 本地置信度评估器测试")
    print("=" * 60)
    
    evaluator = LocalConfidenceEvaluator()
    
    if not evaluator.is_available():
        print("❌ 模型不可用")
        sys.exit(1)
    
    # 测试用例
    test_cases = [
        "PyTorch是一个深度学习框架。",
        "AutoGen是微软开源的多Agent框架。",
        "爱do卷相关的课程。",
        "这个句子有wierd的拼写错误。",
        "1234567890乱码",
    ]
    
    print("\n单条测试:")
    for text in test_cases:
        score = evaluator.evaluate(text)
        status = "✅" if score.is_reliable else "❌"
        print(f"{status} PPL: {score.perplexity:6.2f} | CONF: {score.confidence:.2f} | {text[:30]}...")
    
    print("\n批量测试:")
    scores = evaluator.evaluate_batch(test_cases)
    for text, score in zip(test_cases, scores):
        print(f"  PPL: {score.perplexity:6.2f} | {text[:30]}...")
