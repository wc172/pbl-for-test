"""
语义分割器 - 基于SeqModel进行Entry级语义分组

核心约束：以SRT Entry为原子单位，不切分Entry内部
"""

import re
import logging
from typing import List, Optional
from pathlib import Path

from src.utils.srt_parser import SRTEntry

logger = logging.getLogger(__name__)

# 尝试导入transformers
try:
    from transformers import BertTokenizer, BertForTokenClassification
    import torch
    _transformers_available = True
except ImportError:
    logger.warning("transformers未安装，SeqModel功能不可用")
    _transformers_available = False
    BertTokenizer = None
    BertForTokenClassification = None
    torch = None


class SeqModelSegmenter:
    """
    基于阿里达摩院SeqModel的语义边界检测器
    模型: nlp_bert_document-segmentation_chinese-base
    
    使用transformers直接加载，绕过ModelScope
    """
    
    def __init__(self, model_path: str = "models/nlp_bert_document-segmentation_chinese-base"):
        if not _transformers_available:
            raise ImportError("transformers未安装。请运行: pip install transformers torch")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"SeqModel未找到: {model_path}")
        
        logger.info(f"正在加载SeqModel: {model_path}")
        
        # 直接使用transformers加载
        self.tokenizer = BertTokenizer.from_pretrained(str(model_path))
        self.model = BertForTokenClassification.from_pretrained(str(model_path))
        self.model.eval()
        
        # 标签映射 (从config.json中读取)
        # 0: B-EOP (段落结束), 1: O (非结束)
        self.eop_label = 0
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"SeqModel加载完成，使用设备: {self.device}")
    
    def _detect_eop_positions(self, text: str) -> List[int]:
        """
        检测段落结束位置（EOP: End Of Paragraph）
        
        Returns:
            段落结束位置的字符索引列表
        """
        # 分词
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping").squeeze().tolist()
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # 找到EOP位置
        eop_positions = []
        for idx, pred in enumerate(predictions):
            if pred == self.eop_label and idx < len(offset_mapping):
                # 获取该token的结束字符位置
                char_end = offset_mapping[idx][1]
                eop_positions.append(char_end)
        
        return sorted(set(eop_positions))
    
    def is_boundary(self, prev_entries: List[SRTEntry], next_entries: List[SRTEntry]) -> bool:
        """
        判断两组Entry之间是否是语义边界
        
        策略：
        1. 将两组文本拼接
        2. 用模型检测EOP位置
        3. 如果在prev_text结尾附近有EOP，则认为是边界
        """
        prev_text = " ".join(e.text for e in prev_entries)
        next_text = " ".join(e.text for e in next_entries)
        
        combined = prev_text + "\n" + next_text
        
        # 检测EOP位置
        eop_positions = self._detect_eop_positions(combined)
        
        if not eop_positions:
            return False
        
        # 检查是否有EOP接近prev_text的结尾
        prev_len = len(prev_text)
        # 允许10%的误差范围
        tolerance = prev_len * 0.1
        
        for pos in eop_positions:
            if abs(pos - prev_len) < tolerance:
                return True
        
        return False


class EntryGroupSegmenter:
    """
    Entry组分割器（无LLM，纯规则）
    
    当不使用SeqModel时的备选方案
    """
    
    def is_boundary(self, prev_entries: List[SRTEntry], next_entries: List[SRTEntry]) -> bool:
        """
        基于启发式规则判断边界
        """
        prev_text = " ".join(e.text for e in prev_entries)
        next_text = " ".join(e.text for e in next_entries)
        
        # 规则1：下一段以"首先"、"第一"、"那么"等开头 → 边界
        transition_words = ['首先', '第一', '那么', '接下来', '下面', '我们来看', '本章', '本节']
        for word in transition_words:
            if next_text.startswith(word):
                return True
        
        # 规则2：前一段以"总结一下"、"综上所述"结尾 → 边界
        conclusion_words = ['总结一下', '综上所述', '总之', '最后', '以上就是']
        for word in conclusion_words:
            if prev_text.rstrip().endswith(word):
                return True
        
        return False
