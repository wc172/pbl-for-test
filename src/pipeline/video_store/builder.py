"""
Segment构建器

从校正后的SRT构建Segment存储
核心约束：以SRT Entry为原子单位，不切分Entry内部
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from src.utils.srt_parser import SRTEntry, parse_srt_file
from src.pipeline.video_store.segmenter import SeqModelSegmenter, EntryGroupSegmenter

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """视频Segment原子单位"""
    id: str
    time_start: float      # 秒
    time_end: float        # 秒
    duration: float        # 秒
    srt_start_idx: int     # 对应SRT起始序号
    srt_end_idx: int       # 对应SRT结束序号
    text: str              # 完整转录文本
    keywords: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "duration": self.duration,
            "srt_start_idx": self.srt_start_idx,
            "srt_end_idx": self.srt_end_idx,
            "text": self.text,
            "keywords": self.keywords,
            "features": self.features
        }


class SegmentBuilder:
    """
    Segment构建器
    
    从校正后的SRT构建Segment，以Entry为原子单位
    """
    
    def __init__(self, 
                 course_name: str,
                 use_semantic_segmenter: bool = True,
                 min_duration: float = 60,
                 max_duration: float = 300,
                 min_entries: int = 3):
        """
        Args:
            course_name: 课程名称
            use_semantic_segmenter: 是否使用SeqModel进行语义分割
            min_duration: 最小Segment时长（秒）
            max_duration: 最大Segment时长（秒）
            min_entries: 最少Entry数量
        """
        self.course_name = course_name
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_entries = min_entries
        
        # 选择分割器
        if use_semantic_segmenter:
            try:
                self.segmenter = SeqModelSegmenter()
                logger.info("使用SeqModel进行语义分割")
            except FileNotFoundError as e:
                logger.warning(f"SeqModel未找到，降级为规则分割: {e}")
                self.segmenter = EntryGroupSegmenter()
        else:
            self.segmenter = EntryGroupSegmenter()
            logger.info("使用规则分割器")
    
    def build(self, srt_path: Path) -> List[Segment]:
        """
        从SRT文件构建Segments
        
        Args:
            srt_path: 校正后的SRT文件路径
            
        Returns:
            List[Segment]: 构建的Segment列表
        """
        logger.info(f"开始构建Segments: {srt_path}")
        
        # 1. 解析SRT
        entries = parse_srt_file(str(srt_path))
        logger.info(f"解析SRT完成: {len(entries)} 个Entry")
        
        if not entries:
            return []
        
        # 2. 构建Segments
        segments = self._build_segments(entries)
        logger.info(f"Segments构建完成: {len(segments)} 个")
        
        return segments
    
    def _build_segments(self, entries: List[SRTEntry]) -> List[Segment]:
        """
        核心算法：Entry级语义分组
        
        策略：
        1. 按时间顺序累积Entry
        2. 达到min_duration后，尝试检测语义边界
        3. 如果检测到边界 或 达到max_duration，则切分
        """
        segments = []
        current_group: List[SRTEntry] = []
        
        i = 0
        while i < len(entries):
            entry = entries[i]
            current_group.append(entry)
            
            # 检查是否满足切分条件
            if self._should_split(current_group, entries, i):
                # 创建Segment
                segment = self._create_segment(current_group)
                segments.append(segment)
                
                # 重置当前组
                current_group = []
            
            i += 1
        
        # 处理剩余的Entry
        if current_group:
            if segments:
                # 检查是否合并到最后一个Segment
                last_seg = segments[-1]
                if (last_seg.duration + self._group_duration(current_group) <= self.max_duration and
                    len(current_group) < self.min_entries):
                    # 合并
                    merged_entries = self._get_entries_by_range(
                        entries, last_seg.srt_start_idx, current_group[-1].index
                    )
                    segments[-1] = self._create_segment(merged_entries)
                else:
                    segments.append(self._create_segment(current_group))
            else:
                segments.append(self._create_segment(current_group))
        
        return segments
    
    def _should_split(self, current_group: List[SRTEntry], 
                      all_entries: List[SRTEntry],
                      current_idx: int) -> bool:
        """
        判断是否应该在此处切分
        
        条件（满足任一）：
        1. 达到max_duration，必须切分
        2. 达到min_duration，且检测到语义边界
        3. 已累积足够Entry，且下一个是明显的章节开头
        """
        current_duration = self._group_duration(current_group)
        
        # 条件1：超过最大时长，必须切分
        if current_duration >= self.max_duration:
            return True
        
        # 未达到最小时长，不切分
        if current_duration < self.min_duration:
            return False
        
        # 未达到最少Entry，不切分
        if len(current_group) < self.min_entries:
            return False
        
        # 已经是最后一个Entry，不切分
        if current_idx >= len(all_entries) - 1:
            return False
        
        # 条件2：检测语义边界
        # 预读后续Entry作为"下一段"
        next_entries = all_entries[current_idx + 1:current_idx + 1 + self.min_entries]
        
        if len(next_entries) < self.min_entries:
            # 后续Entry不够，不切分（等最后统一处理）
            return False
        
        # 用分割器判断是否是边界
        try:
            is_boundary = self.segmenter.is_boundary(current_group, next_entries)
            if is_boundary:
                return True
        except Exception as e:
            logger.warning(f"边界检测失败: {e}")
        
        return False
    
    def _create_segment(self, entries: List[SRTEntry]) -> Segment:
        """从Entry组创建Segment"""
        start_ms = entries[0].start_ms
        end_ms = entries[-1].end_ms
        text = " ".join(e.text for e in entries)
        
        return Segment(
            id=f"seg_{entries[0].index:04d}",
            time_start=start_ms / 1000,
            time_end=end_ms / 1000,
            duration=(end_ms - start_ms) / 1000,
            srt_start_idx=entries[0].index,
            srt_end_idx=entries[-1].index,
            text=text,
            keywords=self._extract_keywords(text),
            features=self._extract_features(text)
        )
    
    def _group_duration(self, entries: List[SRTEntry]) -> float:
        """计算Entry组的总时长（秒）"""
        if not entries:
            return 0
        return (entries[-1].end_ms - entries[0].start_ms) / 1000
    
    def _get_entries_by_range(self, all_entries: List[SRTEntry], 
                               start_idx: int, end_idx: int) -> List[SRTEntry]:
        """获取指定index范围的Entry"""
        return [e for e in all_entries if start_idx <= e.index <= end_idx]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简单TF-IDF或词频）"""
        # TODO: 实现关键词提取
        # 暂时返回空列表，后续可接入jieba等分词工具
        return []
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """提取文本特征（规则-based，无LLM）"""
        features = {
            "has_code": bool(re.search(r'\b(def |class |import |from \w+ import|print\(|if __name__)', text)),
            "has_command": bool(re.search(r'^[\$#>]\s+\w', text, re.MULTILINE)),
            "has_url": re.findall(r'https?://[^\s<>"\']+|github\.com/[^\s<>"\']+', text),
            "word_count": len(text),
            "sentence_count": text.count('。') + text.count('？') + text.count('！') + text.count('.')
        }
        return features
