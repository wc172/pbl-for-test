"""
视频内容存储接口

提供9个原子工具接口（供模块5 MCP服务调用）
- 定位层：4个
- 提取层：3个
- 元数据层：2个
- 全局层：1个
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.pipeline.video_store.storage import VideoStorage

logger = logging.getLogger(__name__)


class VideoContentStore:
    """
    视频内容存储与检索核心类
    
    职责：
        - 视频内容的事实存储与多维度检索
        - 提供原子工具接口（无业务组合逻辑）
    
    内部依赖：
        - VideoStorage: JSON存储管理
    """
    
    def __init__(self, course_name: str):
        self.course_name = course_name
        self.storage = VideoStorage(course_name)
        self._segments_cache: Optional[List[Dict]] = None
        self._index_cache: Optional[Dict] = None
    
    def _load_segments(self) -> List[Dict[str, Any]]:
        """懒加载Segments"""
        if self._segments_cache is None:
            self._segments_cache = self.storage.load_segments()
        return self._segments_cache
    
    def _load_index(self) -> Dict[str, Any]:
        """懒加载索引"""
        if self._index_cache is None:
            self._index_cache = self.storage.load_index()
        return self._index_cache
    
    def exists(self) -> bool:
        """检查课程视频存储是否存在"""
        return self.storage.exists()
    
    # ========== 定位层工具（4个）==========
    
    def get_course_structure(self) -> List[Dict[str, Any]]:
        """
        获取课程结构导航
        
        Returns:
            所有Segment的元数据列表（无原文）
        """
        segments = self._load_segments()
        return [
            {
                "id": s["id"],
                "time_start": s["time_start"],
                "time_end": s["time_end"],
                "duration": s["duration"],
                "keywords": s.get("keywords", []),
                "features": s.get("features", {})
            }
            for s in segments
        ]
    
    def get_segment_by_time(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """
        时间戳定位：返回指定时间所在的Segment
        
        Args:
            timestamp: 时间戳（秒）
            
        Returns:
            Segment字典，未找到返回None
        """
        segments = self._load_segments()
        
        for seg in segments:
            if seg["time_start"] <= timestamp <= seg["time_end"]:
                return seg
        
        return None
    
    def get_segments_by_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        概念定位：返回包含该关键词的所有Segment元数据
        
        Args:
            concept: 概念关键词
            
        Returns:
            Segment元数据列表（无原文）
        """
        segments = self._load_segments()
        index = self._load_index()
        
        # 从索引中查找
        keyword_index = index.get("keywords", {})
        seg_ids = keyword_index.get(concept, [])
        
        # 回退：遍历查找（如果索引未命中）
        if not seg_ids:
            seg_ids = [
                s["id"] for s in segments
                if concept.lower() in s["text"].lower() or
                   concept.lower() in [k.lower() for k in s.get("keywords", [])]
            ]
        
        # 返回元数据（无原文）
        return [
            {
                "id": s["id"],
                "time_start": s["time_start"],
                "time_end": s["time_end"],
                "duration": s["duration"],
                "keywords": s.get("keywords", []),
                "features": s.get("features", {})
            }
            for s in segments if s["id"] in seg_ids
        ]
    
    def get_segments_by_type(self, content_type: str) -> List[Dict[str, Any]]:
        """
        类型筛选：如has_code, has_command等
        
        Args:
            content_type: 内容类型（has_code, has_command, has_url）
            
        Returns:
            Segment元数据列表（无原文）
        """
        segments = self._load_segments()
        index = self._load_index()
        
        # 从索引中查找
        type_index = index.get("types", {})
        seg_ids = type_index.get(content_type, [])
        
        # 回退：遍历查找
        if not seg_ids:
            seg_ids = [
                s["id"] for s in segments
                if s.get("features", {}).get(content_type, False)
            ]
        
        return [
            {
                "id": s["id"],
                "time_start": s["time_start"],
                "time_end": s["time_end"],
                "duration": s["duration"],
                "keywords": s.get("keywords", []),
                "features": s.get("features", {})
            }
            for s in segments if s["id"] in seg_ids
        ]
    
    # ========== 提取层工具（3个）==========
    
    def get_segment_content(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """
        获取单个Segment完整内容（含原文）
        
        Args:
            segment_id: Segment ID
            
        Returns:
            完整Segment字典，未找到返回None
        """
        segments = self._load_segments()
        
        for seg in segments:
            if seg["id"] == segment_id:
                return seg
        
        return None
    
    def get_multiple_segments(self, segment_ids: List[str]) -> List[Dict[str, Any]]:
        """
        批量获取Segment内容（性能优化）
        
        Args:
            segment_ids: Segment ID列表
            
        Returns:
            Segment字典列表
        """
        segments = self._load_segments()
        seg_map = {s["id"]: s for s in segments}
        
        return [seg_map[sid] for sid in segment_ids if sid in seg_map]
    
    def get_raw_text_range(self, start_sec: float, end_sec: float) -> Dict[str, Any]:
        """
        获取时间范围内的原始文本拼接
        
        Args:
            start_sec: 开始时间（秒）
            end_sec: 结束时间（秒）
            
        Returns:
            {
                "text": "拼接后的文本",
                "segment_ids": [涉及的Segment ID列表],
                "time_start": 实际开始时间,
                "time_end": 实际结束时间
            }
        """
        segments = self._load_segments()
        
        # 找出与时间范围重叠的Segments
        overlapped = [
            s for s in segments
            if not (s["time_end"] < start_sec or s["time_start"] > end_sec)
        ]
        
        if not overlapped:
            return {
                "text": "",
                "segment_ids": [],
                "time_start": start_sec,
                "time_end": end_sec
            }
        
        # 拼接文本
        texts = []
        for seg in overlapped:
            # 如果是部分重叠，可以裁剪文本（简化处理：返回完整Segment文本）
            seg_text = seg["text"]
            texts.append(seg_text)
        
        return {
            "text": " ".join(texts),
            "segment_ids": [s["id"] for s in overlapped],
            "time_start": max(start_sec, overlapped[0]["time_start"]),
            "time_end": min(end_sec, overlapped[-1]["time_end"])
        }
    
    # ========== 元数据层工具（2个）==========
    
    def get_segment_metadata(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """
        获取Segment轻量级元数据（无原文）
        
        Args:
            segment_id: Segment ID
            
        Returns:
            元数据字典，未找到返回None
        """
        segments = self._load_segments()
        
        for seg in segments:
            if seg["id"] == segment_id:
                return {
                    "id": seg["id"],
                    "time_start": seg["time_start"],
                    "time_end": seg["time_end"],
                    "duration": seg["duration"],
                    "srt_start_idx": seg["srt_start_idx"],
                    "srt_end_idx": seg["srt_end_idx"],
                    "keywords": seg.get("keywords", []),
                    "features": seg.get("features", {})
                }
        
        return None
    
    def get_course_stats(self) -> Dict[str, Any]:
        """
        获取课程统计信息
        
        Returns:
            统计信息字典
        """
        index = self._load_index()
        nav_map = self.storage.load_navigation_map()
        
        return {
            "course_name": self.course_name,
            "total_segments": index.get("total_segments", 0),
            "total_duration": index.get("total_duration", 0),
            "time_range": nav_map.get("time_range", {}),
            "keyword_distribution": {
                kw: len(ids) for kw, ids in index.get("keywords", {}).items()
            },
            "type_distribution": {
                t: len(ids) for t, ids in index.get("types", {}).items()
            }
        }
    
    # ========== 全局层工具（1个）==========
    
    def get_navigation_map(self) -> Dict[str, Any]:
        """
        获取全局导航图
        
        Returns:
            导航图字典
        """
        return self.storage.load_navigation_map()
