"""
视频内容存储管理

JSON Lines格式：轻量、可读、流式处理
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.pipeline.video_store.builder import Segment

logger = logging.getLogger(__name__)


class VideoStorage:
    """
    视频内容存储管理
    
    存储结构：
        .cache/{course_name}/facts/
        ├── segments.jsonl      # Segment主存储
        ├── index.json          # 结构化索引
        └── navigation_map.json # 导航图
    """
    
    def __init__(self, course_name: str):
        self.course_name = course_name
        self.base_dir = Path(f".cache/{course_name}/facts")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.segments_file = self.base_dir / "segments.jsonl"
        self.index_file = self.base_dir / "index.json"
        self.nav_file = self.base_dir / "navigation_map.json"
    
    def save_segments(self, segments: List[Segment]) -> None:
        """
        保存Segments到JSON Lines文件
        
        Args:
            segments: Segment列表
        """
        logger.info(f"保存 {len(segments)} 个Segments")
        
        # 写入segments.jsonl
        with open(self.segments_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(json.dumps(segment.to_dict(), ensure_ascii=False) + '\n')
        
        # 构建并保存索引
        index = self._build_index(segments)
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        # 构建并保存导航图
        nav_map = self._build_navigation_map(segments)
        with open(self.nav_file, 'w', encoding='utf-8') as f:
            json.dump(nav_map, f, ensure_ascii=False, indent=2)
        
        logger.info(f"存储完成: {self.base_dir}")
    
    def load_segments(self) -> List[Dict[str, Any]]:
        """
        加载所有Segments
        
        Returns:
            Segment字典列表
        """
        if not self.segments_file.exists():
            return []
        
        segments = []
        with open(self.segments_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    segments.append(json.loads(line))
        
        return segments
    
    def load_index(self) -> Dict[str, Any]:
        """加载索引"""
        if not self.index_file.exists():
            return {}
        
        with open(self.index_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_navigation_map(self) -> Dict[str, Any]:
        """加载导航图"""
        if not self.nav_file.exists():
            return {}
        
        with open(self.nav_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def exists(self) -> bool:
        """检查存储是否存在"""
        return self.segments_file.exists()
    
    def _build_index(self, segments: List[Segment]) -> Dict[str, Any]:
        """
        构建多级索引
        
        索引结构：
            {
                "time": {timestamp: segment_id},      # 时间索引
                "keywords": {keyword: [segment_ids]}, # 关键词倒排索引
                "types": {type: [segment_ids]}        # 类型索引
            }
        """
        time_index = {}
        keyword_index = {}
        type_index = {
            "has_code": [],
            "has_command": [],
            "has_url": []
        }
        
        for seg in segments:
            seg_dict = seg.to_dict()
            
            # 时间索引：每分钟一个标记点
            for t in range(int(seg.time_start), int(seg.time_end) + 1, 60):
                time_index[t] = seg.id
            
            # 关键词索引
            for kw in seg.keywords:
                if kw not in keyword_index:
                    keyword_index[kw] = []
                keyword_index[kw].append(seg.id)
            
            # 类型索引
            features = seg.features
            if features.get("has_code"):
                type_index["has_code"].append(seg.id)
            if features.get("has_command"):
                type_index["has_command"].append(seg.id)
            if features.get("has_url"):
                type_index["has_url"].append(seg.id)
        
        return {
            "time": time_index,
            "keywords": keyword_index,
            "types": type_index,
            "total_segments": len(segments),
            "total_duration": sum(s.duration for s in segments)
        }
    
    def _build_navigation_map(self, segments: List[Segment]) -> Dict[str, Any]:
        """
        构建导航图
        
        包含：
            - 课程概览
            - Segment列表（仅元数据）
            - 概念时间线
        """
        if not segments:
            return {}
        
        return {
            "course_name": self.course_name,
            "total_segments": len(segments),
            "total_duration": sum(s.duration for s in segments),
            "time_range": {
                "start": segments[0].time_start,
                "end": segments[-1].time_end
            },
            "segment_list": [
                {
                    "id": s.id,
                    "time_start": s.time_start,
                    "time_end": s.time_end,
                    "duration": s.duration,
                    "keywords": s.keywords,
                    "features": s.features
                }
                for s in segments
            ],
            "concept_timeline": self._build_concept_timeline(segments)
        }
    
    def _build_concept_timeline(self, segments: List[Segment]) -> List[Dict[str, Any]]:
        """构建概念时间线"""
        timeline = []
        
        for seg in segments:
            for kw in seg.keywords:
                timeline.append({
                    "concept": kw,
                    "time": seg.time_start,
                    "segment_id": seg.id
                })
        
        # 按时间排序
        timeline.sort(key=lambda x: x["time"])
        return timeline
