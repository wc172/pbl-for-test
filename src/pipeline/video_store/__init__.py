"""
视频内容存储与检索模块（模块4）

提供视频内容的结构化存储和原子检索工具
"""

from src.pipeline.video_store.builder import SegmentBuilder, Segment
from src.pipeline.video_store.storage import VideoStorage
from src.pipeline.video_store.interface import VideoContentStore
from src.pipeline.video_store.segmenter import SeqModelSegmenter, EntryGroupSegmenter

__all__ = [
    'SegmentBuilder',
    'Segment',
    'VideoStorage',
    'VideoContentStore',
    'SeqModelSegmenter',
    'EntryGroupSegmenter',
]
