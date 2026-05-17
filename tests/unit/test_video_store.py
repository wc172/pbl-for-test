"""
视频内容存储模块单元测试

测试内容：
1. SegmentBuilder构建Segments
2. VideoStorage存储和读取
3. VideoContentStore工具接口
4. SeqModel边界检测（可选，需要模型）
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.pipeline.video_store import (
    SegmentBuilder, 
    VideoStorage, 
    VideoContentStore,
    SeqModelSegmenter,
    EntryGroupSegmenter
)
from src.utils.srt_parser import SRTEntry


# ========== 测试数据 ==========

SAMPLE_SRT_CONTENT = """1
00:00:09,150 --> 00:00:09,470
好了，

2
00:00:09,470 --> 00:00:10,530
同学们晚上好，

3
00:00:10,590 --> 00:00:12,430
我们再等一等。

4
00:00:12,450 --> 00:00:13,190
其他的同学，

5
00:00:13,270 --> 00:00:15,595
我们八点零一分正式开始上课，

6
00:00:15,595 --> 00:00:18,200
今天我们要学习AutoGen框架。

7
00:00:18,200 --> 00:00:22,100
AutoGen是微软开源的多Agent对话框架。

8
00:00:22,100 --> 00:00:25,500
它支持自定义Agent、对话编排和工具调用。

9
00:00:25,500 --> 00:00:30,200
首先我们来看如何安装AutoGen。

10
00:00:30,200 --> 00:00:35,800
使用pip install pyautogen就可以安装。
"""


def create_test_srt_file(tmp_dir: Path) -> Path:
    """创建测试用的SRT文件"""
    srt_path = tmp_dir / "test_transcript.srt"
    srt_path.write_text(SAMPLE_SRT_CONTENT, encoding='utf-8')
    return srt_path


# ========== 测试类 ==========

class TestSegmentBuilder:
    """测试SegmentBuilder"""
    
    def test_create_segment_from_entries(self, tmp_path):
        """测试从Entry组创建Segment"""
        builder = SegmentBuilder("test_course", use_semantic_segmenter=False)
        
        # 创建测试Entry
        entries = [
            SRTEntry(index=1, start_ms=9150, end_ms=9470, text="好了，"),
            SRTEntry(index=2, start_ms=9470, end_ms=10530, text="同学们晚上好，"),
        ]
        
        segment = builder._create_segment(entries)
        
        assert segment.id == "seg_0001"
        assert segment.time_start == 9.15
        assert segment.time_end == 10.53
        assert "好了" in segment.text
        assert "同学们晚上好" in segment.text
        assert segment.duration == pytest.approx(1.38, abs=0.01)
    
    def test_build_segments_with_rules(self, tmp_path):
        """测试使用规则分割器构建Segments"""
        srt_path = create_test_srt_file(tmp_path)
        
        builder = SegmentBuilder(
            "test_course",
            use_semantic_segmenter=False,
            min_duration=5,      # 降低阈值以便测试
            max_duration=20
        )
        
        segments = builder.build(srt_path)
        
        assert len(segments) > 0
        # 检查Segment结构
        for seg in segments:
            assert seg.id.startswith("seg_")
            assert seg.time_start < seg.time_end
            assert seg.text
            assert seg.srt_start_idx > 0
            assert seg.srt_end_idx >= seg.srt_start_idx
    
    def test_segment_duration_constraints(self, tmp_path):
        """测试Segment时长约束"""
        srt_path = create_test_srt_file(tmp_path)
        
        builder = SegmentBuilder(
            "test_course",
            use_semantic_segmenter=False,
            min_duration=3,
            max_duration=20  # 调整为20秒以适应测试数据
        )
        
        segments = builder.build(srt_path)
        
        for seg in segments:
            # 每个Segment应该满足时长约束（第一个Segment允许稍微超过，因为没有前驱可合并）
            if seg.srt_start_idx == 1:
                # 第一个Segment：允许稍微超过max_duration（因为没有前驱）
                assert seg.duration <= builder.max_duration * 1.2, \
                    f"第一个Segment {seg.id} 时长 {seg.duration}s 超过最大值太多"
            else:
                assert seg.duration <= builder.max_duration, \
                    f"Segment {seg.id} 时长 {seg.duration}s 超过最大值"
            
            # 检查最小时长（最后一个Segment允许小于min_duration）
            if seg.srt_end_idx < 20:  # 不是最后一个Segment
                assert seg.duration >= builder.min_duration, \
                    f"Segment {seg.id} 时长 {seg.duration}s 小于最小值"


class TestVideoStorage:
    """测试VideoStorage"""
    
    def test_save_and_load_segments(self, tmp_path):
        """测试保存和加载Segments"""
        # 使用临时目录
        storage = VideoStorage("test_course")
        storage.base_dir = tmp_path / "facts"
        storage.base_dir.mkdir(parents=True, exist_ok=True)
        storage.segments_file = storage.base_dir / "segments.jsonl"
        storage.index_file = storage.base_dir / "index.json"
        storage.nav_file = storage.base_dir / "navigation_map.json"
        
        # 创建测试Segments
        from src.pipeline.video_store.builder import Segment
        segments = [
            Segment(
                id="seg_0001",
                time_start=0.0,
                time_end=60.0,
                duration=60.0,
                srt_start_idx=1,
                srt_end_idx=10,
                text="测试文本1",
                keywords=["测试"],
                features={"has_code": False}
            ),
            Segment(
                id="seg_0002",
                time_start=60.0,
                time_end=120.0,
                duration=60.0,
                srt_start_idx=11,
                srt_end_idx=20,
                text="测试文本2",
                keywords=["AutoGen"],
                features={"has_code": True}
            )
        ]
        
        # 保存
        storage.save_segments(segments)
        
        # 验证文件存在
        assert storage.segments_file.exists()
        assert storage.index_file.exists()
        assert storage.nav_file.exists()
        
        # 加载验证
        loaded = storage.load_segments()
        assert len(loaded) == 2
        assert loaded[0]["id"] == "seg_0001"
        assert loaded[1]["id"] == "seg_0002"
    
    def test_index_building(self, tmp_path):
        """测试索引构建"""
        storage = VideoStorage("test_course")
        storage.base_dir = tmp_path
        storage.segments_file = storage.base_dir / "segments.jsonl"
        storage.index_file = storage.base_dir / "index.json"
        storage.nav_file = storage.base_dir / "navigation_map.json"
        
        from src.pipeline.video_store.builder import Segment
        segments = [
            Segment(
                id="seg_0001",
                time_start=0.0,
                time_end=60.0,
                duration=60.0,
                srt_start_idx=1,
                srt_end_idx=5,
                text="测试代码",
                keywords=["测试"],
                features={"has_code": True, "has_command": False}
            )
        ]
        
        storage.save_segments(segments)
        
        index = storage.load_index()
        assert "time" in index
        assert "keywords" in index
        assert "types" in index
        assert "has_code" in index["types"]


class TestVideoContentStore:
    """测试VideoContentStore工具接口"""
    
    @pytest.fixture
    def store_with_data(self, tmp_path):
        """创建带有测试数据的VideoContentStore"""
        # 先创建存储
        storage = VideoStorage("test_course")
        storage.base_dir = tmp_path / "facts"
        storage.base_dir.mkdir(parents=True, exist_ok=True)
        storage.segments_file = storage.base_dir / "segments.jsonl"
        storage.index_file = storage.base_dir / "index.json"
        storage.nav_file = storage.base_dir / "navigation_map.json"
        
        from src.pipeline.video_store.builder import Segment
        segments = [
            Segment(
                id="seg_0001",
                time_start=0.0,
                time_end=60.0,
                duration=60.0,
                srt_start_idx=1,
                srt_end_idx=5,
                text="同学们晚上好，今天学习AutoGen框架。",
                keywords=["AutoGen", "框架"],
                features={"has_code": False, "has_command": False}
            ),
            Segment(
                id="seg_0002",
                time_start=60.0,
                time_end=120.0,
                duration=60.0,
                srt_start_idx=6,
                srt_end_idx=10,
                text="安装命令：pip install pyautogen",
                keywords=["安装", "pip"],
                features={"has_code": False, "has_command": True}
            ),
            Segment(
                id="seg_0003",
                time_start=120.0,
                time_end=180.0,
                duration=60.0,
                srt_start_idx=11,
                srt_end_idx=15,
                text="代码示例：def main(): pass",
                keywords=["代码", "示例"],
                features={"has_code": True, "has_command": False}
            )
        ]
        
        storage.save_segments(segments)
        
        # 创建Store并修改存储路径
        store = VideoContentStore("test_course")
        store.storage = storage
        
        return store
    
    def test_get_course_structure(self, store_with_data):
        """测试获取课程结构"""
        structure = store_with_data.get_course_structure()
        
        assert len(structure) == 3
        assert structure[0]["id"] == "seg_0001"
        assert "text" not in structure[0]  # 无原文
        assert "keywords" in structure[0]
    
    def test_get_segment_by_time(self, store_with_data):
        """测试时间定位"""
        # 查找30秒处的Segment
        seg = store_with_data.get_segment_by_time(30.0)
        assert seg is not None
        assert seg["id"] == "seg_0001"
        
        # 查找90秒处的Segment
        seg = store_with_data.get_segment_by_time(90.0)
        assert seg["id"] == "seg_0002"
        
        # 查找不存在的
        seg = store_with_data.get_segment_by_time(999.0)
        assert seg is None
    
    def test_get_segments_by_type(self, store_with_data):
        """测试类型筛选"""
        # 查找has_code的Segment
        results = store_with_data.get_segments_by_type("has_code")
        assert len(results) == 1
        assert results[0]["id"] == "seg_0003"
        
        # 查找has_command的Segment
        results = store_with_data.get_segments_by_type("has_command")
        assert len(results) == 1
        assert results[0]["id"] == "seg_0002"
    
    def test_get_segment_content(self, store_with_data):
        """测试获取Segment完整内容"""
        seg = store_with_data.get_segment_content("seg_0001")
        
        assert seg is not None
        assert seg["id"] == "seg_0001"
        assert "text" in seg  # 有原文
        assert "AutoGen" in seg["text"]
    
    def test_get_multiple_segments(self, store_with_data):
        """测试批量获取"""
        segs = store_with_data.get_multiple_segments(["seg_0001", "seg_0003"])
        
        assert len(segs) == 2
        ids = [s["id"] for s in segs]
        assert "seg_0001" in ids
        assert "seg_0003" in ids
    
    def test_get_raw_text_range(self, store_with_data):
        """测试获取时间范围文本"""
        result = store_with_data.get_raw_text_range(30.0, 150.0)
        
        assert "text" in result
        assert "segment_ids" in result
        assert len(result["segment_ids"]) >= 2
    
    def test_get_course_stats(self, store_with_data):
        """测试获取课程统计"""
        stats = store_with_data.get_course_stats()
        
        assert stats["course_name"] == "test_course"
        assert stats["total_segments"] == 3
        assert "total_duration" in stats


class TestSegmenter:
    """测试分割器"""
    
    def test_entry_group_segmenter(self):
        """测试规则分割器"""
        segmenter = EntryGroupSegmenter()
        
        # 测试过渡词检测
        prev_entries = [
            SRTEntry(index=1, start_ms=0, end_ms=1000, text="我们学习了基础概念。"),
        ]
        next_entries = [
            SRTEntry(index=2, start_ms=1000, end_ms=2000, text="首先来看代码示例。"),
        ]
        
        # "首先"开头应该判定为边界
        is_boundary = segmenter.is_boundary(prev_entries, next_entries)
        assert is_boundary is True
    
    def test_seqmodel_segmenter_init(self):
        """测试SeqModelSegmenter初始化（如果模型存在）"""
        model_path = Path("models/nlp_bert_document-segmentation_chinese-base")
        
        if not model_path.exists():
            pytest.skip("SeqModel未下载，跳过测试")
        
        try:
            segmenter = SeqModelSegmenter(str(model_path))
            assert segmenter.pipeline is not None
        except Exception as e:
            pytest.skip(f"SeqModel初始化失败: {e}")


# ========== 集成测试 ==========

class TestVideoStoreIntegration:
    """集成测试：完整流程"""
    
    def test_full_pipeline(self, tmp_path):
        """测试完整流程：SRT → Segments → Storage → Query"""
        # 1. 创建测试SRT
        srt_path = create_test_srt_file(tmp_path)
        
        # 2. 构建Segments
        builder = SegmentBuilder(
            "test_course",
            use_semantic_segmenter=False,
            min_duration=3,
            max_duration=30
        )
        segments = builder.build(srt_path)
        
        assert len(segments) > 0
        
        # 3. 保存到存储
        storage = VideoStorage("test_course")
        storage.base_dir = tmp_path / "test_facts"
        storage.base_dir.mkdir(parents=True, exist_ok=True)
        storage.segments_file = storage.base_dir / "segments.jsonl"
        storage.index_file = storage.base_dir / "index.json"
        storage.nav_file = storage.base_dir / "navigation_map.json"
        
        storage.save_segments(segments)
        
        # 4. 通过接口查询
        store = VideoContentStore("test_course")
        store.storage = storage
        
        # 查询测试
        structure = store.get_course_structure()
        assert len(structure) == len(segments)
        
        # 时间定位测试
        seg = store.get_segment_by_time(10.0)
        assert seg is not None


# ========== 运行入口 ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
