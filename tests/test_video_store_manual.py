"""
视频存储模块手动测试脚本

快速验证模块4功能是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.video_store import SegmentBuilder, VideoStorage, VideoContentStore


def test_basic_build():
    """测试基础构建功能"""
    print("=" * 60)
    print("测试1: SegmentBuilder基础构建")
    print("=" * 60)
    
    srt_path = Path("tests/test_data/sample_transcript.srt")
    if not srt_path.exists():
        print(f"❌ 测试SRT文件不存在: {srt_path}")
        return False
    
    # 使用规则分割器（无需下载模型）
    builder = SegmentBuilder(
        "test_course",
        use_semantic_segmenter=False,  # 使用规则分割，不依赖SeqModel
        min_duration=5,      # 降低阈值以便测试
        max_duration=30
    )
    
    print(f"正在构建Segments...")
    segments = builder.build(srt_path)
    
    print(f"✅ 构建完成！共 {len(segments)} 个Segments")
    
    for seg in segments:
        print(f"\n  Segment: {seg.id}")
        print(f"    时间: {seg.time_start:.1f}s - {seg.time_end:.1f}s (时长: {seg.duration:.1f}s)")
        print(f"    SRT索引: {seg.srt_start_idx} - {seg.srt_end_idx}")
        print(f"    文本: {seg.text[:50]}...")
        print(f"    关键词: {seg.keywords}")
        print(f"    特征: {seg.features}")
    
    return segments


def test_storage(segments):
    """测试存储功能"""
    print("\n" + "=" * 60)
    print("测试2: VideoStorage存储")
    print("=" * 60)
    
    storage = VideoStorage("test_course")
    storage.save_segments(segments)
    
    print(f"✅ 存储完成！")
    print(f"  存储路径: {storage.base_dir}")
    print(f"  segments.jsonl: {storage.segments_file.exists()}")
    print(f"  index.json: {storage.index_file.exists()}")
    print(f"  navigation_map.json: {storage.nav_file.exists()}")
    
    # 验证加载
    loaded = storage.load_segments()
    print(f"✅ 加载验证: {len(loaded)} 个Segments")
    
    return storage


def test_interface(storage):
    """测试接口功能"""
    print("\n" + "=" * 60)
    print("测试3: VideoContentStore工具接口")
    print("=" * 60)
    
    store = VideoContentStore("test_course")
    
    # 测试1: 获取课程结构
    print("\n  测试 get_course_structure:")
    structure = store.get_course_structure()
    print(f"    ✅ 获取到 {len(structure)} 个Segment元数据")
    
    # 测试2: 时间定位
    print("\n  测试 get_segment_by_time:")
    seg = store.get_segment_by_time(20.0)  # 查找20秒处
    if seg:
        print(f"    ✅ 20秒处找到Segment: {seg['id']}")
    else:
        print(f"    ⚠️  20秒处未找到Segment")
    
    # 测试3: 类型筛选
    print("\n  测试 get_segments_by_type:")
    code_segs = store.get_segments_by_type("has_code")
    print(f"    ✅ 包含代码的Segments: {len(code_segs)} 个")
    
    # 测试4: 获取内容
    print("\n  测试 get_segment_content:")
    if structure:
        first_id = structure[0]['id']
        content = store.get_segment_content(first_id)
        if content:
            print(f"    ✅ 获取到 {first_id} 的内容: {content['text'][:30]}...")
    
    # 测试5: 时间范围文本
    print("\n  测试 get_raw_text_range:")
    range_result = store.get_raw_text_range(10.0, 40.0)
    print(f"    ✅ 获取10-40秒范围文本，涉及 {len(range_result['segment_ids'])} 个Segments")
    
    # 测试6: 课程统计
    print("\n  测试 get_course_stats:")
    stats = store.get_course_stats()
    print(f"    ✅ 课程统计: {stats}")


def test_with_seqmodel():
    """测试SeqModel语义分割（如果模型存在）"""
    print("\n" + "=" * 60)
    print("测试4: SeqModel语义分割（可选）")
    print("=" * 60)
    
    model_path = Path("models/nlp_bert_document-segmentation_chinese-base")
    if not model_path.exists():
        print(f"  ⚠️  SeqModel未找到，跳过此测试")
        print(f"     模型路径: {model_path}")
        return
    
    print(f"  发现SeqModel，尝试使用...")
    
    try:
        from src.pipeline.video_store.segmenter import SeqModelSegmenter
        segmenter = SeqModelSegmenter(str(model_path))
        print(f"  ✅ SeqModel初始化成功")
        
        # 测试边界检测
        from src.utils.srt_parser import SRTEntry
        
        prev_entries = [
            SRTEntry(index=1, start_ms=0, end_ms=5000, text="今天我们学习Python基础。"),
            SRTEntry(index=2, start_ms=5000, end_ms=10000, text="变量是编程的基本概念。"),
        ]
        next_entries = [
            SRTEntry(index=3, start_ms=10000, end_ms=15000, text="接下来我们学习函数定义。"),
        ]
        
        is_boundary = segmenter.is_boundary(prev_entries, next_entries)
        print(f"  ✅ 边界检测测试: {'是边界' if is_boundary else '不是边界'}")
        
        # 使用SeqModel构建
        srt_path = Path("tests/test_data/sample_transcript.srt")
        builder = SegmentBuilder(
            "test_course_seq",
            use_semantic_segmenter=True,
            min_duration=5,
            max_duration=30
        )
        
        segments = builder.build(srt_path)
        print(f"  ✅ SeqModel构建完成: {len(segments)} 个Segments")
        
    except Exception as e:
        print(f"  ❌ SeqModel测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("视频内容存储模块（模块4）功能测试")
    print("=" * 60)
    
    try:
        # 1. 构建Segments
        segments = test_basic_build()
        if not segments:
            print("\n❌ 构建Segments失败，终止测试")
            return
        
        # 2. 测试存储
        storage = test_storage(segments)
        
        # 3. 测试接口
        test_interface(storage)
        
        # 4. 测试SeqModel（可选）
        test_with_seqmodel()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
