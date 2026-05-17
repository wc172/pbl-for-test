"""
MCP工具定义 - 11个工具Schema

视频工具（9个）+ 课件工具（2个）
工具的实际实现逻辑在 server.py 中
"""

from typing import List, Dict, Any, Optional

# 工具描述常量，供FastMCP装饰器使用

# ========== 视频工具（9个）==========

VIDEO_GET_COURSE_STRUCTURE_DESC = """
获取视频课程结构导航

返回所有Segment的元数据列表（无原文），用于了解课程整体结构。

Args:
    course_name: 课程名称（如"A3.11_autogen"）
    
Returns:
    Segment元数据列表，每项包含：
    - id: Segment ID（如"seg_001"）
    - time_start: 开始时间（秒）
    - time_end: 结束时间（秒）
    - duration: 时长（秒）
    - keywords: 关键词列表
    - features: 特征标记（has_code等）
"""

VIDEO_GET_SEGMENT_BY_TIME_DESC = """
根据时间戳定位Segment

返回指定时间所在的完整Segment内容（含原文）。

Args:
    course_name: 课程名称
    timestamp: 时间戳（秒，如123.5表示2分3秒）
    
Returns:
    Segment完整数据，包含text原文，未找到返回null
"""

VIDEO_GET_SEGMENTS_BY_CONCEPT_DESC = """
根据概念关键词定位Segments

返回包含该关键词的所有Segment元数据（无原文）。

Args:
    course_name: 课程名称
    concept: 概念关键词（如"A3.11_autogen"、"神经网络"）
    
Returns:
    Segment元数据列表（无原文）
"""

VIDEO_GET_SEGMENTS_BY_TYPE_DESC = """
根据内容类型筛选Segments

Args:
    course_name: 课程名称
    content_type: 内容类型，可选值：
        - "has_code": 包含代码
        - "has_command": 包含命令行
        - "has_url": 包含URL
        
Returns:
    Segment元数据列表（无原文）
"""

VIDEO_GET_SEGMENT_CONTENT_DESC = """
获取单个Segment完整内容（含原文）

Args:
    course_name: 课程名称
    segment_id: Segment ID（如"seg_001"）
    
Returns:
    Segment完整内容，包含text原文，未找到返回null
"""

VIDEO_GET_MULTIPLE_SEGMENTS_DESC = """
批量获取多个Segments内容

性能优化接口，减少IO次数。

Args:
    course_name: 课程名称
    segment_ids: Segment ID列表（如["seg_001", "seg_002"]）
    
Returns:
    Segment完整内容列表
"""

VIDEO_GET_RAW_TEXT_RANGE_DESC = """
获取时间范围内的原始文本拼接

Args:
    course_name: 课程名称
    start_sec: 开始时间（秒）
    end_sec: 结束时间（秒）
    
Returns:
    {
        "text": "拼接后的完整文本",
        "segment_ids": ["seg_001", "seg_002"],
        "time_start": 实际开始时间,
        "time_end": 实际结束时间
    }
"""

VIDEO_GET_SEGMENT_METADATA_DESC = """
获取Segment轻量级元数据（无原文）

用于快速获取信息，不包含text字段。

Args:
    course_name: 课程名称
    segment_id: Segment ID
    
Returns:
    元数据字典（无原文），包含：
    - id, time_start, time_end, duration
    - srt_start_idx, srt_end_idx
    - keywords, features
"""

VIDEO_GET_COURSE_STATS_DESC = """
获取课程统计信息

Args:
    course_name: 课程名称
    
Returns:
    统计信息字典：
    - total_segments: 总Segment数
    - total_duration: 总时长（秒）
    - keyword_distribution: 关键词分布
    - type_distribution: 类型分布
"""

VIDEO_GET_NAVIGATION_MAP_DESC = """
获取全局导航图

返回完整的课程导航信息，包含课程概览、Segment列表、概念时间线。

Args:
    course_name: 课程名称
    
Returns:
    导航图字典：
    - course_profile: 课程概览
    - segment_list: Segment列表
    - concept_timeline: 概念时间线
"""

# ========== 课件工具（2个）==========

MATERIAL_SEARCH_DESC = """
语义搜索课件内容

使用向量检索返回最相关的课件chunks。

Args:
    course_name: 课程名称
    query: 查询文本（如"AutoGen多智能体"）
    top_k: 返回结果数量（默认3，最大10）
    
Returns:
    {
        "query": "原始查询",
        "results": [
            {
                "id": "chunk_001",
                "text": "完整内容文本",
                "source_file": "来源文件.md",
                "headings": ["章节1", "小节1.1"],
                "similarity": 0.95
            }
        ]
    }
"""

MATERIAL_BATCH_SEARCH_DESC = """
批量语义搜索课件

一次性搜索多个查询，减少API调用次数。

Args:
    course_name: 课程名称
    queries: 查询文本列表
    top_k: 每个查询返回数量（默认3）
    
Returns:
    {
        "query1": [results...],
        "query2": [results...]
    }
"""

# ========== 课程管理工具（1个）==========

LIST_AVAILABLE_COURSES_DESC = """
列出所有可用的课程

返回系统中所有已处理的课程列表及其数据就绪状态，供Agent在调用其他工具前获取有效的course_name值。

数据就绪状态说明：
- has_video_index: 视频索引是否就绪（.cache/{course_name}/facts/segments.jsonl 存在）
- has_material_rag: 课件RAG是否就绪（vector_db/course_materials/{course_name}/chroma.sqlite3 存在）

Args:
    （无参数）
    
Returns:
    {
        "courses": [
            {
                "name": "A4.2_fine_tuning",
                "has_video_index": true,
                "has_material_rag": true
            },
            {
                "name": "prompt_engineering",
                "has_video_index": true,
                "has_material_rag": false
            }
        ]
    }
    
使用建议:
    1. 在调用其他需要course_name参数的工具前，先调用此工具获取有效课程列表
    2. 使用 difflib 进行模糊匹配用户输入的课程名
    3. 检查 has_video_index 和 has_material_rag 确认数据就绪状态
    4. 如果用户指定的课程不在列表中，提示用户选择有效课程
"""
