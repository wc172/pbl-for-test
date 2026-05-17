"""
SRT字幕解析工具

提供SRT文件的解析、生成和转换功能
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SRTEntry:
    """SRT条目"""
    index: int
    start_ms: int
    end_ms: int
    text: str
    
    def to_srt_format(self) -> str:
        """转换为SRT格式字符串"""
        # 注意：末尾不加换行符，由 generate_srt 统一处理分隔
        return f"{self.index}\n{ms_to_srt_time(self.start_ms)} --> {ms_to_srt_time(self.end_ms)}\n{self.text}"


def ms_to_srt_time(ms: int) -> str:
    """
    将毫秒转换为SRT时间格式: HH:MM:SS,mmm
    
    Args:
        ms: 毫秒
        
    Returns:
        str: SRT时间格式字符串
    """
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def srt_time_to_ms(time_str: str) -> int:
    """
    将SRT时间格式转换为毫秒
    
    Args:
        time_str: SRT时间格式字符串 (HH:MM:SS,mmm 或 HH:MM:SS.mmm)
        
    Returns:
        int: 毫秒
    """
    time_str = time_str.strip().replace('.', ',')
    
    # 匹配 HH:MM:SS,mmm
    pattern = r'(\d+):(\d+):(\d+)[,\.](\d+)'
    match = re.match(pattern, time_str)
    
    if not match:
        raise ValueError(f"无效的时间格式: {time_str}")
    
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    milliseconds = int(match.group(4).ljust(3, '0')[:3])  # 确保3位
    
    return hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds


def parse_srt(content: str) -> List[SRTEntry]:
    """
    解析SRT内容
    
    Args:
        content: SRT格式文本
        
    Returns:
        List[SRTEntry]: SRT条目列表
    """
    entries = []
    
    # 按空行分割条目
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        try:
            # 第一行是序号
            index = int(lines[0].strip())
            
            # 第二行是时间范围
            time_line = lines[1].strip()
            time_match = re.match(r'(.+?)\s*-->\s*(.+)', time_line)
            
            if not time_match:
                continue
            
            start_ms = srt_time_to_ms(time_match.group(1))
            end_ms = srt_time_to_ms(time_match.group(2))
            
            # 剩余行是文本
            text = '\n'.join(lines[2:]).strip()
            
            entries.append(SRTEntry(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text
            ))
            
        except Exception as e:
            # 跳过无效条目
            continue
    
    return entries


def parse_srt_file(file_path: str) -> List[SRTEntry]:
    """
    解析SRT文件
    
    Args:
        file_path: SRT文件路径
        
    Returns:
        List[SRTEntry]: SRT条目列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_srt(content)


def generate_srt(entries: List[SRTEntry]) -> str:
    """
    生成SRT内容
    
    Args:
        entries: SRT条目列表
        
    Returns:
        str: SRT格式文本
    """
    return '\n\n'.join([entry.to_srt_format() for entry in entries])


def save_srt(entries: List[SRTEntry], file_path: str):
    """
    保存SRT文件
    
    Args:
        entries: SRT条目列表
        file_path: 输出文件路径
    """
    content = generate_srt(entries)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def srt_entries_to_plain_text(entries: List[SRTEntry], include_timestamps: bool = False) -> str:
    """
    将SRT条目转换为纯文本
    
    Args:
        entries: SRT条目列表
        include_timestamps: 是否包含时间戳
        
    Returns:
        str: 纯文本
    """
    if include_timestamps:
        return '\n'.join([
            f"[{ms_to_srt_time(e.start_ms)}] {e.text}"
            for e in entries
        ])
    else:
        return '\n'.join([e.text for e in entries])


def merge_srt_entries(entries: List[SRTEntry], 
                     gap_threshold_ms: int = 500) -> List[SRTEntry]:
    """
    合并相邻的SRT条目（如果时间间隔小于阈值）
    
    Args:
        entries: SRT条目列表
        gap_threshold_ms: 合并阈值（毫秒）
        
    Returns:
        List[SRTEntry]: 合并后的条目列表
    """
    if not entries:
        return []
    
    merged = [entries[0]]
    
    for entry in entries[1:]:
        last = merged[-1]
        
        # 如果时间间隔小于阈值，合并
        if entry.start_ms - last.end_ms <= gap_threshold_ms:
            last.end_ms = entry.end_ms
            last.text += ' ' + entry.text
        else:
            # 重新编号
            entry.index = len(merged) + 1
            merged.append(entry)
    
    # 重新编号
    for i, entry in enumerate(merged, 1):
        entry.index = i
    
    return merged
