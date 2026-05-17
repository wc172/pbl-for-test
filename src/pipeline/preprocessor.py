"""
模块1: 输入预处理模块

将输入视频/音频转换为标准格式音频（WAV, 16kHz, 16bit PCM, 单声道），
适配 Paraformer 语音识别模型要求。

处理流程:
1. 检测文件格式及音频编码信息（使用 ffprobe）
2. 统一提取/转换为 WAV 格式（16kHz, 16bit PCM, 单声道）
3. 删除原始文件节省空间（可选）
4. 创建断点标记

输出格式:
- courses/{course_name}/audio.wav (16kHz, 16bit PCM, mono)
- .cache/{course_name}/step_01_audio.done
"""

import os
import json
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.utils.checkpoint import CheckpointManager, compute_file_hash

logger = logging.getLogger(__name__)


@dataclass
class AudioFormatInfo:
    """音频格式信息"""
    file_type: str           # 'video' | 'audio' | 'unsupported'
    codec: Optional[str]     # 音频编码
    sample_rate: Optional[int]   # 当前采样率
    channels: Optional[int]      # 声道数
    duration: Optional[float]    # 时长（秒）
    need_resample: bool      # 是否需要重采样


class AudioPreprocessor:
    """
    音频预处理器 - 适配 Paraformer
    
    统一输出: WAV格式, 16kHz采样率, 16bit PCM, 单声道
    """
    
    # 支持的输入格式
    SUPPORTED_VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    SUPPORTED_AUDIO = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'}
    
    # 目标输出参数（Paraformer要求）
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_BIT_DEPTH = 16
    TARGET_CODEC = "pcm_s16le"
    
    def __init__(self, course_name: str, config: Dict[str, Any] = None):
        """
        初始化音频预处理器
        
        Args:
            course_name: 课程名称
            config: 配置字典，可选覆盖默认配置
        """
        self.course_name = course_name
        self.config = config or {}
        
        # 路径设置
        self.course_dir = Path(f"courses/{course_name}")
        self.output_path = self.course_dir / "audio.wav"
        
        # 检查点管理器
        self.checkpoint_mgr = CheckpointManager(course_name)
        self.checkpoint_name = "step_01_audio"
        
        # FFmpeg 路径
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg/bin/ffmpeg.exe')
        self.ffprobe_path = self.config.get('ffprobe_path', 'ffmpeg/bin/ffprobe.exe')
        
        # 确保目录存在
        self.course_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_format(self, input_path: str) -> AudioFormatInfo:
        """
        步骤1: 检测文件格式及音频编码信息
        
        使用 ffprobe 获取详细的媒体信息
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            AudioFormatInfo: 音频格式信息
        """
        input_path = Path(input_path)
        
        # 检查文件存在性
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 根据扩展名判断类型
        ext = input_path.suffix.lower()
        
        if ext in self.SUPPORTED_VIDEO:
            file_type = "video"
        elif ext in self.SUPPORTED_AUDIO:
            file_type = "audio"
        else:
            return AudioFormatInfo(
                file_type="unsupported",
                codec=None,
                sample_rate=None,
                channels=None,
                duration=None,
                need_resample=True
            )
        
        # 使用 ffprobe 获取详细音频信息
        try:
            cmd = [
                self.ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(input_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                logger.warning(f"ffprobe 执行失败: {result.stderr}")
                # 降级处理：仅根据扩展名判断
                return AudioFormatInfo(
                    file_type=file_type,
                    codec=None,
                    sample_rate=None,
                    channels=None,
                    duration=None,
                    need_resample=True
                )
            
            probe_data = json.loads(result.stdout)
            
            # 查找音频流
            audio_stream = None
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if audio_stream:
                sample_rate = int(audio_stream.get('sample_rate', 0))
                channels = audio_stream.get('channels')
                codec = audio_stream.get('codec_name')
                duration = float(audio_stream.get('duration', 0))
                
                # 判断是否需要重采样
                need_resample = (
                    sample_rate != self.TARGET_SAMPLE_RATE or
                    channels != self.TARGET_CHANNELS or
                    (file_type == "audio" and ext != '.wav')
                )
                
                return AudioFormatInfo(
                    file_type=file_type,
                    codec=codec,
                    sample_rate=sample_rate,
                    channels=channels,
                    duration=duration,
                    need_resample=need_resample
                )
            else:
                # 视频文件但没有音频流
                if file_type == "video":
                    raise ValueError(f"视频文件没有音频流: {input_path}")
                
                return AudioFormatInfo(
                    file_type=file_type,
                    codec=None,
                    sample_rate=None,
                    channels=None,
                    duration=None,
                    need_resample=True
                )
                
        except Exception as e:
            logger.warning(f"ffprobe 解析失败: {e}")
            return AudioFormatInfo(
                file_type=file_type,
                codec=None,
                sample_rate=None,
                channels=None,
                duration=None,
                need_resample=True
            )
    
    def extract_audio(self, input_path: str, format_info: AudioFormatInfo) -> str:
        """
        步骤2: 统一提取/转换为 WAV 格式
        
        - 视频文件: 提取音频流 -> WAV
        - 音频文件: 转码/重采样 -> WAV (16kHz, 16bit, mono)
        
        Args:
            input_path: 输入文件路径
            format_info: 音频格式信息
            
        Returns:
            str: 输出文件路径
        """
        input_path = Path(input_path)
        
        # 构建 FFmpeg 命令
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",  # 减少输出噪音
            "-y",  # 覆盖输出文件
            "-i", str(input_path),
            # 音频参数（Paraformer要求）
            "-ar", str(self.TARGET_SAMPLE_RATE),  # 采样率 16kHz
            "-ac", str(self.TARGET_CHANNELS),     # 单声道
            "-acodec", self.TARGET_CODEC,         # 16bit PCM
            "-f", "wav",                          # 输出格式
            str(self.output_path)
        ]
        
        # 视频文件：禁用视频流
        if format_info.file_type == "video":
            cmd.insert(5, "-vn")  # 在 -i 参数后插入 -vn
        
        logger.info(f"执行 FFmpeg 命令: {' '.join(cmd)}")
        
        # 执行转换
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg 转换失败: {result.stderr}")
            
            if not self.output_path.exists():
                raise RuntimeError("FFmpeg 未生成输出文件")
            
            logger.info(f"音频提取完成: {self.output_path}")
            return str(self.output_path)
            
        except Exception as e:
            # 清理可能的不完整输出
            if self.output_path.exists():
                self.output_path.unlink()
            raise RuntimeError(f"音频提取失败: {e}")
    
    def cleanup(self, input_path: str) -> bool:
        """
        步骤3: 删除原始文件节省空间
        
        Args:
            input_path: 原始输入文件路径
            
        Returns:
            bool: 是否成功删除
        """
        delete_original = self.config.get('delete_original', True)
        
        if not delete_original:
            logger.info("配置设置为保留原始文件")
            return False
        
        input_path = Path(input_path)
        
        # 安全校验：确认输出文件存在且非空
        if not self.output_path.exists():
            logger.warning("输出文件不存在，跳过删除原始文件")
            return False
        
        if self.output_path.stat().st_size == 0:
            logger.warning("输出文件为空，跳过删除原始文件")
            return False
        
        try:
            input_path.unlink()
            logger.info(f"已删除原始文件: {input_path}")
            return True
        except Exception as e:
            logger.warning(f"删除原始文件失败: {e}")
            return False
    
    def create_checkpoint(self, input_path: str, format_info: AudioFormatInfo, 
                         input_hash: str = None) -> Dict[str, Any]:
        """
        步骤4: 创建断点标记
        
        Args:
            input_path: 原始输入文件路径
            format_info: 音频格式信息
            input_hash: 预先计算的原始文件哈希（如果文件已被删除）
            
        Returns:
            Dict: 断点数据
        """
        # 如果未提供哈希且文件仍存在，计算哈希
        if input_hash is None:
            if Path(input_path).exists():
                input_hash = compute_file_hash(input_path)
            else:
                input_hash = "unknown"
        
        checkpoint_data = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "input_hash": input_hash,
            "input_path": str(input_path),
            "output_file": "audio.wav",
            "format": "wav",
            "codec": self.TARGET_CODEC,
            "sample_rate": self.TARGET_SAMPLE_RATE,
            "channels": self.TARGET_CHANNELS,
            "bit_depth": self.TARGET_BIT_DEPTH,
            "original_format": {
                "file_type": format_info.file_type,
                "codec": format_info.codec,
                "sample_rate": format_info.sample_rate,
                "channels": format_info.channels,
                "duration": format_info.duration
            },
            "output_size": self.output_path.stat().st_size if self.output_path.exists() else 0
        }
        
        self.checkpoint_mgr.save(self.checkpoint_name, checkpoint_data)
        logger.info(f"断点标记已创建: {self.checkpoint_mgr.checkpoint_dir}/{self.checkpoint_name}.checkpoint")
        
        return checkpoint_data
    
    def is_completed(self, input_path: str) -> bool:
        """
        检查是否已完成处理
        
        通过比对输入文件哈希判断是否需要重新处理
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            bool: 是否已完成且输入未变更
        """
        checkpoint = self.checkpoint_mgr.load(self.checkpoint_name)
        
        if not checkpoint:
            return False
        
        if not self.output_path.exists():
            return False
        
        # 检查输入文件是否变更
        current_hash = compute_file_hash(input_path)
        saved_hash = checkpoint.get('data', {}).get('input_hash')
        
        if current_hash != saved_hash:
            logger.info("输入文件已变更，需要重新处理")
            return False
        
        logger.info("预处理已完成且输入未变更")
        return True
    
    def process(self, input_path: str) -> Dict[str, Any]:
        """
        执行完整的预处理流程
        
        流程: detect -> extract -> cleanup -> checkpoint
        
        Args:
            input_path: 输入文件路径（视频或音频）
            
        Returns:
            Dict: 处理结果，包含 output_path 和 checkpoint 信息
        """
        input_path = Path(input_path)
        
        logger.info(f"开始音频预处理: {input_path}")
        
        # 检查是否已完成
        if self.is_completed(str(input_path)):
            checkpoint = self.checkpoint_mgr.load(self.checkpoint_name)
            logger.info(f"使用已存在的输出: {self.output_path}")
            return {
                "status": "skipped",
                "output_path": str(self.output_path),
                "checkpoint": checkpoint.get('data', {})
            }
        
        # 步骤1: 检测文件格式
        logger.info("步骤1: 检测文件格式...")
        format_info = self.detect_format(str(input_path))
        
        if format_info.file_type == "unsupported":
            raise ValueError(
                f"不支持的文件格式: {input_path.suffix}. "
                f"支持的视频: {self.SUPPORTED_VIDEO}, "
                f"支持的音频: {self.SUPPORTED_AUDIO}"
            )
        
        logger.info(
            f"检测到 {format_info.file_type} 文件, "
            f"编码: {format_info.codec}, "
            f"采样率: {format_info.sample_rate}, "
            f"声道: {format_info.channels}, "
            f"需要重采样: {format_info.need_resample}"
        )
        
        # 步骤2: 提取/转换音频
        logger.info("步骤2: 提取/转换为 WAV 格式...")
        output_path = self.extract_audio(str(input_path), format_info)
        
        # ⚠️ 在删除原始文件前计算哈希值（用于断点续传检查）
        input_hash = compute_file_hash(str(input_path))
        
        # 步骤3: 清理原始文件
        logger.info("步骤3: 清理原始文件...")
        cleanup_success = self.cleanup(str(input_path))
        
        # 步骤4: 创建断点（传入预先计算的哈希值）
        logger.info("步骤4: 创建断点标记...")
        checkpoint_data = self.create_checkpoint(str(input_path), format_info, input_hash)
        
        logger.info(f"音频预处理完成: {output_path}")
        
        return {
            "status": "completed",
            "output_path": output_path,
            "checkpoint": checkpoint_data,
            "cleanup": cleanup_success,
            "format_info": {
                "file_type": format_info.file_type,
                "codec": format_info.codec,
                "original_sample_rate": format_info.sample_rate,
                "original_channels": format_info.channels,
                "duration": format_info.duration
            }
        }
    
    @staticmethod
    def get_output_path(course_name: str) -> str:
        """
        静态方法：获取输出文件路径
        
        其他模块可直接调用此方法获取音频路径，无需实例化
        
        Args:
            course_name: 课程名称
            
        Returns:
            str: 音频文件路径
        """
        return f"courses/{course_name}/audio.wav"


# 便捷函数
def preprocess_audio(course_name: str, input_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    便捷的音频预处理函数
    
    Args:
        course_name: 课程名称
        input_path: 输入文件路径
        config: 配置字典
        
    Returns:
        Dict: 处理结果
    """
    preprocessor = AudioPreprocessor(course_name, config)
    return preprocessor.process(input_path)


def is_preprocessed(course_name: str) -> bool:
    """
    检查课程是否已完成音频预处理
    
    供其他模块调用，快速检查音频文件是否已就绪
    
    Args:
        course_name: 课程名称
        
    Returns:
        bool: 是否已完成预处理
    """
    audio_path = Path(f"courses/{course_name}/audio.wav")
    checkpoint_path = Path(f".cache/{course_name}/step_01_audio.checkpoint")
    return audio_path.exists() and checkpoint_path.exists()
