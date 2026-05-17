"""
模块2: 离线转录模块

使用 FunASR/Paraformer 进行离线语音识别，生成带时间戳的SRT字幕，支持断点续传。

输入: courses/{course_name}/audio.wav (16kHz, 16bit PCM)
输出: 
  - courses/{course_name}/transcript.srt
  - .cache/{course_name}/transcription_result.json
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.utils.checkpoint import CheckpointManager, compute_file_hash

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """转录片段"""
    index: int
    start_ms: int
    end_ms: int
    text: str
    
    def to_srt_format(self) -> str:
        """转换为SRT格式"""
        start = self._ms_to_srt(self.start_ms)
        end = self._ms_to_srt(self.end_ms)
        return f"{self.index}\n{start} --> {end}\n{self.text}\n"
    
    @staticmethod
    def _ms_to_srt(ms: int) -> str:
        """毫秒转SRT时间"""
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class AudioTranscriber:
    """离线转录器 - 使用 FunASR/Paraformer"""
    
    def __init__(self, course_name: str, config: Dict[str, Any] = None):
        self.course_name = course_name
        self.config = config or {}
        
        # 路径
        self.audio_path = Path(f"courses/{course_name}/audio.wav")
        self.output_srt = Path(f"courses/{course_name}/transcript.srt")
        self.output_raw = Path(f".cache/{course_name}/transcription_result.json")
        
        # 检查点
        self.checkpoint_mgr = CheckpointManager(course_name)
        
        # 模型配置
        self.model_name = self.config.get('model') or \
                         os.getenv('ASR_MODEL_PATH', 'paraformer-zh')
        self.device = self.config.get('device', 'cuda')
        self.batch_size_s = self.config.get('batch_size_s', 300)
        
        # 热词配置：支持直接字符串或从文件加载
        self.hotword = self._load_hotwords()
        
        # 内部状态
        self.model = None
        self._model_loaded = False
    
    def _load_hotwords(self) -> str:
        """
        加载热词，支持两种配置方式：
        1. hotword: 直接传入逗号分隔的热词字符串
        2. hotword_file: 从文件加载热词（支持权重）
        
        文件格式（每行一个热词，可选权重）：
            热词1 权重
            热词2 权重
            # 注释行以#开头
        
        Returns:
            逗号分隔的热词字符串，供FunASR使用
        """
        # 优先使用直接配置的热词字符串
        direct_hotword = self.config.get('hotword', '')
        if direct_hotword:
            logger.info(f"使用直接配置的热词: {len(direct_hotword.split(','))} 个")
            return direct_hotword
        
        # 尝试从文件加载
        hotword_file = self.config.get('hotword_file', '')
        if not hotword_file:
            return ''
        
        hotword_path = Path(hotword_file)
        if not hotword_path.exists():
            # 尝试相对路径
            hotword_path = Path(f"config/{hotword_file}")
            if not hotword_path.exists():
                logger.warning(f"热词文件不存在: {hotword_file}")
                return ''
        
        try:
            hotwords = []
            with open(hotword_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释行
                    if not line or line.startswith('#'):
                        continue
                    
                    # 解析热词和权重
                    parts = line.split()
                    if len(parts) >= 1:
                        word = parts[0]
                        weight = parts[1] if len(parts) >= 2 else None
                        
                        # FunASR Python API 暂不支持权重，只取热词
                        hotwords.append(word)
            
            if hotwords:
                result = ','.join(hotwords)
                logger.info(f"从文件加载热词: {hotword_path} ({len(hotwords)} 个)")
                # 显示部分热词用于调试
                preview = ', '.join(hotwords[:5])
                if len(hotwords) > 5:
                    preview += f", ...({len(hotwords)-5} more)"
                logger.info(f"热词预览: {preview}")
                return result
            else:
                logger.warning(f"热词文件为空: {hotword_path}")
                return ''
                
        except Exception as e:
            logger.error(f"加载热词文件失败: {e}")
            return ''
    
    def _load_model(self):
        """加载模型（包含ASR、VAD、标点）"""
        if self._model_loaded:
            return
        
        # 设置 ffmpeg 路径（如果配置中有）
        ffmpeg_path = self.config.get('ffmpeg_path')
        if ffmpeg_path:
            import os
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            if os.path.exists(ffmpeg_path):
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                logger.info(f"添加 ffmpeg 到 PATH: {ffmpeg_dir}")
        
        from funasr import AutoModel
        from pathlib import Path
        
        # 获取ASR模型绝对路径
        model_path = self.model_name
        if model_path.startswith('models/'):
            abs_path = Path(model_path).resolve()
            if abs_path.exists():
                model_path = str(abs_path)
                is_local = True
                logger.info(f"使用本地ASR模型: {model_path}")
            else:
                is_local = False
                logger.warning(f"本地模型路径不存在: {abs_path}，将尝试在线下载")
        else:
            is_local = model_path.startswith('models/') or '/' in model_path or '\\' in model_path
        
        model_kwargs = {
            "model": model_path, 
            "device": self.device,
            "disable_pbar": False,  # 启用FunASR内置进度条
            "disable_log": True,    # 关闭内部日志
        }
        if not is_local:
            model_kwargs["model_revision"] = self.config.get('model_revision', 'v2.0.4')
        
        # 添加VAD模型（用于长音频分割）
        vad_model = self.config.get('vad_model', 'fsmn-vad')
        if vad_model:
            model_kwargs["vad_model"] = vad_model
            model_kwargs["vad_kwargs"] = {"max_single_segment_time": 60000}  # 60秒最大分段
            logger.info(f"启用VAD模型: {vad_model}")
        
        # 添加标点模型（关键！）
        punc_model = self.config.get('punc_model', 'ct-punc')
        if punc_model:
            model_kwargs["punc_model"] = punc_model
            logger.info(f"启用标点模型: {punc_model}")
        
        self.model = AutoModel(**model_kwargs)
        self._model_loaded = True
        logger.info("模型加载完成（ASR+VAD+PUNC）")
    
    def transcribe(self) -> str:
        """
        执行转录（简化版：直接处理整个音频）
        
        检查：
        1. SRT文件已存在且有效 -> 直接返回
        2. 否则 -> 完整转录
        
        Returns:
            str: SRT文件路径
        """
        # 验证音频
        if not self.audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {self.audio_path}")
        
        import soundfile as sf
        info = sf.info(str(self.audio_path))
        duration_ms = int(info.duration * 1000)
        logger.info(f"音频: {info.duration:.1f}s ({info.duration/60:.1f}分钟), {info.samplerate}Hz")
        
        # 检查SRT文件是否已存在
        if self.output_srt.exists() and self.output_srt.stat().st_size > 0:
            with open(self.output_srt, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if len(content) > 100 and '-->' in content:
                logger.info(f"转录已完成: {self.output_srt}")
                return str(self.output_srt)
            else:
                logger.warning("SRT文件内容异常，重新转录")
        
        all_segments = []
        
        # 加载模型
        self._load_model()
        
        # 最简化版本：直接传入文件路径，使用FunASR内置进度条
        logger.info(f"开始转录: {self.audio_path}")
        
        generate_kwargs = {
            "batch_size_s": self.batch_size_s,
            "beam_size": self.config.get('beam_size', 10),
            "decoding_ctc_weight": self.config.get('decoding_ctc_weight', 0.0),
            "decoding_mode": self.config.get('decoding_mode', 'model1'),
            "sentence_timestamp": True,  # 关键：获取句子级别时间戳
        }
        
        if self.hotword:
            generate_kwargs["hotword"] = self.hotword
        
        # 直接传入音频文件路径
        result = self.model.generate(
            input=str(self.audio_path),
            **generate_kwargs
        )
        
        # 解析结果
        all_segments = []
        if result and len(result) > 0:
            res = result[0]
            
            # 使用 sentence_info 获取句子级别结果
            if "sentence_info" in res:
                logger.info(f"使用 sentence_info 生成字幕")
                for i, sent in enumerate(res["sentence_info"]):
                    seg = TranscriptionSegment(
                        index=i + 1,
                        start_ms=int(sent["start"]),
                        end_ms=int(sent["end"]),
                        text=sent["text"]
                    )
                    all_segments.append(seg)
            else:
                # 降级：使用普通时间戳
                text = res.get("text", "")
                timestamps = res.get("timestamp", [])
                logger.info(f"转录完成: {len(text)} 字符, {len(timestamps)} 个时间戳")
                sentences = self._split_sentences(text)
                all_segments = self._align_absolute(sentences, timestamps, 0)
        
        # 保存结果
        self._save_results(all_segments, duration_ms)
        
        logger.info(f"转录完成: {len(all_segments)} 片段")
        return str(self.output_srt)
    
    @staticmethod
    def _ms_to_time(ms: int) -> str:
        """毫秒转为 HH:MM:SS 格式"""
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def _cleanup_cache(self):
        """转录成功后清理中间缓存文件"""
        try:
            # 删除检查点文件（断点续传用的临时文件）
            self.checkpoint_mgr.delete("step_02_transcription")
            logger.info("已清理转录缓存")
        except Exception:
            pass  # 清理失败不影响主流程
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割文本为句子"""
        import re
        pattern = r'([^。！？.!?]+[。！？.!?])'
        sentences = re.findall(pattern, text)
        remaining = re.sub(pattern, '', text).strip()
        if remaining:
            sentences.append(remaining)
        return [s.strip() for s in sentences if s.strip()]
    
    def _align_absolute(self, sentences: List[str], timestamps: List[List[int]], 
                         base_index: int) -> List[TranscriptionSegment]:
        """
        对齐文本和时间戳（Paraformer返回的是绝对时间戳）
        
        当传入文件路径时，Paraformer返回的时间戳是相对于音频起始的绝对时间
        """
        segments = []
        for i, sent in enumerate(sentences):
            if i < len(timestamps):
                ts = timestamps[i]
                abs_start_ms = int(ts[0]) if len(ts) >= 1 else 0
                abs_end_ms = int(ts[1]) if len(ts) >= 2 else abs_start_ms + 5000
                
                seg = TranscriptionSegment(
                    index=base_index + i + 1,
                    start_ms=abs_start_ms,
                    end_ms=abs_end_ms,
                    text=sent
                )
                segments.append(seg)
        return segments
    
    def _save_checkpoint(self, processed_ms: int, segments: List[TranscriptionSegment],
                        total_ms: int, audio_hash: str, completed: bool = False):
        """保存断点"""
        data = {
            "status": "completed" if completed else "processing",
            "timestamp": datetime.now().isoformat(),
            "audio_hash": audio_hash,
            "processed_ms": processed_ms,
            "total_ms": total_ms,
            "progress": round(processed_ms / total_ms * 100, 2) if total_ms > 0 else 0,
            "segments": [{"index": s.index, "start_ms": s.start_ms, 
                         "end_ms": s.end_ms, "text": s.text} for s in segments]
        }
        self.checkpoint_mgr.save("step_02_transcription", data)
    
    def _save_results(self, segments: List[TranscriptionSegment], duration_ms: int):
        """保存结果文件"""
        # 保存SRT
        srt_content = "\n\n".join([s.to_srt_format() for s in segments])
        with open(self.output_srt, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        logger.info(f"SRT已保存: {self.output_srt}")
        
        # 保存JSON
        data = {
            "text": " ".join([s.text for s in segments]),
            "segments": [{"index": s.index, "start_ms": s.start_ms,
                         "end_ms": s.end_ms, "text": s.text} for s in segments],
            "duration_ms": duration_ms
        }
        with open(self.output_raw, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON已保存: {self.output_raw}")


# 便捷函数
def transcribe_audio(course_name: str, config: Dict[str, Any] = None) -> str:
    """便捷函数：执行转录"""
    transcriber = AudioTranscriber(course_name, config)
    return transcriber.transcribe()
