"""
流水线主控模块

协调各步骤的执行，处理依赖关系和断点恢复。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml

# 导入各步骤模块
from .course_rag import CourseRAGBuilder
from .preprocessor import AudioPreprocessor
from .transcriber import AudioTranscriber, transcribe_audio
from .corrector import TranscriptionCorrector
from .video_store import SegmentBuilder, VideoStorage


def load_config() -> Dict[str, Any]:
    """
    加载配置文件 (config/pipeline.yaml)
    
    将相对路径转换为绝对路径（基于项目根目录）
    """
    # 获取项目根目录 (脚本所在目录的父目录)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    config_path = project_root / "config" / "pipeline.yaml"
    
    if not config_path.exists():
        print(f"警告: 配置文件不存在: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 将相对路径转换为绝对路径
    path_keys = ['ffmpeg_path', 'ffprobe_path']
    
    # audio 部分的路径
    if 'audio' in config:
        for key in path_keys:
            if key in config['audio'] and config['audio'][key]:
                path = Path(config['audio'][key])
                if not path.is_absolute():
                    config['audio'][key] = str(project_root / path)
    
    # transcription 部分的路径
    if 'transcription' in config:
        for key in path_keys:
            if key in config['transcription'] and config['transcription'][key]:
                path = Path(config['transcription'][key])
                if not path.is_absolute():
                    config['transcription'][key] = str(project_root / path)
    
    return config


class Pipeline:
    """处理流水线"""
    
    def __init__(self, course_name: str, config: Dict[str, Any] = None):
        self.course_name = course_name
        self.config = config or {}
        
    def step0_build_course_rag(self):
        """步骤0: 构建课件RAG"""
        print(f"[Step 0] Building course RAG for: {self.course_name}")
        builder = CourseRAGBuilder(self.course_name, self.config)
        builder.build()
        print("[Step 0] Completed")
    
    def step1_preprocess(self, input_path: str = None):
        """步骤1: 输入预处理"""
        print(f"[Step 1] Preprocessing for: {self.course_name}")
        preprocessor = AudioPreprocessor(self.course_name, self.config)
        if input_path is None:
            # 自动查找输入文件（任意名称）
            course_dir = Path(f"courses/{self.course_name}")
            for ext in [".mp4", ".mp3", ".wav", ".m4a"]:
                candidates = list(course_dir.glob(f"*{ext}"))
                if candidates:
                    input_path = str(candidates[0])  # 使用找到的第一个文件
                    break
        if not input_path:
            raise FileNotFoundError(f"No input file found for course: {self.course_name}")
        preprocessor.process(input_path)
        print("[Step 1] Completed")
    
    def step2_transcribe(self):
        """步骤2: 离线转录"""
        print(f"[Step 2] Transcribing for: {self.course_name}")
        transcriber = AudioTranscriber(self.course_name, self.config)
        transcriber.transcribe()
        print("[Step 2] Completed")
    
    def step3_correct(self):
        """步骤3: LLM校对"""
        print(f"[Step 3] Correcting transcription for: {self.course_name}")
        corrector = TranscriptionCorrector(self.course_name, self.config)
        corrector.correct(force_reprocess=False)
        print("[Step 3] Completed")
    
    def step4_build_video_store(self):
        """步骤4: 构建视频内容存储"""
        print(f"[Step 4] Building video content store for: {self.course_name}")
        srt_path = Path(f"courses/{self.course_name}/transcript_corrected.srt")
        if not srt_path.exists():
            raise FileNotFoundError(f"校正后的SRT文件不存在: {srt_path}")
        
        # 构建Segments
        builder = SegmentBuilder(self.course_name, self.config)
        segments = builder.build(srt_path)
        print(f"  构建完成: {len(segments)} 个Segments")
        
        # 保存存储
        storage = VideoStorage(self.course_name)
        storage.save_segments(segments)
        print(f"  存储完成: {storage.base_dir}")
        print("[Step 4] Completed")
    
    def run_all(self, input_path: str = None):
        """运行完整流程"""
        self.step0_build_course_rag()
        self.step1_preprocess(input_path)
        self.step2_transcribe()
        self.step3_correct()
        self.step4_build_video_store()
        print(f"[Pipeline] All steps completed for: {self.course_name}")


def main():
    parser = argparse.ArgumentParser(description="视频转录流水线")
    parser.add_argument("--course", required=True, help="课程名称")
    parser.add_argument("--step", choices=["0", "1", "2", "3", "4", "all"], 
                       default="all", help="执行步骤")
    parser.add_argument("--input", help="输入文件路径（仅步骤1需要）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    # 确保 ffmpeg 目录在 PATH 中（用于 funasr 等库调用）
    ffmpeg_path = config.get('transcription', {}).get('ffmpeg_path') or \
                  config.get('audio', {}).get('ffmpeg_path')
    if ffmpeg_path:
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        if os.path.exists(ffmpeg_dir):
            current_path = os.environ.get('PATH', '')
            if ffmpeg_dir not in current_path:
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
                print(f"[Config] 添加 ffmpeg 到 PATH: {ffmpeg_dir}")
    
    pipeline = Pipeline(args.course, config)
    
    if args.step == "0":
        pipeline.step0_build_course_rag()
    elif args.step == "1":
        pipeline.step1_preprocess(args.input)
    elif args.step == "2":
        pipeline.step2_transcribe()
    elif args.step == "3":
        pipeline.step3_correct()
    elif args.step == "4":
        pipeline.step4_build_video_store()
    else:
        pipeline.run_all(args.input)


if __name__ == "__main__":
    main()
