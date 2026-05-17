"""
示例: 音频预处理

将视频或音频文件转换为 Paraformer 要求的 WAV 格式 (16kHz, 16bit PCM, mono)
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.preprocessor import AudioPreprocessor, preprocess_audio


def main():
    parser = argparse.ArgumentParser(description="音频预处理示例")
    parser.add_argument("--course", required=True, help="课程名称")
    parser.add_argument("--input", required=True, help="输入文件路径（视频或音频）")
    parser.add_argument("--keep-original", action="store_true", help="保留原始文件")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        "ffmpeg_path": "ffmpeg/bin/ffmpeg.exe",
        "ffprobe_path": "ffmpeg/bin/ffprobe.exe",
        "delete_original": not args.keep_original
    }
    
    # 执行预处理
    try:
        result = preprocess_audio(
            course_name=args.course,
            input_path=args.input,
            config=config
        )
        
        print("\n" + "="*50)
        print("预处理结果")
        print("="*50)
        print(f"状态: {result['status']}")
        print(f"输出文件: {result['output_path']}")
        
        if result['status'] == 'completed':
            print(f"原始格式: {result['format_info']['file_type']}")
            print(f"原始编码: {result['format_info']['codec']}")
            print(f"原始采样率: {result['format_info']['original_sample_rate']} Hz")
            print(f"原始时长: {result['format_info']['duration']:.2f} 秒")
            print(f"清理原始文件: {'成功' if result['cleanup'] else '跳过/失败'}")
        
        print("="*50)
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
